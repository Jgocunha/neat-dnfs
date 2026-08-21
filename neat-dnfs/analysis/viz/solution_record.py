import re

_ID_RE = re.compile(r"solution\s+(\d+)\s*\[")
_FIT_PART_RE = re.compile(r"fit\.\:\s*([0-9eE\.\+\-]+).*?part\.\:\s*\(([^)]*)\)")
_SPEC_RE = re.compile(r"spec\.\:\s*(\d+)")
_ADJ_FIT_RE = re.compile(r"adj\.fit\.\:\s*([0-9eE\.\+\-]+)")
_AGE_PARENTS_RE = re.compile(r"age\:\s*(\d+)\s*,\s*(?:parents\s*)?\(\s*(\d+)\s*,\s*(\d+)\s*\)")
_GENOME_SIZE_RE = re.compile(r"genome\s*\(\s*(\d+)\s*field genes,\s*(\d+)\s*connection genes\s*\)")
_FIELD_GENES_BLOCK_RE = re.compile(r"field genes\s*\{([^}]*)\}")
_FIELD_GENE_ITEM_RE = re.compile(r"fg\s*\(id:\s*(\d+),\s*type:\s*(INPUT|OUTPUT|HIDDEN)\)")
_CONN_GENES_BLOCK_RE = re.compile(r"connection genes\s*\{([^}]*)\}")
_CONN_GENE_ITEM_RE = re.compile(
    r"cg\s*\(\s*(\d+)\s*-\s*(\d+)\s*,\s*innov:\s*(\d+)\s*,\s*enabled:\s*(true|false)\s*\)"
)


def find_solution_blob(line: str, start_marker: str = "solution ", start_index: int = 0) -> str | None:
    """Locate a 'solution <id> [ ... ]' blob within a larger line (searching from start_index)
    and return its full bracket-balanced contents, including the outer 'solution <id> [' .. ']'.

    Handles the nested [...] pairs inside a mutation block (e.g. '[fg 3 (...)]') by counting
    '[' / ']' depth rather than matching the first ']', which would truncate mid-genome.
    Returns None if no balanced blob is found.
    """
    start = line.find(start_marker, start_index)
    if start == -1:
        return None
    bracket_start = line.find("[", start)
    if bracket_start == -1:
        return None

    depth = 0
    for i in range(bracket_start, len(line)):
        c = line[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return line[start : i + 1]
    return None


def parse_solution_blob(blob: str) -> dict | None:
    """Parse one 'solution <id> [ fit.: ... ]' body (id and outer brackets may or may not be
    included; the caller need not strip them precisely) into a structured record:
        {
            "id": int,
            "fitness": float,
            "partials": list[float],
            "species_id": int,
            "adjusted_fitness": float,
            "age": int,
            "parent_ids": tuple[int, int],
            "num_field_genes": int,
            "num_connection_genes": int,
            "field_genes": list[dict],       # [{"id": int, "type": "INPUT"|"HIDDEN"|"OUTPUT"}, ...]
            "connection_genes": list[dict],  # [{"src": int, "tgt": int, "innov": int, "enabled": bool}, ...]
            "mutations_raw": str,
        }
    Returns None if the blob doesn't match the expected shape.

    Tolerant of an older log format seen in archived runs, which omits the 'parents' label
    before the parent-id tuple and drops the comma after 'part.: (...)' -- both are handled by
    the regexes above (see the "parents" alternation and the non-anchored field lookups).
    """
    id_m = _ID_RE.search(blob)
    fit_m = _FIT_PART_RE.search(blob)
    spec_m = _SPEC_RE.search(blob)
    adj_m = _ADJ_FIT_RE.search(blob)
    age_m = _AGE_PARENTS_RE.search(blob)
    size_m = _GENOME_SIZE_RE.search(blob)

    if not (id_m and fit_m and spec_m and adj_m and age_m and size_m):
        return None

    partials = []
    for token in fit_m.group(2).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            partials.append(float(token))
        except ValueError:
            pass

    field_genes = []
    fg_block = _FIELD_GENES_BLOCK_RE.search(blob)
    if fg_block:
        for fg_id, fg_type in _FIELD_GENE_ITEM_RE.findall(fg_block.group(1)):
            field_genes.append({"id": int(fg_id), "type": fg_type})

    connection_genes = []
    cg_block = _CONN_GENES_BLOCK_RE.search(blob)
    if cg_block:
        for src, tgt, innov, enabled in _CONN_GENE_ITEM_RE.findall(cg_block.group(1)):
            connection_genes.append(
                {
                    "src": int(src),
                    "tgt": int(tgt),
                    "innov": int(innov),
                    "enabled": enabled == "true",
                }
            )

    mutations_raw = ""
    mut_idx = blob.find("last mutations{")
    if mut_idx != -1:
        mut_end = blob.find("}]", mut_idx)
        if mut_end != -1:
            mutations_raw = blob[mut_idx + len("last mutations{") : mut_end].strip()

    return {
        "id": int(id_m.group(1)),
        "fitness": float(fit_m.group(1)),
        "partials": partials,
        "species_id": int(spec_m.group(1)),
        "adjusted_fitness": float(adj_m.group(1)),
        "age": int(age_m.group(1)),
        "parent_ids": (int(age_m.group(2)), int(age_m.group(3))),
        "num_field_genes": int(size_m.group(1)),
        "num_connection_genes": int(size_m.group(2)),
        "field_genes": field_genes,
        "connection_genes": connection_genes,
        "mutations_raw": mutations_raw,
    }


_STRUCT_TOGGLE_RE = re.compile(r"(toggle cg[^.\}]+\.)")


_STRUCT_ADDED_FG_RE = re.compile(r"\((added fg [^)]*)\)")


_STRUCT_ADDED_CG_RE = re.compile(r"\((added cg [^)]*)\)")


_GENE_MUT_BLOCK_RE = re.compile(r"\[([^\]]+)\]")


_GENE_HEAD_RE = re.compile(r"(?P<gene_type>[fc]g)\s+(?P<ref>[^\s(]+)\s*(?P<rest>.*)")


_INNER_MUT_RE = re.compile(r"\(([^)]+)\)")


_NOOP_MUT_RE = re.compile(r"(fg|cg)\s+\S+$")


def categorize_mutation(mut_str: str, gene_type: str = "") -> str:
    """
    Classify a mutation according to the taxonomy:

    Structural
        - toggle cg to enabled/disabled
        - added fg
        - added cg

    Parametrical mutations
      Field gene mutations
        Kernel mutations
            fg gk width
            fg gk amp
            fg gk amp glob
            fg mhk amp exc
            fg mhk width exc
            fg mhk amp inh
            fg mhk width inh
            fg mhk amp glob
            Type mutations: mhk to gk / gk to mhk
        Neural field mutations
            fg nf tau
            fg nf resting level
            fg nf rand

      Connection gene mutations
        Kernel mutations
            cg gk width
            cg gk amp
            cg gk amp glob
            cg mhk amp exc
            cg mhk width exc
            cg mhk amp inh
            cg mhk width inh
            cg mhk amp glob
            Type mutations: cg to gk / cg to mhk
        Signal mutations
            cg to excitatory / cg to inhibitory
    """
    s = (mut_str or "").lower().strip()

    # ---------- structural mutations ----------
    if s.startswith("toggle cg"):
        return "Structural – toggle connection enabled/disabled"
    if s.startswith("added fg"):
        return "Structural – add field gene"
    if s.startswith("added cg"):
        return "Structural – add connection gene"

    # ---------- field gene mutations ----------
    if gene_type == "fg":
        # neural-field parameters
        if "fg nf tau" in s:
            return "Field – neural field τ"
        if "fg nf rest. lvl" in s or "fg nf resting" in s:
            return "Field – neural field resting level"
        if "fg nf rand" in s:
            return "Field – neural field random reset"

        # type changes
        if "mhk to gk" in s:
            return "Field kernel – type mhk→gk"
        if "gk to mhk" in s:
            return "Field kernel – type gk→mhk"

        # Gaussian kernel params
        if "fg gk width" in s:
            return "Field kernel – gk width"
        if "fg gk amp. glob" in s:
            return "Field kernel – gk global amplitude"
        if "fg gk amp" in s:      # keep after "amp. glob" check
            return "Field kernel – gk amplitude"

        # Mexican-hat kernel params
        if "fg mhk amp. exc" in s:
            return "Field kernel – mhk exc amplitude"
        if "fg mhk width exc" in s:
            return "Field kernel – mhk exc width"
        if "fg mhk amp. inh" in s:
            return "Field kernel – mhk inh amplitude"
        if "fg mhk width inh" in s:
            return "Field kernel – mhk inh width"
        if "fg mhk amp. glob" in s:
            return "Field kernel – mhk global amplitude"

    # ---------- connection gene mutations ----------
    if gene_type == "cg":
        # signal type
        if "cg to excitatory" in s:
            return "Connection signal – to excitatory"
        if "cg to inhibitory" in s:
            return "Connection signal – to inhibitory"

        # type changes (kernel type)
        if "cg to gk" in s:
            return "Connection kernel – type →gk"
        if "cg to mhk" in s:
            return "Connection kernel – type →mhk"

        # Gaussian kernel params
        if "cg gk width" in s:
            return "Connection kernel – gk width"
        if "cg gk amp. glob" in s:
            return "Connection kernel – gk global amplitude"
        if "cg gk amp" in s:      # keep after "amp. glob" check
            return "Connection kernel – gk amplitude"

        # Mexican-hat kernel params
        if "cg mhk amp. exc" in s:
            return "Connection kernel – mhk exc amplitude"
        if "cg mhk width exc" in s:
            return "Connection kernel – mhk exc width"
        if "cg mhk amp. inh" in s:
            return "Connection kernel – mhk inh amplitude"
        if "cg mhk width inh" in s:
            return "Connection kernel – mhk inh width"
        if "cg mhk amp. glob" in s:
            return "Connection kernel – mhk global amplitude"

    # fallback
    return "Other / uncategorised"


def _extract_mutation_events(g: int, sol_id: int, fit: float, muts_block: str, records: list):
    """Parse one solution's 'last mutations{...}' block into individual mutation-event
    records, appended in place to `records`. Mirrors the taxonomy in categorize_mutation."""
    for s in _STRUCT_TOGGLE_RE.findall(muts_block):
        mut_inner = s.strip()
        records.append(
            {
                "generation": g,
                "solution_id": sol_id,
                "fitness": fit,
                "gene_type": "cg",
                "gene_ref": "",
                "mutation_inner": mut_inner,
                "mutation_raw": mut_inner,
                "category": categorize_mutation(mut_inner, gene_type="cg"),
            }
        )

    for inner, gtype in [
        *[(x, "fg") for x in _STRUCT_ADDED_FG_RE.findall(muts_block)],
        *[(x, "cg") for x in _STRUCT_ADDED_CG_RE.findall(muts_block)],
    ]:
        mut_inner = inner.strip()
        records.append(
            {
                "generation": g,
                "solution_id": sol_id,
                "fitness": fit,
                "gene_type": gtype,
                "gene_ref": "",
                "mutation_inner": mut_inner,
                "mutation_raw": mut_inner,
                "category": categorize_mutation(mut_inner, gene_type=gtype),
            }
        )

    for gm in _GENE_MUT_BLOCK_RE.findall(muts_block):
        gm = gm.strip()
        if not gm:
            continue

        m_head = _GENE_HEAD_RE.match(gm)
        if m_head:
            gene_type = m_head.group("gene_type")
            gene_ref = m_head.group("ref")
            rest = m_head.group("rest") or ""
        else:
            gene_type = ""
            gene_ref = ""
            rest = gm

        inners = _INNER_MUT_RE.findall(rest)
        if not inners:
            mut_inner = (rest.strip() or gm).strip()
            if _NOOP_MUT_RE.fullmatch(mut_inner):
                continue
            records.append(
                {
                    "generation": g,
                    "solution_id": sol_id,
                    "fitness": fit,
                    "gene_type": gene_type,
                    "gene_ref": gene_ref,
                    "mutation_inner": mut_inner,
                    "mutation_raw": gm,
                    "category": categorize_mutation(mut_inner, gene_type=gene_type),
                }
            )
        else:
            for inner in inners:
                mut_inner = inner.strip()
                mut_full = f"{gene_type} {gene_ref}: {mut_inner}".strip()
                records.append(
                    {
                        "generation": g,
                        "solution_id": sol_id,
                        "fitness": fit,
                        "gene_type": gene_type,
                        "gene_ref": gene_ref,
                        "mutation_inner": mut_inner,
                        "mutation_raw": mut_full,
                        "category": categorize_mutation(mut_inner, gene_type=gene_type),
                    }
                )


def parse_solution_mutation_events(record: dict, generation: int | None = None) -> list[dict]:
    """Structured mutation events for a parse_solution_blob() record, reusing the same
    taxonomy/regexes as the statistics-file scanner (also used by viz.parsing's single-pass
    statistics scan) instead of re-deriving mutation parsing here. `generation` is caller
    -supplied context (the blob itself doesn't carry its own generation number) and is stored
    verbatim on each event."""
    events: list[dict] = []
    if not record.get("mutations_raw"):
        return events
    _extract_mutation_events(
        generation, record["id"], record["fitness"], record["mutations_raw"], events
    )
    return events
