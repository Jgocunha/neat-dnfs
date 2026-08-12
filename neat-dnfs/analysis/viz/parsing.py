import math
import os
import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import streamlit as st

from .cache import _disk_cache_read_df, _disk_cache_read_json, _disk_cache_write_df, _disk_cache_write_json, _fingerprint_dir, _run_cache_dir
from .genome import kernel_kinds_for_solution

def parse_overview_line(line: str):
    pattern = (
        r"Current generation:\s*(?P<generation>\d+)"
        r".*?Number of solutions:\s*(?P<num_solutions>\d+)"
        r".*?Number of species:\s*(?P<num_species>\d+)"
        r".*?Number of active species:\s*(?P<num_active_species>\d+)"
        r".*?Has fitness improved:\s*(?P<fitness_improved>yes|no)"
        r".*?Number of generations without improvement:\s*(?P<gens_without_improvement>\d+)"
        r".*?Average fitness:\s*(?P<avg_fitness>[0-9eE\.\+\-]+)"
        r".*?Best fitness:\s*(?P<best_fitness>[0-9eE\.\+\-]+)"
        r".*?Innovation number:\s*(?P<innovation_number>\d+)"
        r".*?Average genome size:\s*(?P<avg_genome_size>[0-9eE\.\+\-]+)"
        r".*?Average connection genes:\s*(?P<avg_conn_genes>[0-9eE\.\+\-]+)"
        r".*?Average field genes:\s*(?P<avg_field_genes>[0-9eE\.\+\-]+)"
    )
    m = re.search(pattern, line)
    if not m:
        return None

    g = m.groupdict()

    def to_int(name):
        return int(g[name])

    def to_float(name):
        return float(g[name])

    return {
        "generation": to_int("generation"),
        "num_solutions": to_int("num_solutions"),
        "num_species": to_int("num_species"),
        "num_active_species": to_int("num_active_species"),
        "fitness_improved": g["fitness_improved"] == "yes",
        "gens_without_improvement": to_int("gens_without_improvement"),
        "avg_fitness": to_float("avg_fitness"),
        "best_fitness": to_float("best_fitness"),
        "innovation_number": to_int("innovation_number"),
        "avg_genome_size": to_float("avg_genome_size"),
        "avg_conn_genes": to_float("avg_conn_genes"),
        "avg_field_genes": to_float("avg_field_genes"),
    }


@st.cache_data
def load_overview(run_dir_str: str) -> pd.DataFrame:
    run_dir = Path(run_dir_str)
    overview_path = run_dir / "per_generation_overview.txt"
    if not overview_path.exists():
        raise FileNotFoundError(f"Could not find {overview_path}")

    rows = []
    with overview_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_overview_line(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise RuntimeError(f"No parsable lines in {overview_path}")

    df = pd.DataFrame(rows)
    df.sort_values("generation", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def find_runs_with_overview(base_dir: Path):
    runs = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            overview = child / "per_generation_overview.txt"
            if overview.exists():
                runs.append((child.name, child))
    return runs


_PARTIAL_FIT_RE = re.compile(r"fit\.\:\s*([0-9eE\.\+\-]+).*?part\.\:\s*\(([^)]*)\)")


_SOL_ID_RE = re.compile(r"solution\s+(\d+)\s+\[")


_STRUCT_TOGGLE_RE = re.compile(r"(toggle cg[^.\}]+\.)")


_STRUCT_ADDED_FG_RE = re.compile(r"\((added fg [^)]*)\)")


_STRUCT_ADDED_CG_RE = re.compile(r"\((added cg [^)]*)\)")


_GENE_MUT_BLOCK_RE = re.compile(r"\[([^\]]+)\]")


_GENE_HEAD_RE = re.compile(r"(?P<gene_type>[fc]g)\s+(?P<ref>[^\s(]+)\s*(?P<rest>.*)")


_INNER_MUT_RE = re.compile(r"\(([^)]+)\)")


_NOOP_MUT_RE = re.compile(r"(fg|cg)\s+\S+$")


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


def _scan_run_statistics_uncached(run_dir_str: str, generations: tuple):
    """Single pass over statistics/generation_X.txt that extracts BOTH the
    per-generation partial-fitness vectors and the per-solution mutation events.

    compute_partial_fitness and compute_mutation_events used to each independently
    re-read and re-regex the same (often 100+ MB) statistics/ text; every solution
    line carries both the 'part.: (...)' and 'last mutations{...}' data, so one pass
    is enough. Returns (partial_df, mut_df) matching what those two functions used
    to build directly. See _scan_run_statistics for the cached entry point.
    """
    run_dir = Path(run_dir_str)
    stats_dir = run_dir / "statistics"
    if not stats_dir.exists():
        return None, pd.DataFrame()

    partial_records = []
    mut_records = []

    for g in generations:
        stats_path = stats_dir / f"generation_{g + 1}.txt"
        if not stats_path.exists():
            continue

        best_fit = -1e9
        best_parts = None
        sum_parts = None
        count = 0

        with stats_path.open("r") as f:
            for line in f:
                m = _PARTIAL_FIT_RE.search(line)
                if not m:
                    continue

                fit = float(m.group(1))
                parts = []
                for token in m.group(2).split(","):
                    token = token.strip()
                    if token:
                        try:
                            parts.append(float(token))
                        except ValueError:
                            pass

                if parts:
                    if sum_parts is None:
                        sum_parts = [0.0] * len(parts)
                    for i, v in enumerate(parts):
                        sum_parts[i] += v
                    count += 1
                    if fit > best_fit:
                        best_fit = fit
                        best_parts = parts

                # ---- mutation events ----
                # str.find instead of a backtracking regex: the previous single regex
                # for this (solution ... .*? last mutations\{...\}\]) had to scan across
                # the genome/field/connection-gene text on every line and dominated
                # parse time.
                idx = line.find("last mutations{")
                if idx == -1:
                    continue
                end = line.find("}]", idx)
                if end == -1:
                    continue
                muts_block = line[idx + len("last mutations{") : end].strip()
                if not muts_block:
                    continue

                sol_m = _SOL_ID_RE.search(line)
                if not sol_m:
                    continue
                sol_id = int(sol_m.group(1))

                _extract_mutation_events(g, sol_id, fit, muts_block, mut_records)

        if count and best_parts is not None:
            avg_parts = [s / count for s in sum_parts]
            rec = {"generation": g}
            for i, v in enumerate(best_parts, start=1):
                rec[f"best_p{i}"] = v
            for i, v in enumerate(avg_parts, start=1):
                rec[f"avg_p{i}"] = v
            partial_records.append(rec)

    partial_df = None
    if partial_records:
        partial_df = pd.DataFrame(partial_records)
        partial_df.sort_values("generation", inplace=True)
        partial_df.reset_index(drop=True, inplace=True)

    if mut_records:
        mut_df = pd.DataFrame(mut_records)
        mut_df.sort_values(["generation", "solution_id"], inplace=True)
        mut_df.reset_index(drop=True, inplace=True)
        mut_df["gene_type"] = mut_df["gene_type"].astype("category")
        mut_df["category"] = mut_df["category"].astype("category")
    else:
        mut_df = pd.DataFrame()

    return partial_df, mut_df


def _scan_run_statistics_disk_cached(run_dir_str: str, generations: tuple):
    """Persistent-cache-aware wrapper around _scan_run_statistics_uncached.

    Cache lives in <run_dir>/.viz_cache/, fingerprinted on statistics/generation_*.txt
    (count/size/mtime). Falls back to a full recompute on any cache miss or error.
    """
    run_dir = Path(run_dir_str)
    cache_dir = _run_cache_dir(run_dir)
    fp = _fingerprint_dir(run_dir / "statistics", "generation_*.txt")
    meta_path = cache_dir / "statistics_scan.meta.json"
    meta = _disk_cache_read_json(meta_path)

    if meta is not None and meta.get("fingerprint") == fp:
        mut_df = _disk_cache_read_df(cache_dir / "statistics_scan.mut.parquet")
        partial_df = (
            _disk_cache_read_df(cache_dir / "statistics_scan.partial.parquet")
            if meta.get("has_partial")
            else None
        )
        if mut_df is not None:
            return partial_df, mut_df
        # cache partially unreadable -> fall through to recompute

    partial_df, mut_df = _scan_run_statistics_uncached(run_dir_str, generations)

    _disk_cache_write_df(mut_df, cache_dir / "statistics_scan.mut.parquet")
    if partial_df is not None:
        _disk_cache_write_df(partial_df, cache_dir / "statistics_scan.partial.parquet")
    _disk_cache_write_json(meta_path, {"fingerprint": fp, "has_partial": partial_df is not None})

    return partial_df, mut_df


@st.cache_data
def _scan_run_statistics(run_dir_str: str, generations: tuple):
    """Single pass over statistics/generation_X.txt (see _scan_run_statistics_uncached
    for what it extracts). Cached in-memory for the Streamlit session and on-disk
    (see _scan_run_statistics_disk_cached) so restarts don't re-pay the full parse.
    """
    return _scan_run_statistics_disk_cached(run_dir_str, generations)


@st.cache_data
def compute_partial_fitness(run_dir_str: str, generations: tuple):
    """
    For each generation, read statistics/generation_X.txt and compute:
      - best partial fitness vector (from best total fitness individual)
      - average partial fitness over all individuals
    Returns a DataFrame with columns:
      generation, best_p1..N, avg_p1..N
    """
    partial_df, _ = _scan_run_statistics(run_dir_str, generations)
    return partial_df


def generations_meeting_targets(
    generations: list, partial_vectors: list, partial_targets: dict
) -> list[int]:
    """Return 1-based generations where all partial targets are met, given raw
    per-generation partial-fitness vectors (e.g. from per_generation_overview.txt).

    Same semantics as generations_all_partial_meet_targets (missing/NaN partials
    count as NOT meeting the target), but works directly on in-memory vectors so
    callers don't need to re-parse statistics/generation_X.txt to get a DataFrame.
    """
    if not partial_targets:
        return []

    gens_ok: list[int] = []
    for g0, parts in zip(generations, partial_vectors):
        ok = True
        for comp, thr in partial_targets.items():
            idx = int(comp) - 1
            if idx < 0 or idx >= len(parts):
                ok = False
                break
            try:
                val = float(parts[idx])
                if math.isnan(val) or val < float(thr):
                    ok = False
                    break
            except Exception:
                ok = False
                break

        if ok:
            # stored generation is overview generation (0-based); display is 1-based
            gens_ok.append(int(g0) + 1)

    return gens_ok


def generations_all_partial_meet_targets(partial_df: pd.DataFrame, partial_targets: dict) -> list[int]:
    """Return 1-based generations where all partial best components meet targets."""
    if partial_df is None or partial_df.empty or not partial_targets:
        return []

    max_comp = max(int(c) for c in partial_targets.keys())
    generations = partial_df["generation"].tolist()
    partial_vectors = [
        [
            float(row[f"best_p{i}"]) if f"best_p{i}" in partial_df.columns else float("nan")
            for i in range(1, max_comp + 1)
        ]
        for _, row in partial_df.iterrows()
    ]
    return generations_meeting_targets(generations, partial_vectors, partial_targets)


def parse_species_header(line: str):
    """
    Parse one 'species ...' line from species/generation_X.txt.
    """
    header_pattern = (
        r"species\s+(?P<id>\d+)\s+\["
        r"\s*age:\s*(?P<age>\d+),\s*extinct:\s*(?P<extinct>yes|no),"
        r"\s*improved:\s*(?P<improved>yes|no),\s*gens\. since imp\.\:\s*(?P<since>\d+)"
        r"\s*offs\.\:\s*(?P<offs>\d+),\s*mem:\s*(?P<mem>\d+)"
    )
    m = re.search(header_pattern, line)
    if not m:
        return None

    g = m.groupdict()
    tail = line[m.end():].strip()

    rep_text = ""
    champ_text = ""

    rep_idx = tail.find("rep.:")
    champ_idx = tail.find("champ.:")
    if rep_idx != -1 and champ_idx != -1:
        rep_text = tail[rep_idx + len("rep.:"):champ_idx].strip()
        champ_text = tail[champ_idx + len("champ.:"):].strip()
    elif rep_idx != -1:
        rep_text = tail[rep_idx + len("rep.:"):].strip()
    elif champ_idx != -1:
        champ_text = tail[champ_idx + len("champ.:"):].strip()

    return {
        "id": int(g["id"]),
        "age": int(g["age"]),
        "extinct": g["extinct"] == "yes",
        "improved": g["improved"] == "yes",
        "gens_since_improvement": int(g["since"]),
        "offspring": int(g["offs"]),
        "members": int(g["mem"]),
        "rep_raw": rep_text,
        "champ_raw": champ_text,
    }


@st.cache_data
def compute_species_meta(run_dir_str: str, generations: tuple):
    run_dir = Path(run_dir_str)
    species_dir = run_dir / "species"
    if not species_dir.exists():
        return {}

    meta = {}
    gens_sorted = sorted(generations)
    final_gen = gens_sorted[-1] if gens_sorted else None

    for g in gens_sorted:
        path = species_dir / f"generation_{g + 1}.txt"
        if not path.exists():
            continue
        with path.open("r") as f:
            for line in f:
                if "species" not in line:
                    continue
                parsed = parse_species_header(line)
                if not parsed:
                    continue
                sid = parsed["id"]
                m = meta.setdefault(
                    sid,
                    {
                        "first_gen": g,
                        "last_gen": g,
                        "max_members": parsed["members"],
                        "total_offspring": parsed["offspring"],
                        "last_extinct": parsed["extinct"],
                    },
                )
                m["first_gen"] = min(m["first_gen"], g)
                m["last_gen"] = max(m["last_gen"], g)
                m["max_members"] = max(m["max_members"], parsed["members"])
                m["total_offspring"] += parsed["offspring"]
                if g == final_gen:
                    m["last_extinct"] = parsed["extinct"]

    return meta


@st.cache_data
def get_species_for_generation(run_dir_str: str, generation: int):
    run_dir = Path(run_dir_str)
    species_dir = run_dir / "species"
    path = species_dir / f"generation_{generation + 1}.txt"
    if not path.exists():
        return []

    result = []
    with path.open("r") as f:
        for line in f:
            if "species" not in line:
                continue
            parsed = parse_species_header(line)
            if parsed:
                result.append(parsed)
    result.sort(key=lambda d: d["id"])
    return result


def _load_elements(path: Path):
    """Read a .dnf/.json genome file and return its element list, unwrapping the
    {'deltaT': ..., 'elements': [...]} wrapper used by .dnf files."""
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception:
        return None
    if isinstance(data, dict):
        return data.get("elements")
    return data


def _sample_evenly(items: list, max_count: int) -> list:
    """Evenly-spaced sample of at most max_count items, preserving order."""
    n = len(items)
    if max_count <= 0 or n <= max_count:
        return items
    step = n / max_count
    return [items[int(i * step)] for i in range(max_count)]


@st.cache_data
def load_best_solution_architecture(run_dir_str: str, generation: int):
    """
    Load the JSON architecture for the best solution of a given generation
    from best_solutions/prev_generations.
    Returns list of element dicts (DNF composer JSON) or None.
    """
    run_dir = Path(run_dir_str)
    bs_dir = run_dir / "best_solutions" / "prev_generations"
    if not bs_dir.exists():
        return None

    # each match is a directory named "solution <id> generation <gen> species <s>
    # fitness <f>" containing a single "<same name>.dnf" file.
    pattern = f"solution * generation {generation + 1} *"
    candidates = [p for p in bs_dir.glob(pattern) if p.is_dir()]
    if not candidates:
        return None

    def fitness_from_name(p: Path):
        m = re.search(r"fitness ([0-9eE+\-\.]+)$", p.name)
        if not m:
            return -1e9
        return float(m.group(1))

    best_dir = max(candidates, key=fitness_from_name)
    genome_file = next(best_dir.glob("*.dnf"), None) or next(best_dir.glob("*.json"), None)
    if genome_file is None:
        return None

    return _load_elements(genome_file)


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


@st.cache_data
def compute_population_kernel_usage(
    run_dir_str: str,
    generations: tuple,
    gen_step: int = 10,
    max_solutions_per_gen: int = 100,
):
    """
    For a given run (run_dir_str) and list of generations, compute:

      - Per-generation percentages of Gaussian / Mexican-hat / Other for
        field kernels and interaction kernels across the *entire population*.

      - Overall counts and percentages across all generations.

    A run can have 1000 solutions x 200 generations = 200k genome files, so this
    samples: every `gen_step`-th generation, and at most `max_solutions_per_gen`
    evenly-spaced solutions within each sampled generation. Both are part of the
    cache key (and the disk-cache fingerprint), so changing them re-samples rather
    than silently reusing a differently-sampled result.

    Returns:
      df_usage: DataFrame with columns
        generation,
        field_gaussian_pct, field_mexican_pct, field_other_pct,
        inter_gaussian_pct, inter_mexican_pct, inter_other_pct

      field_overall_counts: dict(kind -> count)
      field_overall_perc:   dict(kind -> %)
      inter_overall_counts: dict(kind -> count)
      inter_overall_perc:   dict(kind -> %)
    """
    run_dir = Path(run_dir_str)
    solutions_root = run_dir / "solutions"

    field_overall_counts = Counter()
    inter_overall_counts = Counter()
    rows = []

    sampled_gens = sorted(generations)[:: max(1, gen_step)]

    for g in sampled_gens:
        gen_dir = solutions_root / f"gen {g + 1}"
        if not gen_dir.exists():
            continue

        # os.scandir + DirEntry.is_dir() reuses the directory-listing syscall's
        # cached file-type bit, unlike Path.iterdir()+Path.is_dir() which issues a
        # fresh stat() per entry -- meaningful when a generation dir has ~1000 entries.
        with os.scandir(gen_dir) as it:
            all_dirs = sorted((Path(e.path) for e in it if e.is_dir()), key=lambda p: p.name)
        solution_dirs = _sample_evenly(all_dirs, max_solutions_per_gen)

        field_kinds_gen = []
        inter_kinds_gen = []

        for sol_dir in solution_dirs:
            genome_file = next(sol_dir.glob("*.dnf"), None) or next(sol_dir.glob("*.json"), None)
            if genome_file is None:
                continue
            elements = _load_elements(genome_file)
            if elements is None:
                continue

            fk, ik = kernel_kinds_for_solution(elements)
            field_kinds_gen.extend(fk)
            inter_kinds_gen.extend(ik)

        if not field_kinds_gen and not inter_kinds_gen:
            continue

        c_field = Counter(field_kinds_gen)
        c_inter = Counter(inter_kinds_gen)

        field_overall_counts.update(c_field)
        inter_overall_counts.update(c_inter)

        def pct(counter, kind):
            total = sum(counter.values())
            return 100.0 * counter.get(kind, 0) / total if total > 0 else float("nan")

        rows.append(
            {
                "generation": g,
                "field_gaussian_pct": pct(c_field, "Gaussian"),
                "field_mexican_pct": pct(c_field, "Mexican-hat"),
                "field_other_pct": pct(c_field, "Other"),
                "inter_gaussian_pct": pct(c_inter, "Gaussian"),
                "inter_mexican_pct": pct(c_inter, "Mexican-hat"),
                "inter_other_pct": pct(c_inter, "Other"),
            }
        )

    if rows:
        df_usage = pd.DataFrame(rows).sort_values("generation")
    else:
        df_usage = pd.DataFrame()

    def perc(counter):
        total = sum(counter.values())
        if total == 0:
            return {}
        return {k: 100.0 * v / total for k, v in counter.items()}

    field_overall_perc = perc(field_overall_counts)
    inter_overall_perc = perc(inter_overall_counts)

    return (
        df_usage,
        dict(field_overall_counts),
        field_overall_perc,
        dict(inter_overall_counts),
        inter_overall_perc,
    )


@st.cache_data
def compute_mutation_events(run_dir_str: str, generations: tuple):
    """
    Parse statistics/generation_X.txt files and extract mutation events.

    We now:
      * split multi-mutations like
          fg 2 (fg gk width -1.0)(fg gk amp.-1.0)
        into TWO separate events;
      * record the gene_type ('fg' or 'cg') so we can
        distinguish field vs connection mutations;
      * also record structural events:
          - toggle cg ... to enabled/disabled.
          - (added fg ...)
          - (added cg ...)

    Returns a DataFrame with columns:
      generation, solution_id, fitness,
      gene_type ('fg' or 'cg' or 'struct'),
      gene_ref  (e.g. '2', '1-3', ...),
      mutation_inner (text inside a single (...) or the structural phrase),
      mutation_raw   (gene + inner, e.g. 'fg 2: fg gk width -1.0'),
      category       (fine-grained category from categorize_mutation).
    """
    _, mut_df = _scan_run_statistics(run_dir_str, generations)
    return mut_df


def compute_target_crossing_mutations(
    mut_events: pd.DataFrame, overview_df: pd.DataFrame, target: float
):
    """
    Approximate 'mutations that pushed above target':

    For each generation where best_fitness crosses from <target to >=target,
    look at solutions in that generation with fitness >= target and collect
    their mutations.
    """
    if mut_events.empty or overview_df.empty:
        return pd.DataFrame()

    rows = []
    for i in range(1, len(overview_df)):
        prev = overview_df.iloc[i - 1]
        curr = overview_df.iloc[i]
        if prev["best_fitness"] < target <= curr["best_fitness"]:
            g = int(curr["generation"])
            events_g = mut_events[mut_events["generation"] == g]
            if events_g.empty:
                continue
            high = events_g[events_g["fitness"] >= target]
            if high.empty:
                continue

            summary = (
                high.groupby(["mutation_raw", "category"])["fitness"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "occurrences_in_gen", "mean": "mean_fitness_in_gen"})
            )
            summary["generation"] = g
            rows.append(summary)

    if not rows:
        return pd.DataFrame()

    df_cross = pd.concat(rows, ignore_index=True)
    return df_cross


def compute_per_generation_best_mutation(mut_events: pd.DataFrame):
    """
    For each generation, find the mutation with the highest average fitness
    advantage over that generation's mean fitness (using only mutations
    with at least 3 occurrences to reduce noise).
    """
    if mut_events.empty:
        return pd.DataFrame()

    rows = []
    for g, df_g in mut_events.groupby("generation"):
        gen_mean = df_g["fitness"].mean()
        summary = (
            df_g.groupby(["mutation_raw", "category"])["fitness"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "occurrences", "mean": "mean_fitness"})
        )
        summary = summary[summary["occurrences"] >= 3]
        if summary.empty:
            continue
        summary["delta_vs_gen"] = summary["mean_fitness"] - gen_mean
        best = summary.sort_values("delta_vs_gen", ascending=False).iloc[0]
        rows.append(
            {
                "generation": g,
                "mutation": best["mutation_raw"],
                "category": best["category"],
                "occurrences": best["occurrences"],
                "mean_fitness": best["mean_fitness"],
                "delta_vs_gen": best["delta_vs_gen"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df_pg = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    return df_pg
