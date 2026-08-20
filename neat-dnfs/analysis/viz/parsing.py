import datetime
import math
import os
import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import streamlit as st

from .cache import _disk_cache_read_df, _disk_cache_read_json, _disk_cache_write_df, _disk_cache_write_json, _fingerprint_dir, _run_cache_dir
from .genome import collect_parameter_values, kernel_kinds_for_solution
from .solution_record import _AGE_PARENTS_RE, _extract_mutation_events, _GENOME_SIZE_RE, categorize_mutation, find_solution_blob, parse_solution_blob  # noqa: F401  (_extract_mutation_events, categorize_mutation re-exported)

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


@st.cache_data
def compute_topology_trajectory(run_dir_str: str) -> pd.DataFrame:
    """Per-generation topology of that generation's best solution: hidden-field count and
    enabled/disabled connection counts. Reads only per_generation_overview.txt (already read by
    load_overview) -- no statistics/ scan needed, since the best solution's full genome is
    embedded in every overview line.

    Returns a DataFrame with columns: generation (0-based), hidden_fields, enabled_connections,
    disabled_connections.
    """
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
            if parsed is None:
                continue

            blob = find_solution_blob(line)
            record = parse_solution_blob(blob) if blob else None
            if record is None:
                continue

            hidden = sum(1 for fg in record["field_genes"] if fg["type"] == "HIDDEN")
            enabled = sum(1 for cg in record["connection_genes"] if cg["enabled"])
            disabled = sum(1 for cg in record["connection_genes"] if not cg["enabled"])

            rows.append(
                {
                    "generation": parsed["generation"],
                    "hidden_fields": hidden,
                    "enabled_connections": enabled,
                    "disabled_connections": disabled,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("generation", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


@st.cache_data
def get_best_solution_id(run_dir_str: str, generation0: int) -> int | None:
    """The id of the given generation's best-total-fitness individual, read from
    per_generation_overview.txt's embedded solution blob -- no statistics/ scan."""
    run_dir = Path(run_dir_str)
    overview_path = run_dir / "per_generation_overview.txt"
    if not overview_path.exists():
        return None
    with overview_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_overview_line(line)
            if parsed is None or parsed["generation"] != generation0:
                continue
            blob = find_solution_blob(line)
            record = parse_solution_blob(blob) if blob else None
            return record["id"] if record else None
    return None


def _find_solution_in_statistics_file(stats_path: Path, sol_id: int) -> dict | None:
    target_prefix = f"solution {sol_id} ["
    if not stats_path.exists():
        return None
    with stats_path.open("r") as f:
        for line in f:
            if line.startswith(target_prefix):
                blob = find_solution_blob(line)
                return parse_solution_blob(blob) if blob else None
    return None


@st.cache_data
def trace_lineage(run_dir_str: str, start_generation0: int, start_solution_id: int, max_depth: int = 500):
    """Walk a solution's ancestry backward via its `parents (a, b)` tuple, following the first
    parent at each hop, one statistics/generation_N.txt lookup per generation, until reaching a
    (0, 0) bootstrap root, generation 0, a missing record, or max_depth hops.

    Returns a list of {"generation": int (0-based), "record": parse_solution_blob(...) dict},
    ordered oldest (root) first.
    """
    run_dir = Path(run_dir_str)
    stats_dir = run_dir / "statistics"

    chain = []
    gen0 = start_generation0
    sol_id = start_solution_id
    visited = set()

    while len(chain) < max_depth:
        if (gen0, sol_id) in visited:
            break
        visited.add((gen0, sol_id))

        record = _find_solution_in_statistics_file(stats_dir / f"generation_{gen0 + 1}.txt", sol_id)
        if record is None:
            break

        chain.append({"generation": gen0, "record": record})

        parent_a, parent_b = record["parent_ids"]
        if (parent_a, parent_b) == (0, 0) or gen0 == 0:
            break

        gen0 -= 1
        sol_id = parent_a

    chain.reverse()
    return chain


def find_runs_with_overview(base_dir: Path):
    runs = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            overview = child / "per_generation_overview.txt"
            if overview.exists():
                runs.append((child.name, child))
    if not runs and (base_dir / "per_generation_overview.txt").exists():
        # base_dir has no run subfolders but is itself a run (flat layout, e.g. the
        # checked-in data/AND, data/XOR samples) -- treat it as its own single run.
        runs.append((base_dir.name, base_dir))
    return runs


def find_experiment_dirs(data_root: Path):
    """One level up from find_runs_with_overview: folders under data_root that themselves
    contain at least one run (a subfolder with per_generation_overview.txt) -- i.e. experiment
    folders like data/HRI Packaging Task C, not run folders like data/.../2026-08-11 15h24m05s."""
    experiments = []
    if not data_root.exists() or not data_root.is_dir():
        return experiments
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if find_runs_with_overview(child):
            experiments.append((child.name, child))
    return experiments


_RUN_TIMESTAMP_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2})h(\d{2})m(\d{2})s$")


def prettify_run_timestamp(dir_name: str) -> str:
    """Turn a run folder's raw timestamp name ("2026-08-12 19h50m57s") into a friendlier
    label ("Aug 12, 19:50"). Returns the name unchanged if it doesn't match that format."""
    m = _RUN_TIMESTAMP_RE.match(dir_name)
    if not m:
        return dir_name
    year, month, day, hour, minute, _second = (int(x) for x in m.groups())
    try:
        dt = datetime.datetime(year, month, day, hour, minute)
    except ValueError:
        return dir_name
    return dt.strftime("%b %d, %H:%M")


@st.cache_data
def run_picker_label(run_dir_str: str) -> str:
    """Cheap per-run label for the run picker: a prettified timestamp plus a signal read from
    only the LAST line of per_generation_overview.txt (generation count, best fitness) --
    deliberately not a full load_overview() parse, which would build a DataFrame from every
    line and be far too slow to call once per run in a dropdown with dozens of runs."""
    run_dir = Path(run_dir_str)
    label = prettify_run_timestamp(run_dir.name)
    overview_path = run_dir / "per_generation_overview.txt"
    try:
        text = overview_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return label

    last_line = ""
    for line in reversed(text.splitlines()):
        if line.strip():
            last_line = line
            break
    if not last_line:
        return label

    info = parse_overview_line(last_line)
    if info is None:
        return label

    gen_display = info["generation"] + 1
    return f"{label} · {gen_display} gen · best {info['best_fitness']:.3f}"


_PARTIAL_FIT_RE = re.compile(r"fit\.\:\s*([0-9eE\.\+\-]+).*?part\.\:\s*\(([^)]*)\)")


_SOL_ID_RE = re.compile(r"solution\s+(\d+)\s+\[")


def _scan_run_statistics_uncached(run_dir_str: str, generations: tuple):
    """Single pass over statistics/generation_X.txt that extracts partial-fitness vectors,
    mutation events, AND per-individual distribution fields (fitness, age, genome size) --
    everything the various population-level views need, in one read.

    compute_partial_fitness and compute_mutation_events used to each independently
    re-read and re-regex the same (often 100+ MB) statistics/ text; every solution
    line carries the 'part.: (...)', 'last mutations{...}', and 'age:'/'genome (...)' data
    together, so one pass is enough. Returns (partial_df, mut_df, dist_df) matching what
    those functions build directly. See _scan_run_statistics for the cached entry point.
    """
    run_dir = Path(run_dir_str)
    stats_dir = run_dir / "statistics"
    if not stats_dir.exists():
        return None, pd.DataFrame(), pd.DataFrame()

    partial_records = []
    mut_records = []
    dist_records = []

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

                # ---- distribution fields (age, species, genome size) ----
                age_m = _AGE_PARENTS_RE.search(line)
                size_m = _GENOME_SIZE_RE.search(line)
                if age_m and size_m:
                    dist_records.append(
                        {
                            "generation": g,
                            "fitness": fit,
                            "age": int(age_m.group(1)),
                            "genome_size": int(size_m.group(1)) + int(size_m.group(2)),
                        }
                    )

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

    if dist_records:
        dist_df = pd.DataFrame(dist_records)
        dist_df.sort_values("generation", inplace=True)
        dist_df.reset_index(drop=True, inplace=True)
    else:
        dist_df = pd.DataFrame()

    return partial_df, mut_df, dist_df


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
        dist_df = _disk_cache_read_df(cache_dir / "statistics_scan.dist.parquet")
        if mut_df is not None and dist_df is not None:
            return partial_df, mut_df, dist_df
        # cache partially unreadable -> fall through to recompute

    partial_df, mut_df, dist_df = _scan_run_statistics_uncached(run_dir_str, generations)

    _disk_cache_write_df(mut_df, cache_dir / "statistics_scan.mut.parquet")
    _disk_cache_write_df(dist_df, cache_dir / "statistics_scan.dist.parquet")
    if partial_df is not None:
        _disk_cache_write_df(partial_df, cache_dir / "statistics_scan.partial.parquet")
    _disk_cache_write_json(meta_path, {"fingerprint": fp, "has_partial": partial_df is not None})

    return partial_df, mut_df, dist_df


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
    partial_df, _, _ = _scan_run_statistics(run_dir_str, generations)
    return partial_df


@st.cache_data
def compute_population_distributions(run_dir_str: str, generations: tuple) -> pd.DataFrame:
    """Per-individual fitness, age, and genome size for every solution across the given
    generations -- the raw material for population-wide distribution plots (violin/box per
    generation). Reuses the same single statistics/ pass as compute_partial_fitness and
    compute_mutation_events rather than re-scanning.

    Returns a DataFrame with columns: generation, fitness, age, genome_size (one row per
    individual).
    """
    _, _, dist_df = _scan_run_statistics(run_dir_str, generations)
    return dist_df


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


def first_crossing_per_component(
    generations: list, partial_vectors: list, partial_targets: dict
) -> dict:
    """Per-component first-crossing generation: for each component in partial_targets, the
    first (1-based) generation where that component alone meets its target -- independent of
    whether the other components also meet theirs at the same time (contrast with
    generations_meeting_targets, which requires all components simultaneously). A component
    that never crosses maps to None.

    This is a per-component failure-attribution analysis: which components gate success, not
    just whether the run succeeded overall.
    """
    result = {int(comp): None for comp in partial_targets}
    if not partial_targets:
        return result

    for g0, parts in zip(generations, partial_vectors):
        for comp, thr in partial_targets.items():
            comp = int(comp)
            if result[comp] is not None:
                continue
            idx = comp - 1
            if idx < 0 or idx >= len(parts):
                continue
            try:
                val = float(parts[idx])
            except Exception:
                continue
            if not math.isnan(val) and val >= float(thr):
                result[comp] = int(g0) + 1

    return result


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
                        "members_by_gen": {},
                        "champ_raw_by_gen": {},
                    },
                )
                m["first_gen"] = min(m["first_gen"], g)
                m["last_gen"] = max(m["last_gen"], g)
                m["max_members"] = max(m["max_members"], parsed["members"])
                m["total_offspring"] += parsed["offspring"]
                if g == final_gen:
                    m["last_extinct"] = parsed["extinct"]
                if parsed["members"] > 0:
                    m["members_by_gen"][g] = parsed["members"]
                if parsed["champ_raw"]:
                    m["champ_raw_by_gen"][g] = parsed["champ_raw"]

    return meta


@st.cache_data
def species_champion_fitness_trajectory(run_dir_str: str, generations: tuple, species_id: int) -> pd.DataFrame:
    """Fitness of the given species' champion across every generation it was active, parsed
    from the champ.: blob already collected by compute_species_meta. Only parses blobs for the
    one requested species, not every species (compute_species_meta stores raw text for all of
    them cheaply, but eagerly parsing every one would be wasted work for species nobody looks at).

    Returns a DataFrame with columns: generation (0-based), fitness.
    """
    meta = compute_species_meta(run_dir_str, generations)
    species = meta.get(species_id)
    if not species:
        return pd.DataFrame()

    rows = []
    for g, champ_raw in sorted(species["champ_raw_by_gen"].items()):
        record = parse_solution_blob(champ_raw)
        if record is not None:
            rows.append({"generation": g, "fitness": record["fitness"]})

    return pd.DataFrame(rows)


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


_CHAMPION_DIR_NAME_RE = re.compile(r"generation (\d+) species (\d+) fitness ([0-9eE+\-\.]+)$")


@st.cache_data
def list_champion_generations(run_dir_str: str, species_id: int):
    """Every generation (0-based) that has a champion genome for the given species, under
    champions/prev_generations/ -- a directory written every generation for every species'
    current champion (never touched anywhere else in this app; can hold tens of thousands of
    entries for a long run with many species, hence @st.cache_data -- avoid re-scanning it on
    every UI interaction within a session).

    Returns a sorted list of (generation0, champion_dir_path) tuples. If more than one entry
    somehow exists for the same (species, generation), the highest-fitness one wins, matching
    load_best_solution_architecture's tie-break convention.
    """
    run_dir = Path(run_dir_str)
    champ_dir = run_dir / "champions" / "prev_generations"
    if not champ_dir.exists():
        return []

    best_by_gen: dict = {}
    with os.scandir(champ_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            m = _CHAMPION_DIR_NAME_RE.search(entry.name)
            if not m:
                continue
            gen1, sid = int(m.group(1)), int(m.group(2))
            if sid != species_id:
                continue
            try:
                fit = float(m.group(3))
            except ValueError:
                fit = -1e9
            gen0 = gen1 - 1
            existing = best_by_gen.get(gen0)
            if existing is None or fit > existing[0]:
                best_by_gen[gen0] = (fit, Path(entry.path))

    return sorted((gen0, path) for gen0, (fit, path) in best_by_gen.items())


@st.cache_data
def load_champion_architecture(champion_dir_str: str):
    """Load the JSON architecture from a champion directory returned by
    list_champion_generations. Returns list of element dicts (DNF composer JSON) or None."""
    champion_dir = Path(champion_dir_str)
    genome_file = next(champion_dir.glob("*.dnf"), None) or next(champion_dir.glob("*.json"), None)
    if genome_file is None:
        return None
    return _load_elements(genome_file)


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
def compute_population_parameter_distributions(
    run_dir_str: str,
    generations: tuple,
    gen_step: int = 10,
    max_solutions_per_gen: int = 100,
) -> pd.DataFrame:
    """Sample genomes across generations (same sampling controls as
    compute_population_kernel_usage: every gen_step-th generation, up to
    max_solutions_per_gen evenly-spaced solutions per sampled generation) and collect every
    numeric field/kernel parameter -- tau, restingLevel, amplitude, width, amplitudeGlobal,
    amplitudeExc/Inh, widthExc/Inh -- across the sampled population. Currently every one of
    these is discarded elsewhere in the app; only kernel *kind* (Gaussian/Mexican-hat/Other) is
    used for the kernel-usage view.

    Returns a long-format DataFrame with columns: generation, parameter, value (one row per
    parameter instance found in a sampled genome).
    """
    run_dir = Path(run_dir_str)
    solutions_root = run_dir / "solutions"

    rows = []
    sampled_gens = sorted(generations)[:: max(1, gen_step)]

    for g in sampled_gens:
        gen_dir = solutions_root / f"gen {g + 1}"
        if not gen_dir.exists():
            continue

        with os.scandir(gen_dir) as it:
            all_dirs = sorted((Path(e.path) for e in it if e.is_dir()), key=lambda p: p.name)
        solution_dirs = _sample_evenly(all_dirs, max_solutions_per_gen)

        for sol_dir in solution_dirs:
            genome_file = next(sol_dir.glob("*.dnf"), None) or next(sol_dir.glob("*.json"), None)
            if genome_file is None:
                continue
            elements = _load_elements(genome_file)
            if elements is None:
                continue

            for param, values in collect_parameter_values(elements).items():
                for v in values:
                    rows.append({"generation": g, "parameter": param, "value": v})

    if not rows:
        return pd.DataFrame(columns=["generation", "parameter", "value"])

    return pd.DataFrame(rows)


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
    _, mut_df, _ = _scan_run_statistics(run_dir_str, generations)
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
