#!/usr/bin/env python3
from pathlib import Path
import math
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import networkx as nx  # for topology graphs
from collections import Counter
import json
import numpy as np
from datetime import datetime


# =========================
# Matplotlib global style
# =========================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond", "Times New Roman", "DejaVu Serif"]
plt.rcParams["figure.dpi"] = 100


def show_fig(fig):
    """Render a figure and close it, so it doesn't leak into Matplotlib's global
    figure registry. Every plot_*() helper creates a new figure via plt.subplots()
    and nothing ever closed them, so figures accumulated on every rerun (e.g. every
    slider drag)."""
    st.pyplot(fig)
    plt.close(fig)


# =========================
# Persistent on-disk cache
#
# @st.cache_data is in-memory only, so every Streamlit restart re-pays the full
# parse cost. These helpers add a cache directory next to the data they summarize
# (<run_dir>/.viz_cache/ for single-run parses, <base_dir>/.viz_cache/ for
# experiment-level aggregation), keyed on a cheap stat-only fingerprint of the
# source files (count/total size/max mtime) plus _PARSER_VERSION. Bump
# _PARSER_VERSION whenever parsing logic changes, so stale caches can't silently
# serve outdated results. All reads/writes are best-effort: any failure falls back
# to recomputing, so a cache problem can never break the app.
# =========================

_CACHE_DIRNAME = ".viz_cache"
_PARSER_VERSION = 1


def _run_cache_dir(dir_path: Path) -> Path:
    return dir_path / _CACHE_DIRNAME


def _fingerprint_dir(dir_path: Path, glob_pattern: str = "*") -> str:
    """Cheap fingerprint (stat-only, no content hashing) of files matching
    glob_pattern under dir_path: file count, total size, max mtime."""
    count = 0
    total_size = 0
    max_mtime = 0.0
    if dir_path.exists():
        for p in dir_path.glob(glob_pattern):
            if not p.is_file():
                continue
            try:
                s = p.stat()
            except OSError:
                continue
            count += 1
            total_size += s.st_size
            max_mtime = max(max_mtime, s.st_mtime)
    return f"v{_PARSER_VERSION}:{count}:{total_size}:{max_mtime:.3f}"


def _disk_cache_read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _disk_cache_write_json(path: Path, obj):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))
    except Exception:
        pass


def _disk_cache_read_df(path: Path):
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        pass
    return None


def _disk_cache_write_df(df: pd.DataFrame, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception:
        pass


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple GitHub-style markdown table."""
    if df is None or df.empty:
        return "_No data available._"

    cols = list(df.columns)
    header = "|" + "|".join(cols) + "|\n"
    sep = "|" + "|".join(["---"] * len(cols)) + "|\n"
    rows = []
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in cols]
        rows.append("|" + "|".join(cells) + "|")
    return header + sep + "\n".join(rows)


# =========================
# Parsing helpers: overview
# =========================

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


# =========================
# Parsing helpers: partial fitness (statistics/)
# =========================

# Pre-compiled at module scope: these run once per line across every
# statistics/generation_X.txt line in a run (hundreds of thousands of lines),
# so avoiding re-compilation per call matters.
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


# =========================
# Parsing helpers: species/ files
# =========================

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


# =========================
# Parsing helpers: best_solutions / solutions genome files (.dnf)
# =========================
#
# On-disk layout: solutions/gen N/<solution name>/<solution name>.dnf and
# best_solutions/prev_generations/<solution name>/<solution name>.dnf -- a
# *directory* per solution containing one .dnf file, not a flat *.json as the
# original globs assumed (which matched nothing). The .dnf JSON root is
# {"deltaT": ..., "elements": [...]}; callers want the "elements" list.


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

def build_topology_graph(elements):
    """
    Build a clean left-to-right interaction graph for the best solution:

      - Nodes: neural fields + kernels that mediate interactions between
        *different* fields (no self-loops).
      - Excludes gauss stimulus and normal noise.
      - Excludes kernels that only form self-loops (nf -> nf).
      - Layout: Inputs on the left, Outputs on the right, Hidden in the middle.
        Kernels are positioned between the fields they connect.

    Returns:
      g          : networkx.DiGraph
      pos        : dict node -> (x, y)
      field_nodes: list of field node names
      kernel_nodes: list of kernel node names (that are in the graph)
    """
    g = nx.DiGraph()
    if elements is None:
        return g, {}, [], []

    # --- prepare element lookup, skipping stimuli/noise ---
    by_name = {}
    for el in elements:
        name = el.get("uniqueName")
        if not name:
            continue
        lab = el.get("label", ["", ""])
        label_text = lab[1] if isinstance(lab, list) and len(lab) > 1 else str(lab)
        if label_text in {"gauss stimulus", "normal noise"}:
            continue  # drop stimuli/noise from this view entirely
        by_name[name] = el

    if not by_name:
        return g, {}, [], []

    def get_label(el):
        lab = el.get("label", ["", ""])
        return lab[1] if isinstance(lab, list) and len(lab) > 1 else str(lab)

    field_nodes = [n for n, el in by_name.items() if get_label(el) == "neural field"]
    kernel_nodes_all = [n for n, el in by_name.items() if "kernel" in get_label(el)]

    inputs_by_target = {n: (el.get("inputs") or []) for n, el in by_name.items()}

    # --- determine which kernels actually mediate field-to-field interactions (no self-loops) ---
    interactions = []  # list of (kernel, src_field, tgt_field) with src != tgt
    used_kernels = set()

    for k in kernel_nodes_all:
        k_el = by_name[k]
        src_fields = [src for src, _ in (k_el.get("inputs") or []) if src in field_nodes]

        tgt_fields = []
        for f in field_nodes:
            for src, _ in inputs_by_target.get(f, []):
                if src == k:
                    tgt_fields.append(f)
                    break

        for s in src_fields:
            for t in tgt_fields:
                if s == t:
                    # self-loop: we skip here; belongs to field dynamics, not interaction
                    continue
                interactions.append((k, s, t))
                used_kernels.add(k)

    kernel_nodes = sorted(used_kernels)

    # --- field connectivity graph (ignoring self-loops) ---
    field_graph = {f: set() for f in field_nodes}
    for _, s, t in interactions:
        field_graph[s].add(t)

    indeg = {f: 0 for f in field_nodes}
    outdeg = {f: 0 for f in field_nodes}
    for s, outs in field_graph.items():
        outdeg[s] += len(outs)
        for t in outs:
            indeg[t] += 1

    # --- classify fields: Input / Hidden / Output ---
    field_role = {}
    inputs_set, hidden_set, outputs_set = [], [], []
    for f in field_nodes:
        if indeg[f] == 0 and outdeg[f] > 0:
            role = "Input"
            inputs_set.append(f)
        elif indeg[f] > 0 and outdeg[f] == 0:
            role = "Output"
            outputs_set.append(f)
        else:
            role = "Hidden"
            hidden_set.append(f)
        field_role[f] = role

    # --- add nodes to graph ---
    for f in field_nodes:
        g.add_node(f, kind="field", role=field_role.get(f, ""))

    for k in kernel_nodes:
        g.add_node(k, kind="kernel")

    # --- add edges: field -> kernel -> field ---
    for k, s, t in interactions:
        if s in g.nodes and k in g.nodes and t in g.nodes:
            g.add_edge(s, k)
            g.add_edge(k, t)

    # --- layout: left→right ---
    pos = {}

    def assign_layer(nodes, x):
        nodes = list(nodes)
        k = len(nodes)
        for i, n in enumerate(sorted(nodes)):
            y = 1.0 - (i + 1) / (k + 1) if k > 0 else 0.5
            pos[n] = (x, y)

    # Fields
    assign_layer(inputs_set, 0.1)
    assign_layer(hidden_set, 0.5)
    assign_layer(outputs_set, 0.9)

    # Kernels: between their fields (average x/y of connected fields)
    kernel_y_fallback = 0.5
    for idx, k in enumerate(kernel_nodes):
        connected_fields = []
        for u, v in g.in_edges(k):
            if u in field_nodes and u in pos:
                connected_fields.append(u)
        for u, v in g.out_edges(k):
            if v in field_nodes and v in pos:
                connected_fields.append(v)

        if connected_fields:
            xs = [pos[f][0] for f in connected_fields]
            ys = [pos[f][1] for f in connected_fields]
            x_k = sum(xs) / len(xs)
            y_k = sum(ys) / len(ys)
        else:
            x_k = 0.5
            y_k = 1.0 - (idx + 1) / (len(kernel_nodes) + 1) if kernel_nodes else kernel_y_fallback

        pos[k] = (x_k, y_k)

    return g, pos, field_nodes, kernel_nodes


# =========================
# Plotting helpers
# =========================

def plot_total_fitness(
    df: pd.DataFrame,
    target_fitness: float,
    success_generations: list[int] | None = None,
    success_label: str = "All partial targets met",
):
    """
    Main fitness plot.

    NOTE: `df["generation"]` is 0-based (as stored in per_generation_overview.txt).
    We display generations as 1-based everywhere in the UI/plots.
    """
    fig, ax = plt.subplots(figsize=(10, 3))

    x = df["generation"] + 1
    ax.plot(x, df["avg_fitness"], label="avg. fitness")
    ax.plot(x, df["best_fitness"], label="best fitness")
    ax.axhline(target_fitness, linestyle="--", label=f"target ({target_fitness:.3f})")

    # Mark ALL generations where all partial targets were met (simultaneously).
    if success_generations:
        best_by_gen = df.assign(gen_display=df["generation"] + 1).set_index("gen_display")["best_fitness"]
        xs = [g for g in success_generations if g in best_by_gen.index]
        ys = [float(best_by_gen.loc[g]) for g in xs]

        if xs:
            label = f'{success_label} (g={", ".join(map(str, xs))})'
            ax.scatter(xs, ys, marker="D", s=40, label=label, zorder=5)

    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return fig

def plot_partial_fitness_grid(partial_df: pd.DataFrame, partial_targets: dict):
    if partial_df is None or partial_df.empty:
        st.info("No partial fitness statistics found (statistics/generation_X.txt missing?).")
        return

    num_partial = 0
    for col in partial_df.columns:
        if col.startswith("best_p"):
            idx = int(col.replace("best_p", ""))
            num_partial = max(num_partial, idx)

    if num_partial == 0:
        st.info("No partial fitness columns could be parsed.")
        return

    st.markdown("#### Partial fitness components")

    per_row = 4
    rows = math.ceil(num_partial / per_row)
    gen = partial_df["generation"]

    comp_index = 1
    for _ in range(rows):
        cols = st.columns(per_row)
        for col in cols:
            if comp_index > num_partial:
                break

            best_col = f"best_p{comp_index}"
            avg_col = f"avg_p{comp_index}"
            if best_col not in partial_df.columns or avg_col not in partial_df.columns:
                comp_index += 1
                continue

            target = partial_targets.get(comp_index)

            with col:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(gen, partial_df[avg_col], label="avg.")
                ax.plot(gen, partial_df[best_col], label="best")
                ax.axhline(target, linestyle="--", label=f"target ({target:.3f})")

                reached = partial_df[partial_df[best_col] >= target]
                if not reached.empty:
                    row = reached.iloc[0]
                    ax.scatter(row["generation"] + 1, row[best_col], marker="o", zorder=5, label="target reached")

                ax.set_xlabel("generation")
                ax.set_ylabel("fitness")
                ax.set_title(f"partial fitness {comp_index}")
                ax.legend(fontsize="x-small")
                ax.grid(True)
                fig.tight_layout()
                show_fig(fig)

            comp_index += 1


def plot_species_counts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["generation"] + 1, df["num_species"], label="species")
    ax.plot(df["generation"] + 1, df["num_active_species"], label="active species")
    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.set_title("Species count evolution")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_innovation_growth(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["generation"] + 1, df["innovation_number"], label="innovation number")
    ax.set_xlabel("generation")
    ax.set_ylabel("innovation number")
    ax.set_title("Innovation numbers growth")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def compute_kernel_usage_stats(elements):
    """
    For a given best-solution genome (elements list) compute:

      - field_kernel_kinds: one entry per field that has a primary kernel
      - interaction_kernel_kinds: one entry per kernel that mediates
        a field-to-field interaction (no self-loops)

    Returns (field_counts, field_percent, inter_counts, inter_percent)
    where each *_counts is a dict(kind -> count)
          each *_percent is a dict(kind -> % of total)
    """
    if elements is None:
        return {}, {}, {}, {}

    by_name = {el.get("uniqueName"): el for el in elements if el.get("uniqueName")}
    if not by_name:
        return {}, {}, {}, {}

    def get_label(el):
        lab = el.get("label", ["", ""])
        return lab[1] if isinstance(lab, list) and len(lab) > 1 else str(lab)

    field_nodes = [n for n, el in by_name.items() if get_label(el) == "neural field"]
    kernel_nodes_all = [n for n, el in by_name.items() if "kernel" in get_label(el)]

    inputs_by_target = {n: (el.get("inputs") or []) for n, el in by_name.items()}

    # ---- interaction kernels (same logic as in build_topology_graph, but only to decide usage) ----
    used_kernels = set()
    for k in kernel_nodes_all:
        k_el = by_name[k]
        src_fields = [src for src, _ in (k_el.get("inputs") or []) if src in field_nodes]

        tgt_fields = []
        for f in field_nodes:
            for src, _ in inputs_by_target.get(f, []):
                if src == k:
                    tgt_fields.append(f)
                    break

        for s in src_fields:
            for t in tgt_fields:
                if s == t:
                    # self-loop: not considered an interaction kernel
                    continue
                used_kernels.add(k)

    # ---- field-level kernels: primary kernel feeding each field ----
    field_kinds = []
    for f in field_nodes:
        f_el = by_name[f]
        k_name = None
        for src, _ in f_el.get("inputs") or []:
            if src in kernel_nodes_all:
                k_name = src
                break
        if k_name:
            kind = classify_kernel_kind(by_name[k_name])
            field_kinds.append(kind)

    # ---- interaction kernels usage ----
    inter_kinds = [classify_kernel_kind(by_name[k]) for k in used_kernels]

    def counts_and_perc(kinds):
        c = Counter(kinds)
        total = sum(c.values())
        if total == 0:
            return {}, {}
        perc = {k: 100.0 * v / total for k, v in c.items()}
        return dict(c), perc

    field_counts, field_perc = counts_and_perc(field_kinds)
    inter_counts, inter_perc = counts_and_perc(inter_kinds)
    return field_counts, field_perc, inter_counts, inter_perc

def kernel_kinds_for_solution(elements):
    """
    Helper used by compute_population_kernel_usage.

    For a single solution (its JSON 'elements' list), return:
      - field_kinds:  list with one entry per field's primary kernel
      - inter_kinds:  list with one entry per field–field interaction kernel

    We simply reuse compute_kernel_usage_stats and expand its counts
    into repeated labels, so the population-level code only has to
    aggregate simple lists.
    """
    field_counts, _, inter_counts, _ = compute_kernel_usage_stats(elements)

    field_kinds = []
    for kind, cnt in field_counts.items():
        field_kinds.extend([kind] * cnt)

    inter_kinds = []
    for kind, cnt in inter_counts.items():
        inter_kinds.extend([kind] * cnt)

    return field_kinds, inter_kinds

def classify_kernel_kind(el) -> str:
    """Return 'Gaussian', 'Mexican-hat', or 'Other' based on the label."""
    if el is None:
        return "Other"
    lab = el.get("label", ["", ""])
    label_text = lab[1] if isinstance(lab, list) and len(lab) > 1 else str(lab)
    low = label_text.lower()
    if "mexican" in low:
        return "Mexican-hat"
    if "gauss" in low:
        return "Gaussian"
    return "Other"


def summarize_best_solution_genome(elements):
    """
    Build two tables (as DataFrames) that describe the best solution's genome:

      - Field genes: one row per neural field
      - Interaction genes: one row per (kernel, from-field -> to-field) pair

    Field roles (Input / Hidden / Output) are inferred from connectivity:
      - Build a field->field graph via kernels, ignoring self-loops.
      - Input:    no incoming edges from other fields.
      - Output:   no outgoing edges to other fields.
      - Hidden:   everything else.

    Self-loop kernels (nf -> nf) are treated as part of that field's own
    dynamics and are NOT listed as separate interaction genes.
    """
    if elements is None:
        return None, None

    by_name = {el.get("uniqueName"): el for el in elements if el.get("uniqueName")}
    if not by_name:
        return None, None

    def get_label(el):
        lab = el.get("label", ["", ""])
        return lab[1] if isinstance(lab, list) and len(lab) > 1 else str(lab)

    field_nodes = [n for n, el in by_name.items() if get_label(el) == "neural field"]
    kernel_nodes = [n for n, el in by_name.items() if "kernel" in get_label(el)]

    # ---------- connectivity via kernels ----------
    inputs_by_target = {n: (el.get("inputs") or []) for n, el in by_name.items()}

    # field -> set of downstream fields (ignoring self-loops)
    field_graph = {f: set() for f in field_nodes}

    for k in kernel_nodes:
        k_el = by_name[k]
        k_inputs = k_el.get("inputs") or []
        src_fields = [src for src, _ in k_inputs if src in field_nodes]

        tgt_fields = []
        for f in field_nodes:
            for src, _ in inputs_by_target.get(f, []):
                if src == k:
                    tgt_fields.append(f)
                    break

        for s in src_fields:
            for t in tgt_fields:
                if s == t:
                    # self-loop: treat as part of field dynamics, not as interaction
                    continue
                field_graph[s].add(t)

    indeg = {f: 0 for f in field_nodes}
    outdeg = {f: 0 for f in field_nodes}
    for s, outs in field_graph.items():
        outdeg[s] += len(outs)
        for t in outs:
            indeg[t] += 1

    # ---------- classify fields as Input / Hidden / Output ----------
    field_role = {}
    for f in field_nodes:
        if indeg[f] == 0 and outdeg[f] > 0:
            role = "Input"
        elif indeg[f] > 0 and outdeg[f] == 0:
            role = "Output"
        else:
            # no connections at all or both in & out
            role = "Hidden"
        field_role[f] = role

    # ---------- parameter formatting helpers ----------
    def kernel_param_str(k_el):
        lab = get_label(k_el)
        lab_lower = lab.lower()

        # Mexican-hat kernel
        if "mexican" in lab_lower:
            Ae = k_el.get("amplitudeExc")
            se = k_el.get("widthExc")
            Ai = k_el.get("amplitudeInh")
            si = k_el.get("widthInh")
            Ag = k_el.get("amplitudeGlobal", 0.0)
            return (
                "Mexican-hat kernel: "
                f"A_exc = {Ae:.2f}, σ_exc = {se:.2f}, "
                f"A_inh = {Ai:.2f}, σ_inh = {si:.2f}, "
                f"A_glob = {Ag:.2f}"
            )

        # Gaussian(-like) kernel
        if "gauss" in lab_lower:
            A = k_el.get("amplitude") or k_el.get("amplitudeExc")
            s = k_el.get("width") or k_el.get("widthExc")
            Ag = k_el.get("amplitudeGlobal", 0.0)
            return f"Gaussian kernel: A = {A:.2f}, σ = {s:.2f}, A_glob = {Ag:.2f}"

        # fallback: just label
        return lab

    # ---------- Field genes table ----------
    field_rows = []
    for f in sorted(field_nodes):
        f_el = by_name[f]
        # pick the first kernel feeding this field as its "primary" kernel
        k_name = None
        for src, _ in f_el.get("inputs") or []:
            if src in kernel_nodes:
                k_name = src
                break
        k_el = by_name.get(k_name) if k_name else None

        role = field_role.get(f, "")
        kernel_type = get_label(k_el) if k_el else ""

        h = f_el.get("restingLevel")
        tau = f_el.get("tau")
        if h is not None and tau is not None:
            field_params = f"h = {h:.2f}, τ = {tau:.2f}"
        else:
            field_params = ""

        kernel_params = kernel_param_str(k_el) if k_el else ""

        field_rows.append(
            {
                "Field": f,
                "Role": role,
                "Kernel type": kernel_type,
                "Field parameters": field_params,
                "Kernel parameters": kernel_params,
            }
        )

    df_fields = pd.DataFrame(field_rows)

    # ---------- Interaction genes table (exclude self-loops) ----------
    inter_rows = []
    for k in kernel_nodes:
        k_el = by_name[k]
        k_params = kernel_param_str(k_el)

        src_fields = [src for src, _ in (k_el.get("inputs") or []) if src in field_nodes]

        tgt_fields = []
        for f in field_nodes:
            for src, _ in inputs_by_target.get(f, []):
                if src == k:
                    tgt_fields.append(f)
                    break

        for s in src_fields:
            for t in tgt_fields:
                if s == t:
                    # self-loop already represented as that field's kernel; skip
                    continue
                inter_rows.append(
                    {
                        "Interaction gene": k,
                        "From → To": f"{s} → {t}",
                        "Kernel parameters": k_params,
                    }
                )

    df_inter = pd.DataFrame(inter_rows)
    return df_fields, df_inter




def plot_genome_topology_curves(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["generation"] + 1, df["avg_genome_size"], label="avg genome size")
    ax.plot(df["generation"] + 1, df["avg_field_genes"], label="avg field genes")
    ax.plot(df["generation"] + 1, df["avg_conn_genes"], label="avg connection genes")
    ax.set_xlabel("generation")
    ax.set_ylabel("genes")
    ax.set_title("Genome topology")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# =========================
# Statistics helpers
# =========================

def render_fitness_stats(
    df: pd.DataFrame,
    target_fitness: float,
    partial_threshold_generation: int | None = None,
):
    # df["generation"] is 0-based (as stored in per_generation_overview.txt).
    # We DISPLAY generations as 1-based everywhere.
    final_row = df.iloc[-1]
    final_gen0 = int(final_row["generation"])
    final_gen = final_gen0 + 1

    best_final = float(final_row["best_fitness"])
    avg_final = float(final_row["avg_fitness"])

    max_best = float(df["best_fitness"].max())
    gen_max_best0 = int(df.loc[df["best_fitness"].idxmax(), "generation"])
    gen_max_best = gen_max_best0 + 1

    # Visual/diagnostic reference: when total best fitness crosses chosen "overall target".
    reached_total = df[df["best_fitness"] >= target_fitness]
    total_cross_gen = None
    if not reached_total.empty:
        total_cross_gen0 = int(reached_total.iloc[0]["generation"])
        total_cross_gen = total_cross_gen0 + 1

    # Stagnation (in best fitness)
    best_series = df["best_fitness"].astype(float).values
    longest_stagnation = 0
    current = 0
    for i in range(1, len(best_series)):
        if best_series[i] <= best_series[i - 1]:
            current += 1
            longest_stagnation = max(longest_stagnation, current)
        else:
            current = 0

    auc_best = float(df["best_fitness"].mean())
    auc_avg = float(df["avg_fitness"].mean())

    st.markdown("#### Statistics")
    st.markdown(
        f"""
        **Final generation (g = {final_gen})**  
        • Best fitness: **{best_final:.4f}**  
        • Average fitness: **{avg_final:.4f}**  

        **Overall**  
        • Max best fitness: **{max_best:.4f}** (reached at generation {gen_max_best})  
        • Mean best fitness over run (AUC): **{auc_best:.4f}**  
        • Mean average fitness over run (AUC): **{auc_avg:.4f}**  
        • Longest stagnation (best fitness not improving): **{longest_stagnation}** generations  
        """,
    )

    if partial_threshold_generation is not None:
        st.success(f"✅ All partial fitness targets met simultaneously at generation **{partial_threshold_generation}**.")
    else:
        st.warning("❌ All partial fitness targets were **never** met simultaneously in this run.")

    if total_cross_gen is not None:
        st.caption(f"Reference: total best fitness first crossed the chosen overall target at generation {total_cross_gen}.")
def render_species_stats(df: pd.DataFrame, species_meta: dict):
    last = df.iloc[-1]
    final_gen = int(last["generation"])
    final_species = int(last["num_species"])
    final_active = int(last["num_active_species"])

    # Basic from overview
    avg_species = df["num_species"].mean()
    avg_active = df["num_active_species"].mean()

    # This is usually monotonic in your setup, so it's less interesting
    max_species = df["num_species"].max()
    gen_max_species = int(df.loc[df["num_species"].idxmax(), "generation"])

    # More informative: max ACTIVE species
    max_active_species = df["num_active_species"].max()
    gen_max_active = int(df.loc[df["num_active_species"].idxmax(), "generation"])

    # From species_meta (unchanged)
    total_species = len(species_meta)
    lifespans = []
    max_members_list = []
    offspring_list = []
    active_final = 0
    for sid, m in species_meta.items():
        span = m["last_gen"] - m["first_gen"] + 1
        lifespans.append(span)
        max_members_list.append(m["max_members"])
        offspring_list.append(m["total_offspring"])
        if m["last_gen"] == final_gen and not m["last_extinct"]:
            active_final += 1

    extinct_species = total_species - active_final

    if lifespans:
        avg_lifespan = sum(lifespans) / len(lifespans)
        max_lifespan = max(lifespans)
        max_life_sid = [
            sid
            for sid, m in species_meta.items()
            if m["last_gen"] - m["first_gen"] + 1 == max_lifespan
        ][0]
    else:
        avg_lifespan = 0.0
        max_lifespan = 0
        max_life_sid = None

    avg_max_members = sum(max_members_list) / len(max_members_list) if max_members_list else 0.0
    avg_offspring = sum(offspring_list) / len(offspring_list) if offspring_list else 0.0

    st.markdown("#### Species statistics")
    st.markdown(
        f"""
        **Final generation (g = {final_gen + 1})**  
        • Species: **{final_species}**  
        • Active species: **{final_active}**  

        **Across run**  
        • Total distinct species ever created: **{total_species}**  
        • Species that went extinct by final generation: **{extinct_species}**  
        • Average number of species per generation: **{avg_species:.2f}**  
        • Average number of active species per generation: **{avg_active:.2f}**  
        • Max number of **active** species in a generation: **{max_active_species}** (at generation {gen_max_active})  
        """
    )

    st.markdown(
        f"""
        **Species lifetime & size**  
        • Average species lifespan: **{avg_lifespan:.2f}** generations  
        • Longest-lived species: **{max_life_sid}** (lifespan {max_lifespan} generations)  
        • Average max members per species: **{avg_max_members:.2f}**  
        • Average total offspring assigned per species: **{avg_offspring:.2f}**  
        """
    )



def render_topology_stats(df: pd.DataFrame):
    first = df.iloc[0]
    last = df.iloc[-1]

    g0 = first["avg_genome_size"]
    gN = last["avg_genome_size"]
    f0 = first["avg_field_genes"]
    fN = last["avg_field_genes"]
    c0 = first["avg_conn_genes"]
    cN = last["avg_conn_genes"]

    gens = last["generation"] - first["generation"]
    gens = gens if gens > 0 else 1

    st.markdown("#### Topology statistics")
    st.markdown(
        f"""
        **Final generation (g = {int(last['generation']) + 1})**  
        • Avg genome size: **{gN:.2f}**  
        • Avg field genes: **{fN:.2f}**  
        • Avg connection genes: **{cN:.2f}**  

        **Growth over run**  
        • Genome size change: **{gN - g0:+.2f}** (≈ {(gN - g0)/gens:+.3f} per generation)  
        • Field genes change: **{fN - f0:+.2f}** (≈ {(fN - f0)/gens:+.3f} per generation)  
        • Connection genes change: **{cN - c0:+.2f}** (≈ {(cN - c0)/gens:+.3f} per generation)  

        **Ratios**  
        • Avg connections per field at final gen: **{(cN / fN) if fN > 0 else 0.0:.2f}**  
        • Avg genome size / population size is accessible from per-generation statistics if needed.  
        """
    )

# =========================
# Mutation helpers
# =========================

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

def plot_kernel_usage_time(df_usage: pd.DataFrame, kind: str):
    """
    kind: 'field' or 'inter'
    """
    if df_usage is None or df_usage.empty:
        return None

    if kind == "field":
        y1 = "field_gaussian_pct"
        y2 = "field_mexican_pct"
        title = "Field kernel usage across population"
        ylabel = "% of field kernels"
    else:
        y1 = "inter_gaussian_pct"
        y2 = "inter_mexican_pct"
        title = "Interaction kernel usage across population"
        ylabel = "% of interaction kernels"

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df_usage["generation"] + 1, df_usage[y1], label="Gaussian")
    ax.plot(df_usage["generation"] + 1, df_usage[y2], label="Mexican-hat")
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig

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



def plot_mutations_per_generation(mut_events: pd.DataFrame):
    """Line plot: how many mutations occurred each generation."""
    per_gen = (
        mut_events.groupby("generation")["mutation_raw"]
        .count()
        .reset_index(name="num_mutations")
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_gen["generation"] + 1, per_gen["num_mutations"])
    ax.set_xlabel("generation")
    ax.set_ylabel("number of mutations")
    ax.set_title("Mutation activity per generation")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_mutation_effectiveness(mut_events: pd.DataFrame, top_n: int = 10):
    """
    Horizontal bar plot of the top mutations with the highest
    positive Δ mean fitness vs the global mean.
    """
    if mut_events.empty:
        return None

    summary = (
        mut_events.groupby("mutation_raw")["fitness"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "occurrences", "mean": "mean_fitness"})
    )

    overall_mean = mut_events["fitness"].mean()
    summary["delta_vs_global"] = summary["mean_fitness"] - overall_mean

    # Top N beneficial
    top = (
        summary.sort_values("delta_vs_global", ascending=False)
        .head(top_n)
        .sort_values("delta_vs_global")  # so bars go from worst -> best
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["mutation_raw"], top["delta_vs_global"])
    ax.set_xlabel("Δ mean fitness vs global")
    ax.set_title(f"Most beneficial mutations (top {len(top)})")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def render_mutation_tables(mut_events: pd.DataFrame):
    """Tables: frequency + effectiveness, with clearer column names."""
    if mut_events.empty:
        st.info("No mutation information found in statistics/generation_X.txt.")
        return

    overall_mean = mut_events["fitness"].mean()

    summary = (
        mut_events.groupby(["mutation_raw", "category"])["fitness"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "occurrences", "mean": "mean_fitness"})
    )
    summary["delta_vs_global"] = summary["mean_fitness"] - overall_mean

    # Most frequent
    st.markdown("#### Most frequent mutations")
    st.caption(
        "How often each mutation appears, and how the average fitness of individuals "
        "with that mutation compares to the overall mean fitness of the run."
    )
    top_freq = summary.sort_values("occurrences", ascending=False).head(10)
    st.dataframe(
        top_freq.rename(
            columns={
                "mutation_raw": "mutation description",
                "category": "mutation category",
                "occurrences": "how many times this mutation was applied",
                "mean_fitness": "mean fitness of individuals with this mutation",
                "delta_vs_global": "Δ vs global mean fitness",
            }
        ),
        width="stretch",
    )

    # Most beneficial (positive delta)
    st.markdown("#### Mutations associated with higher fitness")
    st.caption(
        "Mutations whose carriers tend to have higher fitness than the global mean. "
        "Beware of very rare mutations (low counts) — they can look good just by chance."
    )
    top_good = (
        summary.sort_values("delta_vs_global", ascending=False)
        .head(10)
        .rename(
            columns={
                "mutation_raw": "mutation description",
                "category": "mutation category",
                "occurrences": "how many times this mutation was applied",
                "mean_fitness": "mean fitness of individuals with this mutation",
                "delta_vs_global": "Δ vs global mean fitness",
            }
        )
    )
    st.dataframe(top_good, width="stretch")


def plot_mutation_categories(mut_events: pd.DataFrame):
    """Bar chart: how many mutation events per high-level category."""
    per_cat = (
        mut_events.groupby("category")["mutation_raw"]
        .count()
        .reset_index(name="num_events")
        .sort_values("num_events", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(per_cat["category"], per_cat["num_events"])
    ax.set_ylabel("mutation events")
    ax.set_title("Mutation events by category")

    # rotate + right-align x tick labels for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    fig.tight_layout()
    return fig

def plot_topology_graph(g, pos, field_nodes, kernel_nodes):
    """
    Draw the interaction graph with a clean left→right layout.

    Fields and kernels are shown with different shapes.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(g.nodes) == 0:
        ax.text(0.5, 0.5, "No interaction topology available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    if not pos:
        pos = nx.spring_layout(g, seed=42)

    # Edges
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowstyle="->")

    # Field nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[n for n in field_nodes if n in g.nodes],
        node_shape="o",
        ax=ax,
    )

    # Kernel nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[n for n in kernel_nodes if n in g.nodes],
        node_shape="s",
        ax=ax,
    )

    # Labels: just use node names (nf 1, gk 1-3, etc.)
    nx.draw_networkx_labels(g, pos, font_size=8, ax=ax)

    ax.axis("off")
    fig.tight_layout()
    return fig


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


def plot_best_mutation_timeline(per_gen_df: pd.DataFrame):
    """Line plot: per-generation most beneficial mutation (delta vs gen mean)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_gen_df["generation"] + 1, per_gen_df["delta_vs_gen"])
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("generation")
    ax.set_ylabel("Δ best mutation vs gen mean")
    ax.set_title("Per-generation most beneficial mutation (approx.)")
    ax.grid(True)
    fig.tight_layout()
    return fig

# =========================
# Experiment-level helpers (across many runs)
# =========================

# ---- runtime + total-mutation statistics (from evolution_timestamps.txt,
#      field_gene_statistics_total.txt, genome_statistics_total.txt)
#      adapted from analysis-other-statistics.py
# --------------------------------------------------------------

def _parse_evolution_timestamps(file_path: Path):
    metrics = {}
    if not file_path.exists():
        return metrics

    txt = file_path.read_text()

    num_generations = re.search(r"Number of generations: (\d+)", txt)
    start_time = re.search(r"Evolution Start Time: ([\d\-: ]+)", txt)
    end_time = re.search(r"Evolution End Time: ([\d\-: ]+)", txt)
    duration_seconds = re.search(r"Duration \(seconds\): (\d+)", txt)

    if not (num_generations and start_time and end_time and duration_seconds):
        return metrics

    metrics["num_generations"] = int(num_generations.group(1))
    metrics["start_time"] = start_time.group(1)
    metrics["end_time"] = end_time.group(1)
    metrics["duration_seconds"] = int(duration_seconds.group(1))

    # derived
    if metrics["num_generations"] > 0:
        metrics["seconds_per_generation"] = (
            metrics["duration_seconds"] / metrics["num_generations"]
        )

    start_dt = datetime.strptime(metrics["start_time"], "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(metrics["end_time"], "%Y-%m-%d %H:%M:%S")
    metrics["duration_hours"] = (end_dt - start_dt).total_seconds() / 3600.0
    return metrics


def _parse_field_gene_statistics(file_path: Path):
    metrics = {}
    if not file_path.exists():
        return metrics

    txt = file_path.read_text()
    pattern = r"Total (\w+(?:\s\w+)*) mutations: (\d+)"
    for mutation_type, count in re.findall(pattern, txt):
        key = mutation_type.replace(" ", "_").lower()
        metrics[key] = int(count)

    # kernel mix & kernel/field ratio
    g = metrics.get("gauss_kernel", 0)
    m = metrics.get("mexican_hat_kernel", 0)
    o = metrics.get("oscillatory_kernel", 0)
    total_k_specific = g + m + o
    if total_k_specific > 0:
        metrics["gauss_kernel_pct"] = 100.0 * g / total_k_specific
        metrics["mexican_hat_kernel_pct"] = 100.0 * m / total_k_specific
        metrics["oscillatory_kernel_pct"] = 100.0 * o / total_k_specific

    k = metrics.get("kernel", 0)
    nf = metrics.get("neural_field", 0)
    metrics["kernel_to_field_ratio"] = (k / nf) if nf > 0 else 0.0
    return metrics


def _parse_genome_statistics(file_path: Path):
    metrics = {}
    if not file_path.exists():
        return metrics

    txt = file_path.read_text()
    pattern = r"(\w+(?:\s\w+)*) mutations total: (\d+)"
    for mutation_type, count in re.findall(pattern, txt):
        key = mutation_type.replace(" ", "_").lower()
        metrics[key] = int(count)

    total_mutations = sum(metrics.values())
    metrics["total_mutations"] = total_mutations

    # per-type percentages
    if total_mutations > 0:
        for key, val in list(metrics.items()):
            if key == "total_mutations":
                continue
            metrics[f"{key}_pct"] = 100.0 * val / total_mutations

    # structural vs parametric
    structural = metrics.get("add_connection_gene", 0) + metrics.get(
        "add_field_gene", 0
    )
    parametric = metrics.get("mutate_field_gene", 0) + metrics.get(
        "mutate_connection_gene", 0
    )
    metrics["structural_mutations"] = structural
    metrics["parametric_mutations"] = parametric
    metrics["structural_to_parametric_ratio"] = (
        structural / parametric if parametric > 0 else 0.0
    )
    return metrics


def _analyze_single_run_totals(run_dir: Path):
    """Collect 'global totals' for a single run directory."""

    out = {"run_dir": run_dir.name}

    # evolution_timestamps.txt is now in the RUN ROOT
    out.update(_parse_evolution_timestamps(run_dir / "evolution_timestamps.txt"))

    return out



@st.cache_data
def compute_experiment_totals(base_dir_str: str):
    """
    Aggregate runtime + mutation totals over all runs in base_dir.
    Returns (agg_metrics, df) where df has one row per run.
    """
    base = Path(base_dir_str)

    # USE THE SAME "RUN DETECTION" AS THE REST OF THE APP
    run_dirs = [
        d
        for d in base.iterdir()
        if d.is_dir() and (d / "per_generation_overview.txt").exists()
    ]

    rows = []
    for rd in run_dirs:
        m = _analyze_single_run_totals(rd)
        if m:
            rows.append(m)

    if not rows:
        return {}, pd.DataFrame()

    df = pd.DataFrame(rows)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        agg[f"{col}_mean"] = float(series.mean())
        agg[f"{col}_median"] = float(series.median())
        agg[f"{col}_min"] = float(series.min())
        agg[f"{col}_max"] = float(series.max())
        agg[f"{col}_std"] = float(series.std(ddof=0))

    return agg, df



def render_experiment_totals(agg: dict, df: pd.DataFrame):
    """Streamlit UI for the 'other-statistics' style aggregates."""
    if df.empty:
        st.info("No experiment-level statistics files found in this base directory.")
        return

    st.markdown("### Experiment-level runtime & mutation mix")

    n_runs = len(df)
    st.markdown(f"- Analysed **{n_runs}** runs in this experiment.")

    # --- Time / performance summary ---
    st.markdown("#### Time / performance")
    dur_mean = agg.get("duration_hours_mean")
    spg_mean = agg.get("seconds_per_generation_mean")
    if dur_mean is not None:
        st.markdown(f"- Average run duration: **{dur_mean:.2f} h**")
    if spg_mean is not None:
        st.markdown(f"- Avg. time per generation: **{spg_mean:.2f} s/gen**")

    ratio = agg.get("structural_to_parametric_ratio_mean")
    if ratio is not None:
        st.markdown(
            f"- Structural / parametric mutation ratio (mean): **{ratio:.3f}**"
        )

    kf = agg.get("kernel_to_field_ratio_mean")
    if kf is not None:
        st.markdown(
            f"- Kernel / neural-field mutation ratio (mean): **{kf:.2f}**"
        )

        # ---- a couple of simple histograms ----
    st.markdown("#### Distributions across runs")
    plot_cols = st.columns(3)

    # duration
    if "duration_hours" in df.columns:
        with plot_cols[0]:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df["duration_hours"].dropna(), bins=10)
            ax.set_xlabel("duration (hours)")
            ax.set_ylabel("runs")
            ax.set_title("Run duration")
            fig.tight_layout()
            show_fig(fig)
            st.caption(
                "How long each run took from start to end. This shows whether some runs "
                "are much slower or faster than others."
            )

    # seconds per generation
    if "seconds_per_generation" in df.columns:
        with plot_cols[1]:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df["seconds_per_generation"].dropna(), bins=10)
            ax.set_xlabel("sec / generation")
            ax.set_ylabel("runs")
            ax.set_title("Time per generation")
            fig.tight_layout()
            show_fig(fig)
            st.caption(
                "Average wall-clock time needed to compute one generation in each run. "
                "Useful for spotting performance regressions."
            )

    # structural vs parametric
    if (
        "structural_mutations" in df.columns
        and "parametric_mutations" in df.columns
    ):
        with plot_cols[2]:
            vals = df[["structural_mutations", "parametric_mutations"]].mean()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.pie(
                vals.values,
                labels=["Structural", "Parametric"],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title("Structural vs parametric (mean per run)")
            fig.tight_layout()
            show_fig(fig)
            st.caption(
                "Average mix of structural vs parameter mutations per run. "
                "This summarises how much the algorithm changes topology versus fine-tunes parameters."
            )



# ---- convergence / architecture statistics over runs
#      (from per_generation_overview.txt)
#      adapted from analysis-per-generation-overview.py
# --------------------------------------------------------------


def _format_duration_human(duration_seconds: int | None) -> str | None:
    """Return a compact human-readable duration (e.g., '12m' or '1h 05m')."""
    if duration_seconds is None:
        return None
    try:
        secs = int(duration_seconds)
    except Exception:
        return None
    if secs < 0:
        return None
    mins = secs // 60
    hours = mins // 60
    rem_mins = mins % 60
    if hours > 0:
        return f"{hours}h {rem_mins:02d}m"
    return f"{mins}m"


def _parse_run_overview(stats_file: Path):
    """Parse per_generation_overview.txt into target-independent run metrics.

    This does NOT depend on partial_targets and does NOT touch statistics/ at all
    (the per-generation best partial-fitness vector is already in this file). Keeping
    it target-independent lets callers cache the parse once per run and cheaply
    re-evaluate success/failure against different targets (see _evaluate_run_targets).
    """
    txt = stats_file.read_text()

    # NOTE: generations in per_generation_overview.txt are 0-based (actual generation = g + 1).
    # We keep internal generation indices 0-based; any generation we show to the user is 1-based.

    gen_pattern = (
        r"Current generation: (\d+).*?"
        r"Number of species: (\d+).*?"
        r"Number of active species: (\d+).*?"
        r"Best solution: \[solution (\d+) \[ fit\.: ([\d\.]+), part\.: \((.*?)\),\s*spec\.: (\d+),.*?"
        r"genome \((.*?)\).*?"
        r"field genes \{(.*?)\}.*?"
        r"connection genes \{(.*?)\}"
    )
    generations_data = re.findall(gen_pattern, txt, re.DOTALL)
    if not generations_data:
        return None

    generations = []
    fitness_values = []
    partial_vectors = []
    species_total = []
    species_active = []
    best_solution_ids = []
    best_solution_species = []
    field_genes_strs = []
    conn_genes_strs = []

    for (
        g_str,
        sp_total_str,
        sp_active_str,
        sol_id_str,
        fit_str,
        parts_str,
        sol_species_str,
        _genome_header,
        field_str,
        conn_str,
    ) in generations_data:
        g0 = int(g_str)
        generations.append(g0)
        fitness_values.append(float(fit_str))

        # partials as list[float]
        parts = []
        for x in parts_str.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                parts.append(float(x))
            except Exception:
                pass
        partial_vectors.append(parts)

        species_total.append(int(sp_total_str))
        species_active.append(int(sp_active_str))
        best_solution_ids.append(int(sol_id_str))
        best_solution_species.append(int(sol_species_str))
        field_genes_strs.append(field_str)
        conn_genes_strs.append(conn_str)

    # last generation metadata (species totals/actives)
    final_g0 = max(generations)
    final_idx = generations.index(final_g0)
    final_species_total = species_total[final_idx]
    final_species_active = species_active[final_idx]

    # Best solution diagnostics for the *last* generation line in per_generation_overview.txt
    # (per_generation_overview.txt generations are 0-based, so display generation is g0 + 1).
    last_g0 = final_g0
    last_idx = final_idx

    best_g0 = last_g0
    best_solution_id = best_solution_ids[last_idx]
    best_solution_species_id = best_solution_species[last_idx]
    best_solution_fitness = fitness_values[last_idx]
    best_solution_partials = partial_vectors[last_idx]

    # topology of the last-generation best solution (hidden fields + enabled connections)
    field_types = re.findall(
        r"fg \(id: \d+, type: (INPUT|OUTPUT|HIDDEN)\)", field_genes_strs[last_idx]
    )
    hidden_fields_count = field_types.count("HIDDEN")
    conn_states = re.findall(r"enabled: (true|false)", conn_genes_strs[last_idx])
    enabled_connections_count = conn_states.count("true")

    # duration (from evolution_timestamps.txt, if available)
    ts_metrics = _parse_evolution_timestamps(stats_file.parent / "evolution_timestamps.txt")
    duration_seconds = ts_metrics.get("duration_seconds")
    duration_human = _format_duration_human(duration_seconds)

    # fitness improvements (over total fitness)
    fitness_improvements = []
    for i in range(1, len(fitness_values)):
        improvement = max(0.0, fitness_values[i] - fitness_values[i - 1])
        fitness_improvements.append(improvement)

    max_fitness = max(fitness_values)
    min_fitness = min(fitness_values)
    avg_improvement = (
        sum(fitness_improvements) / len(fitness_improvements)
        if fitness_improvements
        else 0.0
    )

    return {
        # best-solution diagnostics
        "best_solution_generation": best_g0 + 1,  # display (1-based)
        "best_solution_id": best_solution_id,
        "best_solution_species_id": best_solution_species_id,
        "best_solution_fitness": best_solution_fitness,
        "best_solution_partials": best_solution_partials,
        # reference stats
        "max_fitness": max_fitness,
        "min_fitness": min_fitness,
        "total_generations": len(generations),
        "avg_improvement_per_gen": avg_improvement,
        "generations": generations,
        "fitness_values": fitness_values,
        "partial_vectors": partial_vectors,
        "fitness_improvements": fitness_improvements,
        # final run state
        "final_species_total": final_species_total,
        "final_species_active": final_species_active,
        # topology
        "hidden_fields_count": hidden_fields_count,
        "enabled_connections_count": enabled_connections_count,
        # runtime
        "duration_seconds": duration_seconds,
        "duration_human": duration_human,
        # we fill run_dir higher up
    }


def _evaluate_run_targets(parsed: dict, partial_targets: dict) -> dict:
    """Combine a target-independent parsed run (see _parse_run_overview) with a set
    of partial-fitness targets to compute success/threshold/failed-partials.

    Cheap: works entirely on already-parsed in-memory data, no file I/O.
    """
    gens_ok = generations_meeting_targets(
        parsed["generations"], parsed["partial_vectors"], partial_targets
    )
    success = len(gens_ok) > 0
    generation_to_threshold = gens_ok[0] if success else None  # 1-based, per helper

    # which partial targets the BEST solution failed (based on partial_targets only)
    best_solution_partials = parsed["best_solution_partials"]
    failed_partials = []
    for p_num, thr in sorted(partial_targets.items()):
        # Partial fitness targets are specified as p1, p2, ... (1-based indexing).
        if p_num <= 0:
            continue
        idx = p_num - 1  # convert to 0-based index into the partial vector
        if idx >= len(best_solution_partials):
            failed_partials.append(f"p{p_num}=NA<{thr:.3f}")
            continue
        val = best_solution_partials[idx]
        # Treat missing/NaN values as failures.
        if val is None or (isinstance(val, float) and math.isnan(val)) or float(val) < float(thr):
            try:
                failed_partials.append(f"p{p_num}={float(val):.3f}<{thr:.3f}")
            except Exception:
                failed_partials.append(f"p{p_num}=NA<{thr:.3f}")

    failed_partials_str = ", ".join(failed_partials) if failed_partials else "(none)"

    out = dict(parsed)
    out["success"] = success
    out["generation_to_threshold"] = generation_to_threshold
    out["best_solution_failed_partials"] = failed_partials_str
    return out

def _aggregate_convergence_metrics(all_metrics: list):
    total_runs = len(all_metrics)
    successful_runs = [m for m in all_metrics if m["success"]]
    num_successful = len(successful_runs)

    if num_successful > 0:
        # gens to threshold
        gens_to_thr = [
            m["generation_to_threshold"] for m in successful_runs
            if m["generation_to_threshold"] is not None
        ]
        if gens_to_thr:
            mean_generations = float(np.mean(gens_to_thr))
            median_generations = float(np.median(gens_to_thr))
            std_generations = float(np.std(gens_to_thr))
        else:
            mean_generations = median_generations = std_generations = 0.0

        # architecture
        hidden_counts = [m["hidden_fields_count"] for m in successful_runs]
        conn_counts = [m["enabled_connections_count"] for m in successful_runs]

        mean_hidden = float(np.mean(hidden_counts))
        median_hidden = float(np.median(hidden_counts))
        std_hidden = float(np.std(hidden_counts))

        mean_conn = float(np.mean(conn_counts))
        median_conn = float(np.median(conn_counts))
        std_conn = float(np.std(conn_counts))

        # convergence rate & improvement
        convergence_rates = []
        for m in successful_runs:
            first_fit = m["fitness_values"][0]
            g_thr = m["generation_to_threshold"]  # 1-based
            if g_thr is None or g_thr <= 0:
                continue
            # internal generations are 0-based from per_generation_overview
            g0 = g_thr - 1
            if g0 not in m["generations"]:
                continue
            idx = m["generations"].index(g0)
            thr_fit = m["fitness_values"][idx]
            convergence_rates.append((thr_fit - first_fit) / float(g_thr))

        mean_conv_rate = float(np.mean(convergence_rates)) if convergence_rates else 0.0
        mean_improvement = float(
            np.mean([m["avg_improvement_per_gen"] for m in successful_runs])
        )
    else:
        mean_generations = median_generations = std_generations = 0.0
        mean_conv_rate = mean_improvement = 0.0
        mean_hidden = median_hidden = std_hidden = 0.0
        mean_conn = median_conn = std_conn = 0.0

    # best/worst runs
    max_fit_run = max(all_metrics, key=lambda m: m["max_fitness"])
    min_fit_run = min(all_metrics, key=lambda m: m["max_fitness"])

    fastest = None
    slowest = None
    most_hidden = None
    most_conn = None
    if num_successful > 0:
        successful_runs = [m for m in all_metrics if m["success"]]
        fastest = min(
            successful_runs,
            key=lambda m: m["generation_to_threshold"]
            if m["generation_to_threshold"] is not None
            else float("inf"),
        )
        slowest = max(
            successful_runs,
            key=lambda m: m["generation_to_threshold"]
            if m["generation_to_threshold"] is not None
            else -1,
        )
        most_hidden = max(successful_runs, key=lambda m: m["hidden_fields_count"])
        most_conn = max(
            successful_runs, key=lambda m: m["enabled_connections_count"]
        )

    return {
        "total_runs": total_runs,
        "successful_runs": num_successful,
        "success_rate": num_successful / total_runs if total_runs > 0 else 0.0,
        "mean_generations_to_threshold": mean_generations,
        "median_generations_to_threshold": median_generations,
        "std_generations_to_threshold": std_generations,
        "mean_convergence_rate": mean_conv_rate,
        "mean_improvement_per_gen": mean_improvement,
        "mean_hidden_fields": mean_hidden,
        "median_hidden_fields": median_hidden,
        "std_hidden_fields": std_hidden,
        "mean_enabled_connections": mean_conn,
        "median_enabled_connections": median_conn,
        "std_enabled_connections": std_conn,
        "all_run_metrics": all_metrics,
        "max_fit_run": max_fit_run,
        "min_fit_run": min_fit_run,
        "fastest_run": fastest,
        "slowest_run": slowest,
        "most_hidden_run": most_hidden,
        "most_connections_run": most_conn,
    }


def _load_experiment_runs_parsed_uncached(base_dir_str: str):
    base = Path(base_dir_str)
    parsed_runs = []
    for rd in base.iterdir():
        if not rd.is_dir():
            continue
        stats_file = rd / "per_generation_overview.txt"
        if not stats_file.exists():
            continue
        m = _parse_run_overview(stats_file)
        if m is None:
            continue
        m["run_dir"] = rd.name
        parsed_runs.append(m)
    return parsed_runs


def _experiment_runs_fingerprint(base_dir: Path) -> str:
    fp1 = _fingerprint_dir(base_dir, "*/per_generation_overview.txt")
    fp2 = _fingerprint_dir(base_dir, "*/evolution_timestamps.txt")
    return f"{fp1}|{fp2}"


@st.cache_data
def _load_experiment_runs_parsed(base_dir_str: str):
    """Target-independent parse of every run's per_generation_overview.txt in base_dir.

    Cached in-memory (per Streamlit session) AND on disk in <base_dir>/.viz_cache/, so
    changing partial-fitness targets in the UI never re-parses (that used to also
    re-scan every statistics/generation_X.txt, ~2.5 GB of I/O per target tweak for a
    33-run experiment), and a Streamlit restart doesn't re-pay the parse either.
    """
    base = Path(base_dir_str)
    cache_dir = _run_cache_dir(base)
    fp = _experiment_runs_fingerprint(base)
    meta_path = cache_dir / "experiment_runs.meta.json"
    meta = _disk_cache_read_json(meta_path)

    if meta is not None and meta.get("fingerprint") == fp:
        cached = _disk_cache_read_json(cache_dir / "experiment_runs.json")
        if cached is not None:
            return cached

    parsed_runs = _load_experiment_runs_parsed_uncached(base_dir_str)

    _disk_cache_write_json(cache_dir / "experiment_runs.json", parsed_runs)
    _disk_cache_write_json(meta_path, {"fingerprint": fp})

    return parsed_runs


def compute_experiment_convergence(base_dir_str: str, partial_targets_items: tuple):
    """
    Go through all run folders (subdirs with per_generation_overview.txt)
    and compute convergence/architecture statistics.
    """
    # convert cached tuple back to dict
    partial_targets = {int(k): float(v) for k, v in partial_targets_items}

    parsed_runs = _load_experiment_runs_parsed(base_dir_str)
    all_metrics = [_evaluate_run_targets(parsed, partial_targets) for parsed in parsed_runs]

    if not all_metrics:
        return {}

    return _aggregate_convergence_metrics(all_metrics)


def render_experiment_convergence(conv: dict, partial_targets: dict):
    """Streamlit UI for multi-run convergence statistics."""
    if not conv:
        st.info("No per_generation_overview.txt files found for this base directory.")
        return

    st.markdown("### Convergence & architecture across runs")

    total = conv["total_runs"]
    succ = conv["successful_runs"]
    rate = conv["success_rate"] * 100.0

    targets_str = ", ".join(
        [f"p{k}≥{v:.3f}" for k, v in sorted(partial_targets.items())]
    ) or "(no partial targets set)"
    st.markdown(
        f"- Analysed **{total}** runs; "
        f"**{succ}** met **all partial fitness targets simultaneously** "
        f"(**{rate:.1f}%** success)."  
        f"- Targets: {targets_str}"
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### Generations to success (successful runs)")
        st.markdown(
            f"- Mean: **{conv['mean_generations_to_threshold']:.2f}**  \n"
            f"- Median: **{conv['median_generations_to_threshold']:.2f}**  \n"
            f"- Std: **{conv['std_generations_to_threshold']:.2f}**"
        )

    with cols[1]:
        st.markdown("#### Convergence speed")
        st.markdown(
            f"- Mean convergence rate (fitness gain/gen): "
            f"**{conv['mean_convergence_rate']:.4f}**  \n"
            f"- Mean fitness improvement/gen: "
            f"**{conv['mean_improvement_per_gen']:.4f}**"
        )

    with cols[2]:
        st.markdown("#### Architecture (successful solutions)")
        st.markdown(
            f"- Hidden fields (mean ± std): "
            f"**{conv['mean_hidden_fields']:.2f} ± {conv['std_hidden_fields']:.2f}**  \n"
            f"- Enabled connections (mean ± std): "
            f"**{conv['mean_enabled_connections']:.2f} ± {conv['std_enabled_connections']:.2f}**"
        )

    # A couple of small histograms / scatter plots
    successful = [
        m for m in conv["all_run_metrics"]
        if m["success"] and m["generation_to_threshold"] is not None
    ]
    if successful:
        col1, col2 = st.columns(2)

        # --- Convergence generations histogram ---
        with col1:
            gens = [m["generation_to_threshold"] for m in successful]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(gens, bins=min(10, len(gens)))
            ax.set_xlabel("generations to success")
            ax.set_ylabel("runs")
            ax.set_title("How many generations runs need\n to meet all partial targets")
            fig.tight_layout()
            show_fig(fig)
            st.caption(
                "Each bar shows how many runs first met all partial targets "
                "in a given generation range. Left = faster convergence."
            )

        # --- Architecture complexity scatter ---
        with col2:
            hidden = [m["hidden_fields_count"] for m in successful]
            conns = [m["enabled_connections_count"] for m in successful]
            labels = [m["run_dir"] for m in successful]

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(hidden, conns)

            # label each point with the run folder name
            for h, c, label in zip(hidden, conns, labels):
                ax.annotate(
                    label,
                    (h, c),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                )

            ax.set_xlabel("hidden fields")
            ax.set_ylabel("enabled connections")
            ax.set_title("Architecture complexity (successful runs)")
            fig.tight_layout()
            show_fig(fig)
            st.caption(
                "Each dot is one successful run. The x-axis is the number of hidden "
                "fields in the final best solution, the y-axis is the number of "
                "enabled connections. Labels show the run directory."
            )

    # Small “notable runs” table (unchanged except formatting)
    st.markdown("#### Notable runs")

    rows = []
    if conv["max_fit_run"] is not None:
        rows.append(
            {
                "label": "Highest max fitness",
                "run": conv["max_fit_run"]["run_dir"],
                "value": conv["max_fit_run"]["max_fitness"],
            }
        )
    if conv["min_fit_run"] is not None:
        rows.append(
            {
                "label": "Lowest max fitness",
                "run": conv["min_fit_run"]["run_dir"],
                "value": conv["min_fit_run"]["max_fitness"],
            }
        )
    if conv["fastest_run"] is not None:
        rows.append(
            {
                "label": "Fastest to threshold",
                "run": conv["fastest_run"]["run_dir"],
                "value": conv["fastest_run"]["generation_to_threshold"],
            }
        )
    if conv["slowest_run"] is not None:
        rows.append(
            {
                "label": "Slowest to threshold",
                "run": conv["slowest_run"]["run_dir"],
                "value": conv["slowest_run"]["generation_to_threshold"],
            }
        )
    if conv["most_hidden_run"] is not None:
        rows.append(
            {
                "label": "Most hidden fields",
                "run": conv["most_hidden_run"]["run_dir"],
                "value": conv["most_hidden_run"]["hidden_fields_count"],
            }
        )
    if conv["most_connections_run"] is not None:
        rows.append(
            {
                "label": "Most enabled connections",
                "run": conv["most_connections_run"]["run_dir"],
                "value": conv["most_connections_run"]["enabled_connections_count"],
            }
        )

    if rows:
        st.table(pd.DataFrame(rows))

    # --- Best performer in each run ---
    if conv.get("all_run_metrics"):
        st.markdown("#### Success per run")

        rows = []
        for m in conv["all_run_metrics"]:
            rows.append(
                {
                    "run id": m.get("run_dir"),
                    "best solution fitness": m.get("best_solution_fitness"),
                    "failed partial fitness (last gen)": m.get("best_solution_failed_partials"),
                    "above threshold?": "✅ yes" if m.get("success") else "✖ no",
                    "generations to threshold": m.get("generation_to_threshold"),
                    "species (total/active)": f"{m.get('final_species_total')}/{m.get('final_species_active')}",
                    "duration": m.get("duration_human"),
                    "topology (hidden + enabled conns)": f"{m.get('hidden_fields_count')} + {m.get('enabled_connections_count')}",
                }
            )

        df_runs = pd.DataFrame(rows)

        # Sort: successful first, then by earliest success generation, then by best fitness
        df_runs["_succ"] = df_runs["above threshold?"].map({"✅ yes": 1, "✖ no": 0}).fillna(0)
        df_runs = (
            df_runs.sort_values(
                by=["_succ", "generations to threshold", "best solution fitness"],
                ascending=[False, True, False],
                na_position="last",
            )
            .drop(columns=["_succ"], errors="ignore")
            .reset_index(drop=True)
        )

        st.dataframe(df_runs, width="stretch")
        st.caption(
            "Main diagnosis per run. A run is marked successful when there exists at least one generation "
            "where all partial fitness components are ≥ their targets in the same generation. "
            "`generations to threshold` is the first such generation (1-based). "
            "`failed partial fitness (last gen)` lists which partial targets the best-total-fitness solution failed, if any."
        )


def export_run_markdown(run_dir_str: str, target_fitness: float = 0.9, out_path: str | None = None) -> str:
    """
    Create a markdown summary for a single run directory and write it to disk.

    Parameters
    ----------
    run_dir_str : str
        Path to the run directory (the one that has per_generation_overview.txt).
    target_fitness : float
        Target fitness to use in the report.
    out_path : str | None
        Optional explicit output path for the .md file.
        If None, a file '<run_dir_name>_summary.md' is created inside the run dir.

    Returns
    -------
    str
        The path to the written markdown file (as string).
    """
    run_dir = Path(run_dir_str).expanduser().resolve()
    df = load_overview(run_dir_str)
    gens_tuple = tuple(df["generation"].tolist())
    species_meta = compute_species_meta(run_dir_str, gens_tuple)

    # ---------- FITNESS STATS (reusing logic from render_fitness_stats) ----------
    final_row = df.iloc[-1]
    final_gen = int(final_row["generation"])
    best_final = float(final_row["best_fitness"])
    avg_final = float(final_row["avg_fitness"])

    max_best = float(df["best_fitness"].max())
    gen_max_best = int(df.loc[df["best_fitness"].idxmax(), "generation"])

    reached = df[df["best_fitness"] >= target_fitness]
    if not reached.empty:
        gen_target = int(reached["generation"].iloc[0])
        best_at_target = float(reached["best_fitness"].iloc[0])
    else:
        gen_target = None
        best_at_target = None

    best_series = df["best_fitness"]
    improved = best_series.diff().fillna(0) > 1e-9
    longest_stagnation = 0
    current = 0
    for imp in improved[1:]:
        if imp:
            longest_stagnation = max(longest_stagnation, current)
            current = 0
        else:
            current += 1
    longest_stagnation = max(longest_stagnation, current)

    auc_best = float(best_series.mean())
    auc_avg = float(df["avg_fitness"].mean())

    # ---------- SPECIES STATS (reusing logic from render_species_stats) ----------
    last = df.iloc[-1]
    final_species = int(last["num_species"])
    final_active = int(last["num_active_species"])

    avg_species = float(df["num_species"].mean())
    avg_active = float(df["num_active_species"].mean())

    max_active_species = int(df["num_active_species"].max())
    gen_max_active = int(df.loc[df["num_active_species"].idxmax(), "generation"])

    total_species = len(species_meta)
    lifespans = []
    max_members_list = []
    offspring_list = []
    active_final = 0
    for sid, m in species_meta.items():
        span = m["last_gen"] - m["first_gen"] + 1
        lifespans.append(span)
        max_members_list.append(m["max_members"])
        offspring_list.append(m["total_offspring"])
        if m["last_gen"] == final_gen and not m["last_extinct"]:
            active_final += 1

    extinct_species = total_species - active_final

    if lifespans:
        avg_lifespan = sum(lifespans) / len(lifespans)
        max_lifespan = max(lifespans)
        max_life_sid = [
            sid
            for sid, m in species_meta.items()
            if m["last_gen"] - m["first_gen"] + 1 == max_lifespan
        ][0]
    else:
        avg_lifespan = 0.0
        max_lifespan = 0
        max_life_sid = None

    avg_max_members = sum(max_members_list) / len(max_members_list) if max_members_list else 0.0
    avg_offspring = sum(offspring_list) / len(offspring_list) if offspring_list else 0.0

    # ---------- TOPOLOGY STATS (reusing render_topology_stats) ----------
    first = df.iloc[0]
    g0 = float(first["avg_genome_size"])
    gN = float(last["avg_genome_size"])
    f0 = float(first["avg_field_genes"])
    fN = float(last["avg_field_genes"])
    c0 = float(first["avg_conn_genes"])
    cN = float(last["avg_conn_genes"])

    gens = last["generation"] - first["generation"]
    gens = gens if gens > 0 else 1

    avg_conn_per_field_final = (cN / fN) if fN > 0 else 0.0

    # ---------- POPULATION-LEVEL KERNEL USAGE ----------
    (
        df_kernel_usage,
        field_overall_counts,
        field_overall_perc,
        inter_overall_counts,
        inter_overall_perc,
    ) = compute_population_kernel_usage(run_dir_str, gens_tuple)

    def fmt_kernel_counts(counts: dict, perc: dict) -> str:
        if not counts:
            return "none"
        parts = []
        for k in sorted(counts.keys()):
            p = perc.get(k, 0.0)
            parts.append(f"{k}: {counts[k]} ({p:.1f}%)")
        return ", ".join(parts)

    field_kernel_line = fmt_kernel_counts(field_overall_counts, field_overall_perc)
    inter_kernel_line = fmt_kernel_counts(inter_overall_counts, inter_overall_perc)

    # ---------- BEST-SOLUTION GENOME ----------
    # Use final generation's best solution
    elements = load_best_solution_architecture(run_dir_str, final_gen)
    if elements is not None:
        df_fields, df_inter = summarize_best_solution_genome(elements)
        fields_md = df_to_markdown_table(df_fields)
        inter_md = df_to_markdown_table(df_inter)
    else:
        fields_md = "_No best-solution JSON found for this generation._"
        inter_md = ""

    # ---------- BUILD MARKDOWN TEXT ----------
    timestamp = datetime.now().strftime("%Y-%m-%d %Hh%Mm%Ss")

    lines = []

    # Header: path 
    lines.append(f"# {run_dir}")
    lines.append("")  # keep one blank line

    # ---- Fitness statistics ----
    lines.append("## Fitness statistics")
    lines.append("")
    lines.append(f"Final generation (g = {final_gen + 1})")
    lines.append(f"- Best fitness: {best_final:.4f}")
    lines.append(f"- Target fitness: {target_fitness:.2f}")
    lines.append(f"- Average fitness: {avg_final:.4f}")
    lines.append("")
    lines.append("Overall")
    lines.append(f"- Max best fitness: {max_best:.4f} (reached at generation {gen_max_best})")
    lines.append(f"- Mean best fitness over run (AUC): {auc_best:.4f}")
    lines.append(f"- Mean average fitness over run (AUC): {auc_avg:.4f}")
    lines.append(f"- Longest stagnation period: {longest_stagnation} generations")
    if gen_target is not None:
        lines.append(
            f"- Target fitness {target_fitness:.3f} first reached at generation {gen_target} "
            f"(best fitness ≈ {best_at_target:.4f})"
        )
    else:
        lines.append(f"- Target fitness {target_fitness:.3f} was not reached.")

    lines.append("")
    lines.append("## Species statistics")
    lines.append("")
    lines.append(f"Final generation (g = {final_gen + 1})")
    lines.append(f"- Species: {final_species}")
    lines.append(f"- Active species: {final_active}")
    lines.append("")
    lines.append("Across run")
    lines.append(f"- Total distinct species created: {total_species}")
    lines.append(f"- Species extinct by final generation: {extinct_species}")
    lines.append(f"- Average species per generation: {avg_species:.2f}")
    lines.append(f"- Average active species per generation: {avg_active:.2f}")
    lines.append(f"- Max active species in a generation: {max_active_species} (at g={gen_max_active})")
    lines.append("")
    lines.append("Species lifetime & size")
    lines.append(f"- Average species lifespan: {avg_lifespan:.2f} generations")
    lines.append(f"- Longest-lived species: {max_life_sid} (lifespan {max_lifespan})")
    lines.append(f"- Average max members per species: {avg_max_members:.2f}")
    lines.append(f"- Average offspring per species: {avg_offspring:.2f}")

    lines.append("")
    lines.append("## Topology statistics")
    lines.append("")
    lines.append(f"Final generation (g = {final_gen + 1})")
    lines.append(f"- Avg genome size: {gN:.2f}")
    lines.append(f"- Avg field genes: {fN:.2f}")
    lines.append(f"- Avg connection genes: {cN:.2f}")
    lines.append("")
    lines.append("Growth over run")
    lines.append(f"- Genome size change: {gN-g0:+.2f} (≈ {(gN-g0)/gens:+.3f}/gen)")
    lines.append(f"- Field genes change: {fN-f0:+.2f} (≈ {(fN-f0)/gens:+.3f}/gen)")
    lines.append(f"- Connection genes change: {cN-c0:+.2f} (≈ {(cN-c0)/gens:+.3f}/gen)")
    lines.append("")
    lines.append("Ratios")
    lines.append(f"- Avg connections per field at final gen: {avg_conn_per_field_final:.2f}")
    lines.append("")
    lines.append("Population-level kernel usage")
    lines.append(f"- Field kernels: {field_kernel_line}")
    lines.append(f"- Interaction kernels: {inter_kernel_line}")

    lines.append("")
    lines.append(
        f"Genome representation of the highest-performing solution for generation "
        f"{final_gen} with f = {best_final:.4f}"
    )
    lines.append("")
    lines.append("### Field genes")
    lines.append("")
    lines.append(fields_md)
    if inter_md:
        lines.append("")
        lines.append("### Interaction genes")
        lines.append("")
        lines.append(inter_md)


    markdown_text = "\n".join(lines)

    # ---------- WRITE FILE ----------
    if out_path is None:
        out_path = run_dir / f"{run_dir.name}_summary.md"
    else:
        out_path = Path(out_path).expanduser().resolve()

    out_path.write_text(markdown_text, encoding="utf-8")
    return str(out_path)

def export_experiment_markdown(
    base_dir_str: str,
    partial_targets_items: tuple | None = None,
    out_path: str | None = None,
) -> str:
    """
    Export experiment-level statistics (across all runs in base_dir)
    to a markdown file.

    Returns the path to the written .md file.
    """
    base_dir = Path(base_dir_str).expanduser().resolve()

    if partial_targets_items is None:
        partial_targets_items = tuple()

    # reuse your cached computations
    conv = compute_experiment_convergence(base_dir_str, tuple(partial_targets_items))
    agg_totals, df_totals = compute_experiment_totals(base_dir_str)

    lines: list[str] = []

    # Header
    lines.append(f"# {base_dir}")
    lines.append("")
    lines.append("## Experiment-level statistics across runs")
    lines.append("")

    if not conv:
        lines.append("_No per_generation_overview.txt files found in this base directory._")
        markdown_text = "\n".join(lines)

        if out_path is None:
            out_path = base_dir / f"{base_dir.name}_experiment_summary.md"
        else:
            out_path = Path(out_path).expanduser().resolve()
        out_path.write_text(markdown_text, encoding="utf-8")
        return str(out_path)

    # ---------- Main convergence & architecture stats ----------
    total = conv["total_runs"]
    succ = conv["successful_runs"]
    rate = conv["success_rate"] * 100.0

    partial_targets = {int(k): float(v) for k, v in partial_targets_items}
    targets_str = ", ".join(
        [f"p{k}≥{v:.3f}" for k, v in sorted(partial_targets.items())]
    ) or "(no partial targets set)"

    lines.append(
        f"Analysed {total} runs; {succ} met all partial fitness targets simultaneously "
        f"({rate:.1f}% success). Targets: {targets_str}"
    )
    lines.append("")
    lines.append("### Generations to threshold (successful runs)")
    lines.append(f"- Mean: {conv['mean_generations_to_threshold']:.2f}")
    lines.append(f"- Median: {conv['median_generations_to_threshold']:.2f}")
    lines.append(f"- Std: {conv['std_generations_to_threshold']:.2f}")
    lines.append("")
    lines.append("### Convergence speed")
    lines.append(
        f"- Mean convergence rate (fitness gain/gen): {conv['mean_convergence_rate']:.4f}"
    )
    lines.append(
        f"- Mean fitness improvement/gen: {conv['mean_improvement_per_gen']:.4f}"
    )
    lines.append("")
    lines.append("### Architecture (successful solutions)")
    lines.append(
        f"- Hidden fields (mean ± std): "
        f"{conv['mean_hidden_fields']:.2f} ± {conv['std_hidden_fields']:.2f}"
    )
    lines.append(
        f"- Enabled connections (mean ± std): "
        f"{conv['mean_enabled_connections']:.2f} ± {conv['std_enabled_connections']:.2f}"
    )

    # ---------- Time / performance (from agg_totals) ----------
    lines.append("")
    lines.append("### Time / performance")
    dur_mean = agg_totals.get("duration_hours_mean")
    spg_mean = agg_totals.get("seconds_per_generation_mean")

    if dur_mean is not None:
        lines.append(f"- Average run duration: {dur_mean:.2f} h")
    if spg_mean is not None:
        lines.append(f"- Avg. time per generation: {spg_mean:.2f} s/gen")
    if dur_mean is None and spg_mean is None:
        lines.append("- _No timing information available._")

    # ---------- Notable runs table ----------
    lines.append("")
    lines.append("### Notable runs")
    rows_notable = []

    if conv.get("max_fit_run") is not None:
        rows_notable.append(
            {
                "label": "Highest max fitness",
                "run": conv["max_fit_run"]["run_dir"],
                "value": conv["max_fit_run"]["max_fitness"],
            }
        )
    if conv.get("min_fit_run") is not None:
        rows_notable.append(
            {
                "label": "Lowest max fitness",
                "run": conv["min_fit_run"]["run_dir"],
                "value": conv["min_fit_run"]["max_fitness"],
            }
        )
    if conv.get("fastest_run") is not None:
        rows_notable.append(
            {
                "label": "Fastest to threshold",
                "run": conv["fastest_run"]["run_dir"],
                "value": conv["fastest_run"]["generation_to_threshold"],
            }
        )
    if conv.get("slowest_run") is not None:
        rows_notable.append(
            {
                "label": "Slowest to threshold",
                "run": conv["slowest_run"]["run_dir"],
                "value": conv["slowest_run"]["generation_to_threshold"],
            }
        )
    if conv.get("most_hidden_run") is not None:
        rows_notable.append(
            {
                "label": "Most hidden fields",
                "run": conv["most_hidden_run"]["run_dir"],
                "value": conv["most_hidden_run"]["hidden_fields_count"],
            }
        )
    if conv.get("most_connections_run") is not None:
        rows_notable.append(
            {
                "label": "Most enabled connections",
                "run": conv["most_connections_run"]["run_dir"],
                "value": conv["most_connections_run"]["enabled_connections_count"],
            }
        )

    if rows_notable:
        df_notable = pd.DataFrame(rows_notable)
        lines.append("")
        lines.append(df_to_markdown_table(df_notable))
    else:
        lines.append("")
        lines.append("_No notable runs information available._")

    

    # ---------- Success per run table ----------
    lines.append("")
    lines.append("### Success per run")
    lines.append("")
    lines.append("A run is **successful** if there exists at least one generation where **all partial fitness targets are met simultaneously** (same generation).")
    lines.append("")

    if conv.get("all_run_metrics"):
        rows_success = []
        for m in conv["all_run_metrics"]:
            rows_success.append(
                {
                    "run id": m.get("run_dir"),
                    "best solution fitness": m.get("best_solution_fitness"),
                    "failed partial fitness (last gen)": m.get("best_solution_failed_partials"),
                    "above threshold?": "yes" if m.get("success", False) else "no",
                    "generations to threshold": m.get("generation_to_threshold", None),
                    "species (total/active)": f"{m.get('final_species_total')}/{m.get('final_species_active')}",
                    "duration": m.get("duration_human"),
                    "topology (hidden + enabled conns)": f"{m.get('hidden_fields_count')} + {m.get('enabled_connections_count')}",
                }
            )
        df_success = (
            pd.DataFrame(rows_success)
            .sort_values(
                ["above threshold?", "generations to threshold", "best solution fitness"],
                ascending=[False, True, False],
                na_position="last",
            )
            .reset_index(drop=True)
        )
        lines.append(df_to_markdown_table(df_success))
    else:
        lines.append("_No run-level information available._")

    markdown_text = "\n".join(lines)

    if out_path is None:
        out_path = base_dir / f"{base_dir.name}_experiment_summary.md"
    else:
        out_path = Path(out_path).expanduser().resolve()

    out_path.write_text(markdown_text, encoding="utf-8")
    return str(out_path)

# =========================
# Main app
# =========================
#
# The @st.fragment blocks below own a slider/number_input whose every drag used to
# rerun the ENTIRE page (re-render every tab's plots and widgets). Wrapping them in
# a fragment confines a widget interaction's rerun to just that block.


@st.fragment
def render_fitness_view(df: pd.DataFrame, gens_tuple: tuple, selected_run_path: str):
    st.markdown("---")

    min_fit = float(min(df["avg_fitness"].min(), df["best_fitness"].min(), 0.0))
    max_fit = float(max(df["avg_fitness"].max(), df["best_fitness"].max(), 1.0))
    slider_min = min_fit
    slider_max = max_fit if max_fit > min_fit else min_fit + 1.0

    c1, c2 = st.columns([4, 1])
    with c1:
        target = st.slider(
            "Overall target fitness (for main plot)",
            min_value=slider_min,
            max_value=slider_max,
            value=float(st.session_state["target_fitness"]),
            step=(slider_max - slider_min) / 100.0 if slider_max > slider_min else 0.01,
        )
    with c2:
        target = st.number_input(
            "Target (manual)",
            value=float(target),
            min_value=slider_min,
            max_value=slider_max,
        )

    st.session_state["target_fitness"] = float(target)

    # ---------- Partial targets + "success" generation ----------
    partial_df = compute_partial_fitness(selected_run_path, gens_tuple)
    partial_targets = st.session_state["partial_targets"]
    first_ok = None
    gens_ok = []

    if partial_df is not None and not partial_df.empty:
        # Determine number of partial components
        num_partial = 0
        for col in partial_df.columns:
            if col.startswith("best_p"):
                idx = int(col.replace("best_p", ""))
                num_partial = max(num_partial, idx)

        if num_partial > 0:
            with st.expander("Targets for partial fitness components", expanded=False):
                rows = math.ceil(num_partial / 3)
                comp = 1
                for _ in range(rows):
                    cols = st.columns(3)
                    for c in cols:
                        if comp > num_partial:
                            break

                        best_col = f"best_p{comp}"
                        avg_col = f"avg_p{comp}"
                        if best_col not in partial_df.columns or avg_col not in partial_df.columns:
                            comp += 1
                            continue

                        if comp not in partial_targets:
                            partial_targets[comp] = float(st.session_state["target_fitness"])

                        col_min = float(
                            min(partial_df[best_col].min(), partial_df[avg_col].min(), 0.0)
                        )
                        col_max = float(
                            max(partial_df[best_col].max(), partial_df[avg_col].max(), 1.0)
                        )

                        with c:
                            st.markdown(f"**partial {comp}**")
                            val = st.number_input(
                                f"target p{comp}",
                                value=float(partial_targets[comp]),
                                min_value=col_min,
                                max_value=col_max,
                                key=f"num_p{comp}",
                            )
                            partial_targets[comp] = float(val)

                        comp += 1

            st.session_state["partial_targets"] = partial_targets

            # generations where all partial best fitnesses >= their targets
            gens_ok = generations_all_partial_meet_targets(partial_df, partial_targets)
            if gens_ok:
                first_ok = int(gens_ok[0])
                st.info(
                    "Generations where **all partial best fitnesses** are ≥ their targets: "
                    f"{gens_ok} (first at generation {first_ok})."
                )
            else:
                st.info(
                    "In this run, **no generation** reached all partial best fitness targets simultaneously."
                )

    # ---------- Plots + stats (threshold based on partials) ----------
    st.markdown("### Total fitness")
    fig_total = plot_total_fitness(df, st.session_state["target_fitness"], success_generations=gens_ok, success_label="All partial targets met")
    show_fig(fig_total)

    st.markdown("---")
    stats_col, partial_col = st.columns([1, 3])

    with stats_col:
        render_fitness_stats(
            df,
            st.session_state["target_fitness"],
            partial_threshold_generation=first_ok,
        )

    with partial_col:
        plot_partial_fitness_grid(partial_df, st.session_state["partial_targets"])


@st.fragment
def render_species_genome_inspector(df: pd.DataFrame, selected_run_path: str):
    st.markdown("#### Species genome (per generation)")

    min_gen = int(df["generation"].min())
    max_gen = int(df["generation"].max())
    gen_sel = st.slider(
        "Generation to inspect",
        min_value=min_gen,
        max_value=max_gen,
        value=max_gen,
    )

    species_list = get_species_for_generation(selected_run_path, gen_sel)

    if not species_list:
        st.info(f"No species file found or parsable for generation {gen_sel}.")
    else:
        active_species = [
            s for s in species_list if (not s["extinct"]) and s["members"] > 0
        ]

        st.markdown(
            f"Generation **{gen_sel}** — active species: **{len(active_species)}** "
            f"(total logged species in file: {len(species_list)})"
        )

        if not active_species:
            st.info("No active species with members > 0 in this generation.")
        else:
            for s in active_species:
                improved_str = "yes" if s["improved"] else "no"
                header = (
                    f"Species {s['id']} "
                    f"(age {s['age']}, members {s['members']}, "
                    f"offspring {s['offspring']}, improved this gen: {improved_str}, "
                    f"gens since imp.: {s['gens_since_improvement']})"
                )
                with st.expander(header, expanded=False):
                    st.markdown("**Representative solution (raw log snippet):**")
                    if s["rep_raw"]:
                        st.code(s["rep_raw"], language="text")
                    else:
                        st.write("_none_")

                    st.markdown("**Champion solution (raw log snippet):**")
                    if s["champ_raw"]:
                        st.code(s["champ_raw"], language="text")
                    else:
                        st.write("_none_")


@st.fragment
def render_population_kernel_usage(df: pd.DataFrame, selected_run_path: str):
    gens_tuple = tuple(df["generation"].tolist())

    st.markdown("#### Population-level kernel usage")
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        gen_step = st.number_input(
            "Sample every Nth generation",
            min_value=1,
            max_value=max(1, len(gens_tuple)),
            value=min(10, max(1, len(gens_tuple))),
            help="A run can have 1000 solutions x 200 generations of genome "
            "files; reading all of them is slow, so this samples generations "
            "and solutions instead of scanning everything.",
        )
    with sample_col2:
        max_solutions_per_gen = st.number_input(
            "Max solutions sampled per generation",
            min_value=1,
            max_value=1000,
            value=100,
        )

    (
        df_kernel_usage,
        field_overall_counts,
        field_overall_perc,
        inter_overall_counts,
        inter_overall_perc,
    ) = compute_population_kernel_usage(
        selected_run_path, gens_tuple, int(gen_step), int(max_solutions_per_gen)
    )

    if df_kernel_usage is not None and not df_kernel_usage.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_fu = plot_kernel_usage_time(df_kernel_usage, kind="field")
            if fig_fu is not None:
                show_fig(fig_fu)
        with col_b:
            fig_iu = plot_kernel_usage_time(df_kernel_usage, kind="inter")
            if fig_iu is not None:
                show_fig(fig_iu)

        # Overall summary across entire run
        def fmt_counts(counts, perc):
            if not counts:
                return "none"
            parts = []
            for k in sorted(counts.keys()):
                p = perc.get(k, 0.0)
                parts.append(f"{k}: {counts[k]} ({p:.1f}%)")
            return ", ".join(parts)

        st.caption(
            f"Sampled every {int(gen_step)} generation(s), up to "
            f"{int(max_solutions_per_gen)} solutions per sampled generation "
            "-- not an exhaustive count over the full population."
        )
        st.markdown(
            f"- **Field kernels:** {fmt_counts(field_overall_counts, field_overall_perc)}"
        )
        st.markdown(
            f"- **Interaction kernels:** {fmt_counts(inter_overall_counts, inter_overall_perc)}"
        )

    else:
        st.info(
            "No population-level kernel usage data found. "
            "This usually means there are no genome files under "
            "`solutions/gen X/<solution>/*.dnf` for this run."
        )


@st.fragment
def render_best_solution_genome(df: pd.DataFrame, selected_run_path: str):
    min_gen = int(df["generation"].min())
    max_gen = int(df["generation"].max())
    gen_sel = st.slider(
        "Generation to inspect (best solution)",
        min_value=min_gen,
        max_value=max_gen,
        value=max_gen,
    )

    # best fitness for this generation (for the title)
    row_sel = df[df["generation"] == gen_sel]
    if not row_sel.empty:
        best_f = float(row_sel["best_fitness"].iloc[0])
    else:
        best_f = float("nan")

    elements = load_best_solution_architecture(selected_run_path, gen_sel)
    if elements is None:
        st.info(
            "No best-solution JSON found in "
            "`best_solutions/prev_generations` for this generation."
        )
    else:
        # Genome tables
        df_fields, df_inter = summarize_best_solution_genome(elements)

        st.markdown(
            f"#### Genome representation of the highest-performing solution "
            f"for generation {gen_sel} with f = {best_f:.4f}"
        )

        if df_fields is not None and not df_fields.empty:
            st.markdown("**Field genes**")
            st.dataframe(df_fields, width="stretch")
        else:
            st.info("No field genes found in this solution.")

        if df_inter is not None and not df_inter.empty:
            st.markdown("**Interaction genes**")
            st.dataframe(df_inter, width="stretch")
        else:
            st.info("No interaction genes (field-to-field kernels) found in this solution.")

        # --- NEW: kernel usage statistics for this genome ---
        field_counts, field_perc, inter_counts, inter_perc = compute_kernel_usage_stats(elements)

        st.markdown("#### Kernel usage in this genome")

        if field_counts:
            parts = [
                f"{k}: {field_counts[k]} fields ({field_perc[k]:.1f}%)"
                for k in sorted(field_counts.keys())
            ]
            st.markdown(
                "**Field kernels:** " + ", ".join(parts)
            )
        else:
            st.markdown("**Field kernels:** no kernels associated with fields.")

        if inter_counts:
            parts = [
                f"{k}: {inter_counts[k]} interaction kernels ({inter_perc[k]:.1f}%)"
                for k in sorted(inter_counts.keys())
            ]
            st.markdown(
                "**Interaction kernels (field–field):** " + ", ".join(parts)
            )
        else:
            st.markdown("**Interaction kernels (field–field):** none in this genome.")

        # Optional: graph view
        with st.expander("Graph view of field interactions", expanded=False):
            g, pos, field_nodes, kernel_nodes = build_topology_graph(elements)
            fig_top = plot_topology_graph(g, pos, field_nodes, kernel_nodes)
            show_fig(fig_top)


def main():
    st.set_page_config(page_title="neat-dnfs evolution overview", layout="wide")

    if "view" not in st.session_state:
        st.session_state["view"] = "Fitness"
    if "target_fitness" not in st.session_state:
        st.session_state["target_fitness"] = 0.9
    if "partial_targets" not in st.session_state:
        st.session_state["partial_targets"] = {}

    left_col, main_col = st.columns([1, 5])

    # ---------- LEFT PANEL ----------
    with left_col:
        logo_candidate = Path("../resources/images/logo.png")
        if logo_candidate.exists():
            st.image(str(logo_candidate.resolve()), width="stretch")
        else:
            st.markdown("### neat-dnfs")

        st.markdown("**Base experiment directory**")
        default_base = Path("../data").resolve()
        base_dir_str = st.text_input(
            label="Base experiment directory path",
            value=str(default_base),
            help="Directory containing your run folders (each with per_generation_overview.txt).",
            label_visibility="collapsed",  # hides the label visually but keeps it non-empty
        )
        base_dir = Path(base_dir_str).expanduser()

        if not base_dir.exists() or not base_dir.is_dir():
            st.error(f"Base directory does not exist or is not a directory:\n{base_dir}")
            st.stop()

        runs = find_runs_with_overview(base_dir)
        if not runs:
            st.warning("No subfolders with per_generation_overview.txt found.")
            st.stop()

        run_names = [name for name, _ in runs]
        selected_run_name = st.selectbox("Selected run:", run_names)
        selected_run_path = str(dict(runs)[selected_run_name])

        st.markdown(f"<small>{selected_run_path}</small>", unsafe_allow_html=True)

        # --- Experiment + export controls (left side) ---
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

        with ctrl_col1:
            if st.button("Experiment", use_container_width=True):
                st.session_state["view"] = "Experiment"

        with ctrl_col2:
            if st.button("Export run .md", use_container_width=True):
                md_path = export_run_markdown(
                    selected_run_path,
                    target_fitness=float(st.session_state.get("target_fitness", 0.9)),
                )
                st.success(f"Run summary exported to:\n{md_path}")

        with ctrl_col3:
            if st.button("Export experiment .md", use_container_width=True):
                partial_targets = st.session_state.get("partial_targets", {})
                targets_items = tuple(sorted((int(k), float(v)) for k, v in partial_targets.items()))
                md_path = export_experiment_markdown(
                    base_dir_str,
                    partial_targets_items=targets_items,
                )
                st.success(f"Experiment summary exported to:\n{md_path}")

    # ---------- MAIN PANEL ----------
    with main_col:
        h1_col, h2_col = st.columns([3, 2])
        with h1_col:
            st.markdown("## neat-dnfs evolution overview dashboard")
        with h2_col:
            bcols = st.columns(4)
            with bcols[0]:
                if st.button("Fitness", width="stretch"):
                    st.session_state["view"] = "Fitness"
            with bcols[1]:
                if st.button("Species", width="stretch"):
                    st.session_state["view"] = "Species"
            with bcols[2]:
                if st.button("Topology", width="stretch"):
                    st.session_state["view"] = "Topology"
            with bcols[3]:
                if st.button("Mutations", width="stretch"):
                    st.session_state["view"] = "Mutations"

        view = st.session_state["view"]
        df = load_overview(selected_run_path)
        gens_tuple = tuple(df["generation"].tolist())

        # ---------------- FITNESS ----------------
        if view == "Fitness":
            render_fitness_view(df, gens_tuple, selected_run_path)

        # ---------------- SPECIES ----------------
        elif view == "Species":
            st.markdown("---")

            top_left, top_right = st.columns(2)
            with top_left:
                fig_sc = plot_species_counts(df)
                show_fig(fig_sc)
            with top_right:
                fig_innov = plot_innovation_growth(df)
                show_fig(fig_innov)

            st.markdown("---")
            bottom_left, bottom_right = st.columns([1, 3])

            species_meta = compute_species_meta(selected_run_path, gens_tuple)

            with bottom_left:
                render_species_stats(df, species_meta)

            with bottom_right:
                render_species_genome_inspector(df, selected_run_path)

        # ---------------- TOPOLOGY ----------------
        elif view == "Topology":
            st.markdown("---")

            # Top: genome topology curves + stats
            top_left, top_right = st.columns(2)
            with top_left:
                fig_gen = plot_genome_topology_curves(df)
                show_fig(fig_gen)
            with top_right:
                render_topology_stats(df)

            st.markdown("---")
            render_population_kernel_usage(df, selected_run_path)

            st.markdown("---")
            render_best_solution_genome(df, selected_run_path)


        # ---------------- MUTATIONS (placeholder) ----------------
               # ---------------- MUTATIONS ----------------
        elif view == "Mutations":
            st.markdown("---")

            mut_events = compute_mutation_events(selected_run_path, gens_tuple)

            if mut_events.empty:
                st.info(
                    "No mutation logs found in statistics/generation_X.txt "
                    "(the 'last mutations{...}' field appears empty)."
                )
            else:
                # --- Top row: activity + category breakdown ---
                top_row_left, top_row_right = st.columns(2)
                with top_row_left:
                    fig_muts = plot_mutations_per_generation(mut_events)
                    show_fig(fig_muts)
                    st.caption("Total number of mutation events applied in each generation.")

                with top_row_right:
                    fig_cat = plot_mutation_categories(mut_events)
                    show_fig(fig_cat)
                    st.caption("How mutation events are distributed across high-level categories.")

                # --- Middle: most beneficial mutations (bar plot) ---
                st.markdown("---")
                fig_eff = plot_mutation_effectiveness(mut_events)
                if fig_eff is not None:
                    show_fig(fig_eff)
                    st.caption(
                        "Mutations that, on average, appear in higher-fitness individuals "
                        "compared to the global mean. Bar length shows the improvement."
                    )

                # --- Tables: frequency + effectiveness ---
                st.markdown("---")
                render_mutation_tables(mut_events)

                # --- Mutations involved when crossing the target fitness ---
                target = float(st.session_state.get("target_fitness", 0.9))
                crossing_df = compute_target_crossing_mutations(mut_events, df, target)

                if not crossing_df.empty:
                    st.markdown("#### Mutations present when best fitness crossed the target")
                    st.caption(
                        "Generations where best fitness first moved from below the selected "
                        f"target (currently {target:.3f}) to above it. "
                        "For those generations, this table shows which mutations were present "
                        "in above-target individuals. This is an approximation of "
                        "\"mutations that pushed solutions over the threshold\"."
                    )
                    show_cols = crossing_df.copy()
                    show_cols = show_cols.rename(
                        columns={
                            "generation": "generation",
                            "mutation_raw": "mutation description",
                            "category": "mutation category",
                            "occurrences_in_gen": "times this mutation appears in that generation",
                            "mean_fitness_in_gen": "mean fitness of its carriers in that generation",
                        }
                    )
                    show_cols = show_cols.sort_values(
                        ["generation", "mean_fitness_in_gen"], ascending=[True, False]
                    )
                    st.dataframe(show_cols.head(20), width="stretch")

                # --- Per-generation most impactful mutation timeline ---
                per_gen_best = compute_per_generation_best_mutation(mut_events)
                if not per_gen_best.empty:
                    st.markdown("#### Per-generation most beneficial mutation")
                    st.caption(
                        "For each generation, this considers mutations that appear at least "
                        "three times and selects the one whose carriers have the largest "
                        "advantage over that generation's average fitness."
                    )

                    fig_pg = plot_best_mutation_timeline(per_gen_best)
                    show_fig(fig_pg)

                    st.markdown("Top generations by mutation impact:")
                    st.dataframe(
                        per_gen_best.sort_values("delta_vs_gen", ascending=False)
                        .head(10)
                        .rename(
                            columns={
                                "generation": "generation",
                                "mutation": "mutation description",
                                "category": "mutation category",
                                "occurrences": "count in that generation",
                                "mean_fitness": "mean fitness of its carriers",
                                "delta_vs_gen": "Δ vs generation mean fitness",
                            }
                        ),
                        width="stretch",
                    )
        
        # ---------------- EXPERIMENT (multi-run stats) ----------------
        elif view == "Experiment":
            st.markdown("---")

            # reuse the same base_dir_str that you typed on the left
            base_dir_str_for_runs = base_dir_str

            st.markdown("### Experiment-level statistics across runs")

            # Success criterion (refactor): run is successful if there exists at least one generation
            # where *all* partial fitness components meet/exceed their targets simultaneously.
            if "partial_targets" not in st.session_state:
                st.session_state["partial_targets"] = {}

            partial_targets = dict(st.session_state["partial_targets"])

            # If not set yet, infer how many partial components exist from the first run we find.
            # Default each partial target to the current global target_fitness.
            if not partial_targets:
                try:
                    base_dir = Path(base_dir_str_for_runs)
                    runs = find_runs_with_overview(base_dir)
                    if runs:
                        _, first_run = runs[0]
                        df_first = load_overview(str(first_run))

                        # Determine how many partial fitness components exist.
                        # Prefer per_generation_overview.txt (it is the source of truth for partials),
                        # because some runs may not have per-partial series in statistics.
                        n_parts = 0
                        try:
                            if "partial_best" in df_first.columns:
                                n_parts = int(
                                    max(
                                        (
                                            len(v)
                                            for v in df_first["partial_best"]
                                            if isinstance(v, (list, tuple))
                                        ),
                                        default=0,
                                    )
                                )
                        except Exception:
                            n_parts = 0

                        # Fallback: infer from computed partial-fitness dataframe columns.
                        if n_parts == 0:
                            gens_first = tuple(df_first["generation"].tolist())
                            pf = compute_partial_fitness(str(first_run), gens_first)
                            if pf is not None:
                                n_parts = len([c for c in pf.columns if c.startswith("best_p")])

                        default_thr = float(st.session_state.get("target_fitness", 0.9))
                        for i in range(1, n_parts + 1):
                                partial_targets[i] = default_thr
                except Exception:
                    pass

            with st.expander("Partial fitness targets (success criteria)", expanded=True):
                if not partial_targets:
                    st.info(
                        "No partial fitness components detected yet. "
                        "Make sure your runs have statistics/generation_X.txt files."
                    )
                else:
                    cols = st.columns(min(6, max(1, len(partial_targets))))
                    for idx, comp in enumerate(sorted(partial_targets.keys())):
                        with cols[idx % len(cols)]:
                            partial_targets[comp] = float(
                                st.number_input(
                                    f"Target p{comp}",
                                    value=float(partial_targets[comp]),
                                    min_value=-10.0,
                                    max_value=10.0,
                                    step=0.01,
                                )
                            )

            st.session_state["partial_targets"] = partial_targets

            # convergence / architecture (across runs)
            targets_items = tuple(sorted((int(k), float(v)) for k, v in partial_targets.items()))
            conv = compute_experiment_convergence(base_dir_str_for_runs, targets_items)
            render_experiment_convergence(conv, partial_targets)

            st.markdown("---")

            # runtime + total mutation statistics (statistics/*_total.txt etc.)
            agg_totals, df_totals = compute_experiment_totals(base_dir_str_for_runs)
            render_experiment_totals(agg_totals, df_totals)


if __name__ == "__main__":
    main()
