import math
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

from .cache import _disk_cache_read_json, _disk_cache_write_json, _fingerprint_dir, _run_cache_dir
from .parsing import generations_meeting_targets

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


@st.cache_data
def compute_partial_fitness_best_only(run_dir_str: str, generations: tuple) -> pd.DataFrame:
    """best_p* for every generation, read directly from per_generation_overview.txt's embedded
    'part.: (...)' field -- no statistics/ scan at all. Thin wrapper around _parse_run_overview
    (already used by the experiment-aggregation path), reshaped into the same best_p*-columned
    DataFrame shape viz.parsing.compute_partial_fitness produces, so it's a drop-in for any
    caller that only needs best_p* -- avg_p* genuinely requires the full statistics/ scan and
    has no fast-path equivalent (there is no cheaper source for a population-wide average).
    """
    run_dir = Path(run_dir_str)
    overview_path = run_dir / "per_generation_overview.txt"
    if not overview_path.exists():
        return pd.DataFrame()

    parsed = _parse_run_overview(overview_path)
    if parsed is None:
        return pd.DataFrame()

    gen_set = set(generations) if generations else None
    rows = []
    for g0, parts in zip(parsed["generations"], parsed["partial_vectors"]):
        if gen_set is not None and g0 not in gen_set:
            continue
        rec = {"generation": g0}
        for i, v in enumerate(parts, start=1):
            rec[f"best_p{i}"] = v
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values("generation", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


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
    failed_components = []  # structured sibling of failed_partials_str, for aggregation
    for p_num, thr in sorted(partial_targets.items()):
        # Partial fitness targets are specified as p1, p2, ... (1-based indexing).
        if p_num <= 0:
            continue
        idx = p_num - 1  # convert to 0-based index into the partial vector
        if idx >= len(best_solution_partials):
            failed_partials.append(f"p{p_num}=NA<{thr:.3f}")
            failed_components.append(p_num)
            continue
        val = best_solution_partials[idx]
        # Treat missing/NaN values as failures.
        if val is None or (isinstance(val, float) and math.isnan(val)) or float(val) < float(thr):
            failed_components.append(p_num)
            try:
                failed_partials.append(f"p{p_num}={float(val):.3f}<{thr:.3f}")
            except Exception:
                failed_partials.append(f"p{p_num}=NA<{thr:.3f}")

    failed_partials_str = ", ".join(failed_partials) if failed_partials else "(none)"

    out = dict(parsed)
    out["success"] = success
    out["generation_to_threshold"] = generation_to_threshold
    out["best_solution_failed_partials"] = failed_partials_str
    out["failed_partial_components"] = failed_components
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
