"""Standalone (non-Streamlit) CLI to compute the cross-experiment ablation summary table
(success rate, median generations to threshold, mean hidden fields, mean enabled
connections) directly from data/<experiment>/ folders.

Reuses the same parsing/aggregation logic as the Streamlit "Compare" page
(viz.experiment / viz.parsing) so results are consistent with the interactive app,
without needing to drive a browser session.

Usage:
    python ablation_summary_cli.py "data/HRI Packaging Task" "data/HRI Packaging Task No Speciation" ...
    python ablation_summary_cli.py --data-root ../data --all
"""
import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
logging.getLogger("streamlit").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from viz.experiment import (
    _load_experiment_runs_parsed,
    _evaluate_run_targets,
    _aggregate_convergence_metrics,
)


def infer_partial_target_count(parsed_runs, target_fitness: float) -> dict:
    """HRIPackagingTask's success condition (Population::endConditionMet) requires ALL
    partial-fitness components to exceed targetFitness -- there's no fixed component
    count baked into the analysis code, so infer it from the widest partial vector
    actually observed across all parsed runs.
    """
    max_len = 0
    for run in parsed_runs:
        for parts in run["partial_vectors"]:
            max_len = max(max_len, len(parts))
    return {i: target_fitness for i in range(1, max_len + 1)}


def summarize_experiment(data_dir: Path, target_fitness: float = 0.90):
    parsed_runs = _load_experiment_runs_parsed(str(data_dir))
    if not parsed_runs:
        return None

    partial_targets = infer_partial_target_count(parsed_runs, target_fitness)
    all_metrics = [_evaluate_run_targets(p, partial_targets) for p in parsed_runs]
    agg = _aggregate_convergence_metrics(all_metrics)
    agg["experiment"] = data_dir.name
    agg["partial_target_count"] = len(partial_targets)
    return agg


def format_row(agg: dict) -> str:
    success_rate = agg["success_rate"] * 100.0
    failure_rate = 100.0 - success_rate
    median_gens = agg["median_generations_to_threshold"] if agg["successful_runs"] > 0 else None
    median_gens_str = f"{median_gens:.1f}" if median_gens is not None else "N/A"
    return (
        f"| {agg['experiment']} | {agg['total_runs']} | {success_rate:.2f}% | "
        f"{failure_rate:.2f}% | {median_gens_str} | {agg['mean_hidden_fields']:.2f} | "
        f"{agg['mean_enabled_connections']:.2f} |"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", help="One or more data/<experiment> directories")
    ap.add_argument("--data-root", default="../data", help="Root data/ folder (used with --all)")
    ap.add_argument("--all", action="store_true", help="Summarize every subfolder of --data-root")
    ap.add_argument("--target-fitness", type=float, default=0.90)
    ap.add_argument("--out", help="Optional path to write the markdown table to")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    if args.all:
        root = Path(args.data_root)
        dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")]

    if not dirs:
        ap.error("Provide experiment directories, or --data-root with --all")

    lines = [
        "| experiment | runs | success rate | failure rate | median gens to threshold | hidden fields (mean) | enabled connections (mean) |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in dirs:
        agg = summarize_experiment(d, args.target_fitness)
        if agg is None:
            print(f"warning: no parseable runs found in {d}", file=sys.stderr)
            continue
        lines.append(format_row(agg))
        print(f"{d.name}: {agg['total_runs']} runs, {agg['successful_runs']} successful, "
              f"{len(infer_partial_target_count(_load_experiment_runs_parsed(str(d)), args.target_fitness))} partial targets inferred",
              file=sys.stderr)

    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
