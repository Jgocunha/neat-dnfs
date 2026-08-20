from pathlib import Path
import pandas as pd

from .genome import summarize_best_solution_genome
from .stats import compute_fitness_stats, compute_species_stats, compute_topology_stats, display_gen
from .parsing import compute_population_kernel_usage, compute_species_meta, load_best_solution_architecture, load_overview
from .experiment import compute_experiment_convergence, compute_experiment_totals

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

    fitness = compute_fitness_stats(df, target_fitness)
    species = compute_species_stats(df, species_meta)
    topology = compute_topology_stats(df)

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
    elements = load_best_solution_architecture(run_dir_str, fitness.final_gen0)
    if elements is not None:
        df_fields, df_inter = summarize_best_solution_genome(elements)
        fields_md = df_to_markdown_table(df_fields)
        inter_md = df_to_markdown_table(df_inter)
    else:
        fields_md = "_No best-solution JSON found for this generation._"
        inter_md = ""

    # ---------- BUILD MARKDOWN TEXT ----------
    lines = []

    # Header: path
    lines.append(f"# {run_dir}")
    lines.append("")  # keep one blank line

    # ---- Fitness statistics ----
    lines.append("## Fitness statistics")
    lines.append("")
    lines.append(f"Final generation (g = {display_gen(fitness.final_gen0)})")
    lines.append(f"- Best fitness: {fitness.best_final:.4f}")
    lines.append(f"- Target fitness: {target_fitness:.2f}")
    lines.append(f"- Average fitness: {fitness.avg_final:.4f}")
    lines.append("")
    lines.append("Overall")
    lines.append(f"- Max best fitness: {fitness.max_best:.4f} (reached at generation {display_gen(fitness.gen_max_best0)})")
    lines.append(f"- Mean best fitness over run (AUC): {fitness.auc_best:.4f}")
    lines.append(f"- Mean average fitness over run (AUC): {fitness.auc_avg:.4f}")
    lines.append(f"- Longest stagnation period: {fitness.longest_stagnation} generations")
    if fitness.total_cross_gen0 is not None:
        lines.append(
            f"- Target fitness {target_fitness:.3f} first reached at generation {display_gen(fitness.total_cross_gen0)} "
            f"(best fitness ≈ {fitness.best_at_target:.4f})"
        )
    else:
        lines.append(f"- Target fitness {target_fitness:.3f} was not reached.")

    lines.append("")
    lines.append("## Species statistics")
    lines.append("")
    lines.append(f"Final generation (g = {display_gen(species.final_gen0)})")
    lines.append(f"- Species: {species.final_species}")
    lines.append(f"- Active species: {species.final_active}")
    lines.append("")
    lines.append("Across run")
    lines.append(f"- Total distinct species created: {species.total_species}")
    lines.append(f"- Species extinct by final generation: {species.extinct_species}")
    lines.append(f"- Average species per generation: {species.avg_species:.2f}")
    lines.append(f"- Average active species per generation: {species.avg_active:.2f}")
    lines.append(f"- Max active species in a generation: {species.max_active_species} (at g={display_gen(species.gen_max_active0)})")
    lines.append("")
    lines.append("Species lifetime & size")
    lines.append(f"- Average species lifespan: {species.avg_lifespan:.2f} generations")
    lines.append(f"- Longest-lived species: {species.max_life_sid} (lifespan {species.max_lifespan})")
    lines.append(f"- Average max members per species: {species.avg_max_members:.2f}")
    lines.append(f"- Average offspring per species: {species.avg_offspring:.2f}")

    lines.append("")
    lines.append("## Topology statistics")
    lines.append("")
    lines.append(f"Final generation (g = {display_gen(topology.final_gen0)})")
    lines.append(f"- Avg genome size: {topology.genome_size_final:.2f}")
    lines.append(f"- Avg field genes: {topology.field_genes_final:.2f}")
    lines.append(f"- Avg connection genes: {topology.conn_genes_final:.2f}")
    lines.append("")
    lines.append("Growth over run")
    lines.append(f"- Genome size change: {topology.genome_size_delta:+.2f} (≈ {topology.genome_size_per_gen:+.3f}/gen)")
    lines.append(f"- Field genes change: {topology.field_genes_delta:+.2f} (≈ {topology.field_genes_per_gen:+.3f}/gen)")
    lines.append(f"- Connection genes change: {topology.conn_genes_delta:+.2f} (≈ {topology.conn_genes_per_gen:+.3f}/gen)")
    lines.append("")
    lines.append("Ratios")
    lines.append(f"- Avg connections per field at final gen: {topology.avg_conn_per_field_final:.2f}")
    lines.append("")
    lines.append("Population-level kernel usage")
    lines.append(f"- Field kernels: {field_kernel_line}")
    lines.append(f"- Interaction kernels: {inter_kernel_line}")

    lines.append("")
    lines.append(
        f"Genome representation of the highest-performing solution for generation "
        f"{display_gen(fitness.final_gen0)} with f = {fitness.best_final:.4f}"
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
