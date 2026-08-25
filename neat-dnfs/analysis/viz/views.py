import math
from pathlib import Path
import pandas as pd
import streamlit as st

from .genome import build_topology_graph, compute_kernel_usage_stats, summarize_best_solution_genome
from .stats import compute_fitness_stats, compute_partial_component_failure_rates, compute_species_stats, compute_topology_frequency, compute_topology_stats, display_gen, find_invariant_violations, mann_whitney_u, spearman_correlation, topology_distance
from .plots import chart_all_runs_overlay, chart_architecture_complexity_scatter, chart_best_mutation_timeline, chart_convergence_generations_histogram, chart_cross_experiment_boxplot, chart_first_crossing_per_component, chart_genome_topology_curves, chart_innovation_growth, chart_kernel_usage_time, chart_lineage_fitness, chart_mutation_categories, chart_mutation_effectiveness, chart_mutations_per_generation, chart_partial_component_failure_rates, chart_partial_component_heatmap, chart_partial_fitness_grid, chart_population_distribution, chart_run_duration_histogram, chart_seconds_per_generation_histogram, chart_species_champion_trajectory, chart_species_counts, chart_species_lifespans, chart_species_membership, chart_topology_frequency_heatmap, chart_topology_trajectory, chart_total_fitness, plot_topology_graph, show_fig
from .parsing import _sample_evenly, compute_mutation_events, compute_partial_fitness, compute_per_generation_best_mutation, compute_population_distributions, compute_population_kernel_usage, compute_population_parameter_distributions, compute_species_meta, compute_target_crossing_mutations, compute_topology_trajectory, find_experiment_dirs, first_crossing_per_component, format_ram_bytes, generations_all_partial_meet_targets, get_best_solution_id, get_species_for_generation, list_champion_generations, load_best_solution_architecture, load_champion_architecture, load_run_metadata, parse_vcpkg_package_list, species_champion_fitness_trajectory, trace_lineage
from .experiment import _load_experiment_runs_parsed, compute_experiment_convergence, compute_experiment_totals, compute_partial_fitness_best_only
from .solution_record import parse_solution_blob

def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v into [lo, hi]. Used to keep a target value carried over in session
    state (from a previous run/experiment selection) within a widget's current
    min/max, which are derived from the newly selected data and can be narrower."""
    return float(min(max(v, lo), hi))


def render_fitness_stats(
    df: pd.DataFrame,
    target_fitness: float,
    partial_threshold_generation: int | None = None,
):
    s = compute_fitness_stats(df, target_fitness)

    st.markdown("#### Statistics")
    st.caption(f"Final generation: g = {display_gen(s.final_gen0)}")

    row1 = st.columns(3)
    with row1[0]:
        st.metric(
            "Best fitness",
            f"{s.best_final:.4f}",
            delta=f"{s.best_final - target_fitness:+.4f}",
            help=f"vs. target ({target_fitness:.3f})",
            border=True,
        )
    with row1[1]:
        st.metric("Average fitness", f"{s.avg_final:.4f}", border=True)
    with row1[2]:
        st.metric(
            "Max best fitness",
            f"{s.max_best:.4f}",
            help=f"Reached at generation {display_gen(s.gen_max_best0)}",
            border=True,
        )

    row2 = st.columns(3)
    with row2[0]:
        st.metric("Mean best fitness (AUC)", f"{s.auc_best:.4f}", border=True)
    with row2[1]:
        st.metric("Mean avg fitness (AUC)", f"{s.auc_avg:.4f}", border=True)
    with row2[2]:
        st.metric("Longest stagnation", f"{s.longest_stagnation} gen", border=True)

    if partial_threshold_generation is not None:
        st.success(f"✅ All partial fitness targets met simultaneously at generation **{partial_threshold_generation}**.")
    else:
        st.warning("❌ All partial fitness targets were **never** met simultaneously in this run.")

    if s.total_cross_gen0 is not None:
        st.caption(f"Reference: total best fitness first crossed the chosen overall target at generation {display_gen(s.total_cross_gen0)}.")


def render_species_stats(df: pd.DataFrame, species_meta: dict):
    s = compute_species_stats(df, species_meta)

    st.markdown("#### Species statistics")
    st.caption(f"Final generation: g = {display_gen(s.final_gen0)}")

    row1 = st.columns(4)
    with row1[0]:
        st.metric("Species (final)", s.final_species, border=True)
    with row1[1]:
        st.metric("Active species (final)", s.final_active, border=True)
    with row1[2]:
        st.metric("Total species ever created", s.total_species, border=True)
    with row1[3]:
        st.metric("Extinct by final gen", s.extinct_species, border=True)

    row2 = st.columns(3)
    with row2[0]:
        st.metric("Avg species / gen", f"{s.avg_species:.2f}", border=True)
    with row2[1]:
        st.metric("Avg active species / gen", f"{s.avg_active:.2f}", border=True)
    with row2[2]:
        st.metric(
            "Max active species",
            s.max_active_species,
            help=f"At generation {display_gen(s.gen_max_active0)}",
            border=True,
        )

    row3 = st.columns(4)
    with row3[0]:
        st.metric(
            "Longest-lived species",
            s.max_life_sid,
            help=f"Lifespan: {s.max_lifespan} generations",
            border=True,
        )
    with row3[1]:
        st.metric("Avg species lifespan", f"{s.avg_lifespan:.2f} gen", border=True)
    with row3[2]:
        st.metric("Avg max members / species", f"{s.avg_max_members:.2f}", border=True)
    with row3[3]:
        st.metric("Avg offspring / species", f"{s.avg_offspring:.2f}", border=True)


def render_topology_stats(df: pd.DataFrame):
    s = compute_topology_stats(df)

    st.markdown("#### Topology statistics")
    st.caption(f"Final generation: g = {display_gen(s.final_gen0)}")

    row1 = st.columns(3)
    with row1[0]:
        st.metric(
            "Avg genome size",
            f"{s.genome_size_final:.2f}",
            delta=f"{s.genome_size_delta:+.2f}",
            help=f"≈ {s.genome_size_per_gen:+.3f} per generation, over the whole run",
            border=True,
        )
    with row1[1]:
        st.metric(
            "Avg field genes",
            f"{s.field_genes_final:.2f}",
            delta=f"{s.field_genes_delta:+.2f}",
            help=f"≈ {s.field_genes_per_gen:+.3f} per generation, over the whole run",
            border=True,
        )
    with row1[2]:
        st.metric(
            "Avg connection genes",
            f"{s.conn_genes_final:.2f}",
            delta=f"{s.conn_genes_delta:+.2f}",
            help=f"≈ {s.conn_genes_per_gen:+.3f} per generation, over the whole run",
            border=True,
        )

    st.metric("Connections per field (final gen)", f"{s.avg_conn_per_field_final:.2f}", border=True)


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

    mutation_column_config = {
        "mutation_raw": st.column_config.TextColumn("mutation", help="The mutation description as logged"),
        "category": st.column_config.TextColumn("category", help="High-level mutation taxonomy"),
        "occurrences": st.column_config.NumberColumn(
            "count", help="How many times this mutation was applied", format="%d"
        ),
        "mean_fitness": st.column_config.NumberColumn(
            "mean fitness", help="Mean fitness of individuals carrying this mutation", format="%.4f"
        ),
        "delta_vs_global": st.column_config.NumberColumn(
            "Δ vs global", help="Mean fitness minus the overall mean fitness of the run", format="%.4f"
        ),
    }

    # Most frequent
    st.markdown("#### Most frequent mutations")
    st.caption(
        "How often each mutation appears, and how the average fitness of individuals "
        "with that mutation compares to the overall mean fitness of the run."
    )
    top_freq = summary.sort_values("occurrences", ascending=False).head(10)
    st.dataframe(top_freq, width="stretch", column_config=mutation_column_config, hide_index=True)

    # Most beneficial (positive delta)
    st.markdown("#### Mutations associated with higher fitness")
    st.caption(
        "Mutations whose carriers tend to have higher fitness than the global mean. "
        "Beware of very rare mutations (low counts) — they can look good just by chance."
    )
    top_good = summary.sort_values("delta_vs_global", ascending=False).head(10)
    st.dataframe(top_good, width="stretch", column_config=mutation_column_config, hide_index=True)


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

    # ---- a couple of simple histograms ----
    st.markdown("#### Distributions across runs")
    plot_cols = st.columns(2)

    # duration
    if "duration_hours" in df.columns:
        with plot_cols[0]:
            st.altair_chart(chart_run_duration_histogram(df["duration_hours"]), width="stretch")
            st.caption(
                "How long each run took from start to end. This shows whether some runs "
                "are much slower or faster than others."
            )

    # seconds per generation
    if "seconds_per_generation" in df.columns:
        with plot_cols[1]:
            st.altair_chart(chart_seconds_per_generation_histogram(df["seconds_per_generation"]), width="stretch")
            st.caption(
                "Average wall-clock time needed to compute one generation in each run. "
                "Useful for spotting performance regressions."
            )


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

    top_row = st.columns(2)
    with top_row[0]:
        st.metric("Runs analysed", total, border=True)
    with top_row[1]:
        st.metric(
            "Met all targets simultaneously",
            succ,
            delta=f"{rate:.1f}% success rate",
            delta_color="off",
            border=True,
        )
    st.caption(f"Targets: {targets_str}")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("##### Generations to success (successful runs)")
        sub = st.columns(3)
        with sub[0]:
            st.metric("Mean", f"{conv['mean_generations_to_threshold']:.2f}", border=True)
        with sub[1]:
            st.metric("Median", f"{conv['median_generations_to_threshold']:.2f}", border=True)
        with sub[2]:
            st.metric("Std", f"{conv['std_generations_to_threshold']:.2f}", border=True)

    with cols[1]:
        st.markdown("##### Convergence speed")
        sub = st.columns(2)
        with sub[0]:
            st.metric("Mean rate (fitness gain/gen)", f"{conv['mean_convergence_rate']:.4f}", border=True)
        with sub[1]:
            st.metric("Mean improvement/gen", f"{conv['mean_improvement_per_gen']:.4f}", border=True)

    with cols[2]:
        st.markdown("##### Architecture (successful solutions)")
        sub = st.columns(2)
        with sub[0]:
            st.metric(
                "Hidden fields (mean)",
                f"{conv['mean_hidden_fields']:.2f}",
                help=f"± {conv['std_hidden_fields']:.2f} std",
                border=True,
            )
        with sub[1]:
            st.metric(
                "Enabled conns (mean)",
                f"{conv['mean_enabled_connections']:.2f}",
                help=f"± {conv['std_enabled_connections']:.2f} std",
                border=True,
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
            st.altair_chart(chart_convergence_generations_histogram(gens), width="stretch")
            st.caption(
                "Each bar shows how many runs first met all partial targets "
                "in a given generation range. Left = faster convergence."
            )

        # --- Architecture complexity scatter ---
        with col2:
            hidden = [m["hidden_fields_count"] for m in successful]
            conns = [m["enabled_connections_count"] for m in successful]
            labels = [m["run_dir"] for m in successful]

            st.altair_chart(chart_architecture_complexity_scatter(hidden, conns, labels), width="stretch")
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
        notable_column_config = {
            "label": st.column_config.TextColumn("metric"),
            "run": st.column_config.TextColumn("run"),
            "value": st.column_config.NumberColumn("value", help="Units depend on the metric -- see the metric column"),
        }
        st.dataframe(pd.DataFrame(rows), width="stretch", column_config=notable_column_config, hide_index=True)

    # --- Best performer in each run ---
    if conv.get("all_run_metrics"):
        st.markdown("#### Success per run")

        rows = []
        for m in conv["all_run_metrics"]:
            rows.append(
                {
                    "run_id": m.get("run_dir"),
                    "best_fitness": m.get("best_solution_fitness"),
                    "failed_partials": m.get("best_solution_failed_partials"),
                    "success": bool(m.get("success")),
                    "gens_to_threshold": m.get("generation_to_threshold"),
                    "species_total": m.get("final_species_total"),
                    "species_active": m.get("final_species_active"),
                    "duration": m.get("duration_human"),
                    "hidden_fields": m.get("hidden_fields_count"),
                    "enabled_connections": m.get("enabled_connections_count"),
                }
            )

        df_runs = pd.DataFrame(rows)

        # Sort: successful first, then by earliest success generation, then by best fitness
        df_runs = df_runs.sort_values(
            by=["success", "gens_to_threshold", "best_fitness"],
            ascending=[False, True, False],
            na_position="last",
        ).reset_index(drop=True)

        success_column_config = {
            "run_id": st.column_config.TextColumn("run", help="Run directory name"),
            "best_fitness": st.column_config.NumberColumn("best fitness", format="%.4f"),
            "failed_partials": st.column_config.TextColumn(
                "failed partials (last gen)",
                help="Which partial targets the best-total-fitness solution failed, if any",
            ),
            "success": st.column_config.CheckboxColumn(
                "above threshold?",
                help="At least one generation had all partial fitness components at/above target",
            ),
            "gens_to_threshold": st.column_config.NumberColumn(
                "gens to threshold",
                help="First generation (1-based) where all partial targets were met simultaneously",
                format="%d",
            ),
            "species_total": st.column_config.NumberColumn("species (total)", format="%d"),
            "species_active": st.column_config.NumberColumn("species (active)", format="%d"),
            "duration": st.column_config.TextColumn("duration"),
            "hidden_fields": st.column_config.NumberColumn("hidden fields", format="%d"),
            "enabled_connections": st.column_config.NumberColumn("enabled connections", format="%d"),
        }

        st.dataframe(df_runs, width="stretch", column_config=success_column_config, hide_index=True)
        st.caption(
            "Main diagnosis per run. A run is marked successful when there exists at least one generation "
            "where all partial fitness components are ≥ their targets in the same generation. "
            "`gens to threshold` is the first such generation (1-based)."
        )


@st.fragment
def render_fitness_view(df: pd.DataFrame, gens_tuple: tuple, selected_run_path: str):
    st.divider()

    min_fit = float(min(df["avg_fitness"].min(), df["best_fitness"].min(), 0.0))
    max_fit = float(max(df["avg_fitness"].max(), df["best_fitness"].max(), 1.0))
    slider_min = min_fit
    slider_max = max_fit if max_fit > min_fit else min_fit + 1.0

    # The slider and number_input below both edit target_fitness. Each owns its own widget
    # key; on_change callbacks keep the two in sync in both directions, and neither widget is
    # given a `value=` after its key is first seeded, so Streamlit's session-state-owns-the-
    # value contract stays consistent (mixing `value=` with externally-set session_state for a
    # keyed widget is what caused the old silent last-write-wins behavior here).
    if "target_fitness_slider" not in st.session_state:
        st.session_state["target_fitness_slider"] = float(st.session_state["target_fitness"])
    if "target_fitness_input" not in st.session_state:
        st.session_state["target_fitness_input"] = float(st.session_state["target_fitness"])
    st.session_state["target_fitness_slider"] = _clamp(st.session_state["target_fitness_slider"], slider_min, slider_max)
    st.session_state["target_fitness_input"] = _clamp(st.session_state["target_fitness_input"], slider_min, slider_max)

    def _sync_target_from_slider():
        st.session_state["target_fitness_input"] = st.session_state["target_fitness_slider"]

    def _sync_target_from_input():
        st.session_state["target_fitness_slider"] = st.session_state["target_fitness_input"]

    c1, c2 = st.columns([4, 1])
    with c1:
        st.slider(
            "Overall target fitness (for main plot)",
            min_value=slider_min,
            max_value=slider_max,
            step=(slider_max - slider_min) / 100.0 if slider_max > slider_min else 0.01,
            key="target_fitness_slider",
            on_change=_sync_target_from_slider,
        )
    with c2:
        st.number_input(
            "Target (manual)",
            min_value=slider_min,
            max_value=slider_max,
            key="target_fitness_input",
            on_change=_sync_target_from_input,
        )

    target = float(st.session_state["target_fitness_input"])
    st.session_state["target_fitness"] = target

    # ---------- Partial targets + "success" generation ----------
    # best_p* is already in per_generation_overview.txt (read here with no statistics/ scan);
    # avg_p* genuinely requires scanning every statistics/generation_X.txt, so that full scan
    # is opt-in -- everything that only needs best_p* (this banner, the target-crossing
    # generation list) uses the fast path unconditionally.
    partial_df = compute_partial_fitness_best_only(selected_run_path, gens_tuple)
    include_avg = st.checkbox(
        "Include population-average line in partial-fitness grid",
        value=False,
        help="Requires a full scan of statistics/generation_X.txt (can be slow for large "
        "runs). Without this, the grid below shows only each component's best-individual line.",
    )
    if include_avg:
        with st.spinner("Scanning statistics/generation_X.txt for population-average fitness..."):
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
                        if best_col not in partial_df.columns:
                            comp += 1
                            continue

                        if comp not in partial_targets:
                            partial_targets[comp] = float(st.session_state["target_fitness"])

                        bound_vals = [partial_df[best_col].min(), partial_df[best_col].max()]
                        if avg_col in partial_df.columns:
                            bound_vals += [partial_df[avg_col].min(), partial_df[avg_col].max()]
                        col_min = float(min(bound_vals + [0.0]))
                        col_max = float(max(bound_vals + [1.0]))
                        partial_targets[comp] = _clamp(partial_targets[comp], col_min, col_max)

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
                max_shown = 15
                shown = ", ".join(str(g) for g in gens_ok[:max_shown])
                more = f" (+{len(gens_ok) - max_shown} more)" if len(gens_ok) > max_shown else ""
                st.info(
                    f"**{len(gens_ok)}** generation(s) where all partial best fitnesses are ≥ their "
                    f"targets (first at generation **{first_ok}**): {shown}{more}."
                )
            else:
                st.info(
                    "In this run, **no generation** reached all partial best fitness targets simultaneously."
                )

    # ---------- Plots + stats (threshold based on partials) ----------
    st.markdown("### Total fitness")
    chart_total = chart_total_fitness(df, st.session_state["target_fitness"], success_generations=gens_ok, success_label="All partial targets met")
    st.altair_chart(chart_total, width="stretch")

    st.divider()
    render_fitness_stats(
        df,
        st.session_state["target_fitness"],
        partial_threshold_generation=first_ok,
    )

    st.divider()
    chart_partial_fitness_grid(partial_df, st.session_state["partial_targets"])

    if partial_targets and partial_df is not None and not partial_df.empty:
        st.divider()
        render_partial_component_bottlenecks(partial_df, partial_targets)


def render_partial_component_bottlenecks(partial_df: pd.DataFrame, partial_targets: dict):
    st.markdown("#### Partial-component bottleneck analysis")
    st.caption(
        "Which of the partial-fitness components gates success in this run, shown flat and "
        "independently weighted across all logged components -- any higher-level grouping and "
        "internal weights are computed in C++ but never written to disk, so they can't be "
        "reconstructed here."
    )

    heat_col, cross_col = st.columns(2)
    with heat_col:
        st.altair_chart(chart_partial_component_heatmap(partial_df, partial_targets), width="stretch")

    generations = partial_df["generation"].tolist()
    partial_vectors = _partial_vectors_from_df(partial_df, partial_targets)
    crossing = first_crossing_per_component(generations, partial_vectors, partial_targets)
    with cross_col:
        st.altair_chart(chart_first_crossing_per_component(crossing), width="stretch")


def _partial_vectors_from_df(partial_df: pd.DataFrame, partial_targets: dict) -> list:
    max_comp = max(int(c) for c in partial_targets)
    return [
        [
            float(row[f"best_p{i}"]) if f"best_p{i}" in partial_df.columns else float("nan")
            for i in range(1, max_comp + 1)
        ]
        for _, row in partial_df.iterrows()
    ]


def _render_solution_record(raw_blob: str):
    """Render a species rep./champ. solution blob as structured tables instead of a raw-text
    dump, via the shared solution_record parser. Falls back to the raw text (in a collapsed
    expander) if the blob is empty or doesn't match the expected shape, so nothing is lost."""
    if not raw_blob:
        st.caption("No record available.")
        return

    record = parse_solution_blob(raw_blob)
    if record is None:
        st.info("Could not parse this solution record; showing raw text instead.")
        st.code(raw_blob, language="text")
        return

    st.markdown(
        f"fitness: **{record['fitness']:.4f}**  \n"
        f"adjusted fitness: **{record['adjusted_fitness']:.4f}**  \n"
        f"age: **{record['age']}**, parents: **{record['parent_ids']}**  \n"
        f"genome: **{record['num_field_genes']}** field genes, "
        f"**{record['num_connection_genes']}** connection genes"
    )

    if record["field_genes"]:
        st.markdown("Field genes")
        field_genes_column_config = {
            "id": st.column_config.NumberColumn("id", format="%d"),
            "type": st.column_config.TextColumn("type"),
        }
        st.dataframe(
            pd.DataFrame(record["field_genes"]), width="stretch", column_config=field_genes_column_config, hide_index=True
        )
    if record["connection_genes"]:
        st.markdown("Connection genes")
        connection_genes_column_config = {
            "src": st.column_config.NumberColumn("source", format="%d"),
            "tgt": st.column_config.NumberColumn("target", format="%d"),
            "innov": st.column_config.NumberColumn("innovation #", format="%d"),
            "enabled": st.column_config.CheckboxColumn("enabled"),
        }
        st.dataframe(
            pd.DataFrame(record["connection_genes"]),
            width="stretch",
            column_config=connection_genes_column_config,
            hide_index=True,
        )

    with st.expander("Raw text", expanded=False):
        st.code(raw_blob, language="text")


def _render_kernel_usage_table(counts: dict, perc: dict, empty_message: str):
    """Small table for a {kernel_kind: count} / {kernel_kind: percentage} pair -- replaces a
    comma-joined "kind: count (pct%)" string with a real, sortable table."""
    if not counts:
        st.caption(empty_message)
        return
    rows = [{"kernel": k, "count": counts[k], "share": perc.get(k, 0.0) / 100.0} for k in sorted(counts.keys())]
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "kernel": st.column_config.TextColumn("kernel kind"),
            "count": st.column_config.NumberColumn("count", format="%d"),
            "share": st.column_config.ProgressColumn("share", min_value=0.0, max_value=1.0, format="percent"),
        },
    )


@st.fragment
def render_species_genome_inspector(df: pd.DataFrame, selected_run_path: str):
    st.markdown("#### Species genome (per generation)")

    min_gen0 = int(df["generation"].min())
    max_gen0 = int(df["generation"].max())
    gen_sel_display = st.slider(
        "Generation to inspect",
        min_value=display_gen(min_gen0),
        max_value=display_gen(max_gen0),
        value=display_gen(max_gen0),
    )
    gen_sel = gen_sel_display - 1

    species_list = get_species_for_generation(selected_run_path, gen_sel)

    if not species_list:
        st.info(f"No species file found or parsable for generation {gen_sel_display}.")
    else:
        active_species = [
            s for s in species_list if (not s["extinct"]) and s["members"] > 0
        ]

        st.markdown(
            f"Generation **{gen_sel_display}** — active species: **{len(active_species)}** "
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
                    rep_col, champ_col = st.columns(2)
                    with rep_col:
                        st.markdown("**Representative solution**")
                        _render_solution_record(s["rep_raw"])
                    with champ_col:
                        st.markdown("**Champion solution**")
                        _render_solution_record(s["champ_raw"])


def _render_sampling_controls(
    key_prefix: str, num_generations: int, include_max_solutions: bool = True, help_text: str | None = None
) -> tuple[int, int | None]:
    """Shared "sample every Nth generation" (+ optional "max solutions per generation") control
    pair, used by every view that subsamples genome files across a large population.
    `key_prefix` keeps each call site's widget state independent."""
    if include_max_solutions:
        col1, col2 = st.columns(2)
        with col1:
            gen_step = st.number_input(
                "Sample every Nth generation",
                min_value=1,
                max_value=max(1, num_generations),
                value=min(10, max(1, num_generations)),
                key=f"{key_prefix}_gen_step",
                help=help_text,
            )
        with col2:
            max_solutions_per_gen = st.number_input(
                "Max solutions sampled per generation",
                min_value=1,
                max_value=1000,
                value=100,
                key=f"{key_prefix}_max_solutions",
            )
        return int(gen_step), int(max_solutions_per_gen)

    gen_step = st.number_input(
        "Sample every Nth generation",
        min_value=1,
        max_value=max(1, num_generations),
        value=min(10, max(1, num_generations)),
        key=f"{key_prefix}_gen_step",
        help=help_text,
    )
    return int(gen_step), None


@st.fragment
def render_population_kernel_usage(df: pd.DataFrame, selected_run_path: str):
    gens_tuple = tuple(df["generation"].tolist())

    st.markdown("#### Population-level kernel usage")
    gen_step, max_solutions_per_gen = _render_sampling_controls(
        "kernel_usage",
        len(gens_tuple),
        help_text="A run can have 1000 solutions x 200 generations of genome "
        "files; reading all of them is slow, so this samples generations "
        "and solutions instead of scanning everything.",
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
            chart_fu = chart_kernel_usage_time(df_kernel_usage, kind="field")
            if chart_fu is not None:
                st.altair_chart(chart_fu, width="stretch")
        with col_b:
            chart_iu = chart_kernel_usage_time(df_kernel_usage, kind="inter")
            if chart_iu is not None:
                st.altair_chart(chart_iu, width="stretch")

        # Overall summary across entire run
        st.caption(
            f"Sampled every {int(gen_step)} generation(s), up to "
            f"{int(max_solutions_per_gen)} solutions per sampled generation "
            "-- not an exhaustive count over the full population."
        )
        usage_col_a, usage_col_b = st.columns(2)
        with usage_col_a:
            st.markdown("**Field kernels**")
            _render_kernel_usage_table(field_overall_counts, field_overall_perc, "none")
        with usage_col_b:
            st.markdown("**Interaction kernels**")
            _render_kernel_usage_table(inter_overall_counts, inter_overall_perc, "none")

    else:
        st.info(
            "No population-level kernel usage data found. "
            "This usually means there are no genome files under "
            "`solutions/gen X/<solution>/*.dnf` for this run."
        )


@st.fragment
def render_parameter_distributions(df: pd.DataFrame, selected_run_path: str):
    st.markdown("#### Parameter distributions across the population")
    st.caption(
        "Every numeric field/kernel parameter (tau, resting level, kernel amplitudes and "
        "widths) discarded elsewhere in the app -- only kernel *kind* is used for the kernel "
        "-usage view above. Shares the same generation/solution sampling controls."
    )

    gens_tuple = tuple(df["generation"].tolist())
    gen_step, max_solutions_per_gen = _render_sampling_controls("param_dist", len(gens_tuple))

    param_df = compute_population_parameter_distributions(
        selected_run_path, gens_tuple, gen_step, max_solutions_per_gen
    )
    if param_df.empty:
        st.info(
            "No parameter data found. This usually means there are no genome files under "
            "`solutions/gen X/<solution>/*.dnf` for this run."
        )
        return

    available_params = sorted(param_df["parameter"].unique().tolist())
    param_name = st.selectbox("Parameter", available_params, key="param_dist_param_name")

    filtered = param_df[param_df["parameter"] == param_name].rename(columns={"value": param_name})
    sampled_generations = sorted(filtered["generation"].unique().tolist())
    chart = chart_population_distribution(filtered, param_name, sampled_generations)
    st.altair_chart(chart, width="stretch")
    st.caption(f"{len(filtered)} values sampled across {len(sampled_generations)} generations.")


@st.fragment
def render_best_solution_genome(df: pd.DataFrame, selected_run_path: str):
    min_gen0 = int(df["generation"].min())
    max_gen0 = int(df["generation"].max())
    gen_sel_display = st.slider(
        "Generation to inspect (best solution)",
        min_value=display_gen(min_gen0),
        max_value=display_gen(max_gen0),
        value=display_gen(max_gen0),
    )
    gen_sel = gen_sel_display - 1

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
            f"for generation {gen_sel_display} with f = {best_f:.4f}"
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

        usage_col_a, usage_col_b = st.columns(2)
        with usage_col_a:
            st.markdown("**Field kernels**")
            _render_kernel_usage_table(field_counts, field_perc, "No kernels associated with fields.")
        with usage_col_b:
            st.markdown("**Interaction kernels (field–field)**")
            _render_kernel_usage_table(inter_counts, inter_perc, "None in this genome.")

        # Optional: graph view
        with st.expander("Graph view of field interactions", expanded=False):
            g, pos, field_nodes, kernel_nodes = build_topology_graph(elements)
            fig_top = plot_topology_graph(g, pos, field_nodes, kernel_nodes)
            show_fig(fig_top)


@st.fragment
def render_species_view(df: pd.DataFrame, gens_tuple: tuple, selected_run_path: str):
    st.divider()

    top_left, top_right = st.columns(2)
    with top_left:
        st.altair_chart(chart_species_counts(df), width="stretch")
    with top_right:
        st.altair_chart(chart_innovation_growth(df), width="stretch")

    st.divider()
    species_meta = compute_species_meta(selected_run_path, gens_tuple)
    render_species_stats(df, species_meta)

    st.divider()
    render_species_genome_inspector(df, selected_run_path)

    st.divider()
    render_species_dynamics(species_meta, gens_tuple, selected_run_path)


def render_species_dynamics(species_meta: dict, gens_tuple: tuple, selected_run_path: str):
    st.markdown("#### Species dynamics over time")

    if not species_meta:
        st.info("No species data found for this run.")
        return

    top_left, top_right = st.columns(2)
    with top_left:
        st.altair_chart(chart_species_membership(species_meta), width="stretch")
    with top_right:
        st.altair_chart(chart_species_lifespans(species_meta), width="stretch")

    species_ids = sorted(species_meta.keys())
    default_id = max(species_meta.keys(), key=lambda sid: species_meta[sid]["max_members"])
    selected_sid = st.selectbox(
        "Species to trace champion fitness for",
        species_ids,
        index=species_ids.index(default_id),
        key="species_champion_select",
    )
    traj_df = species_champion_fitness_trajectory(selected_run_path, gens_tuple, selected_sid)
    if traj_df.empty:
        st.info(f"No champion fitness records found for species {selected_sid}.")
    else:
        st.altair_chart(chart_species_champion_trajectory(traj_df, selected_sid), width="stretch")


@st.fragment
def render_topology_view(df: pd.DataFrame, selected_run_path: str):
    tab_structure, tab_population, tab_genealogy, tab_archive = st.tabs(
        ["Structure", "Population", "Genealogy", "Archive"]
    )

    with tab_structure:
        # Top: genome topology curves + stats
        top_left, top_right = st.columns(2)
        with top_left:
            st.altair_chart(chart_genome_topology_curves(df), width="stretch")
        with top_right:
            render_topology_stats(df)

        st.divider()
        traj_df = compute_topology_trajectory(selected_run_path)
        if not traj_df.empty:
            st.altair_chart(chart_topology_trajectory(traj_df), width="stretch")
            first_change_idx = traj_df["hidden_fields"].diff().fillna(0).ne(0).idxmax() if traj_df["hidden_fields"].diff().fillna(0).ne(0).any() else None
            final_hidden = int(traj_df["hidden_fields"].iloc[-1])
            if first_change_idx is not None:
                first_change_gen = display_gen(int(traj_df.loc[first_change_idx, "generation"]))
                held_from = traj_df[traj_df["hidden_fields"] == final_hidden]["generation"]
                stable_since = display_gen(int(held_from.iloc[0])) if not held_from.empty else None
                st.caption(
                    f"First structural change (hidden-field count) at generation {first_change_gen}. "
                    f"Final hidden-field count ({final_hidden}) first reached at generation {stable_since}."
                )
            else:
                st.caption(f"Hidden-field count held constant at {final_hidden} across all recorded generations.")

            with st.expander("Check a topology invariant across generations", expanded=False):
                st.caption(
                    "General-purpose invariant check -- e.g. for the ablation study's A1/A2 "
                    "conditions, which must hold an exact field/connection count on every "
                    "generation, not just the final one."
                )
                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    target_hidden = st.number_input("Target hidden fields", min_value=0, value=int(final_hidden), key="inv_target_hidden")
                with ic2:
                    target_conn = st.number_input(
                        "Target enabled connections", min_value=0, value=int(traj_df["enabled_connections"].iloc[-1]), key="inv_target_conn"
                    )
                with ic3:
                    tolerance = st.number_input("Tolerance", min_value=0, value=0, key="inv_tolerance")

                violations = find_invariant_violations(traj_df, int(target_hidden), int(target_conn), int(tolerance))
                if not violations:
                    st.success(
                        f"Invariant held on all {len(traj_df)} recorded generations: "
                        f"{int(target_hidden)} hidden field(s), {int(target_conn)} enabled connection(s) "
                        f"(± {int(tolerance)})."
                    )
                else:
                    shown = [display_gen(g) for g in violations[:20]]
                    more = f" (+{len(violations) - 20} more)" if len(violations) > 20 else ""
                    st.error(
                        f"Invariant violated at {len(violations)} of {len(traj_df)} generation(s): "
                        f"{shown}{more}."
                    )

    with tab_population:
        render_population_kernel_usage(df, selected_run_path)
        st.divider()
        render_parameter_distributions(df, selected_run_path)
        st.divider()
        render_population_distributions(df, selected_run_path)

    with tab_genealogy:
        render_best_solution_genome(df, selected_run_path)
        st.divider()
        render_lineage_tracer(df, selected_run_path)

    with tab_archive:
        render_champion_archive(df, selected_run_path)


@st.fragment
def render_champion_archive(df: pd.DataFrame, selected_run_path: str):
    st.markdown("#### Champion archive (per-species lineage)")
    st.caption(
        "Browse champions/prev_generations/ -- one genome per species per generation, "
        "untouched by any other view. Pick a species and step through its champion's genome "
        "over time."
    )

    gens_tuple = tuple(df["generation"].tolist())
    species_meta = compute_species_meta(selected_run_path, gens_tuple)
    if not species_meta:
        st.info("No species data found for this run.")
        return

    species_ids = sorted(species_meta.keys())
    default_id = max(species_meta.keys(), key=lambda sid: species_meta[sid]["max_members"])
    selected_sid = st.selectbox(
        "Species",
        species_ids,
        index=species_ids.index(default_id),
        key="champion_archive_species_select",
    )

    with st.spinner("Scanning champions/prev_generations/ (cached after the first run)..."):
        champion_gens = list_champion_generations(selected_run_path, selected_sid)
    if not champion_gens:
        st.info(f"No champion archive entries found for species {selected_sid}.")
        return

    gen0_options = [g for g, _ in champion_gens]
    gen_sel_display = st.select_slider(
        "Generation",
        options=[display_gen(g) for g in gen0_options],
        value=display_gen(gen0_options[-1]),
        key="champion_archive_gen_slider",
    )
    gen_sel0 = gen_sel_display - 1

    champion_dir = dict(champion_gens)[gen_sel0]
    elements = load_champion_architecture(str(champion_dir))
    if elements is None:
        st.info("Could not load this champion's genome file.")
        return

    st.markdown(f"Species **{selected_sid}** champion at generation **{gen_sel_display}**")
    df_fields, df_inter = summarize_best_solution_genome(elements)
    if df_fields is not None and not df_fields.empty:
        st.markdown("**Field genes**")
        st.dataframe(df_fields, width="stretch")
    if df_inter is not None and not df_inter.empty:
        st.markdown("**Interaction genes**")
        st.dataframe(df_inter, width="stretch")

    field_counts, field_perc, _, _ = compute_kernel_usage_stats(elements)
    if field_counts:
        parts = [f"{k}: {field_counts[k]} ({field_perc[k]:.1f}%)" for k in sorted(field_counts.keys())]
        st.caption("Field kernels: " + ", ".join(parts))


@st.fragment
def render_population_distributions(df: pd.DataFrame, selected_run_path: str):
    st.markdown("#### Population distributions")

    gens_tuple = tuple(df["generation"].tolist())
    gen_step, _ = _render_sampling_controls(
        "pop_dist",
        len(gens_tuple),
        include_max_solutions=False,
        help_text="Distributions are computed over the whole population already scanned for "
        "mutations/partial fitness, but plotting every generation as a separate box would be "
        "unreadable, so only every Nth sampled generation is shown.",
    )

    value_col = st.selectbox(
        "Value",
        ["fitness", "genome_size", "age"],
        key="pop_dist_value_col",
    )

    dist_df = compute_population_distributions(selected_run_path, gens_tuple)
    if dist_df.empty:
        st.info(
            "No population distribution data found. This usually means there are no "
            "statistics/generation_X.txt files for this run."
        )
        return

    sampled_generations = _sample_evenly(sorted(dist_df["generation"].unique().tolist()), int(len(gens_tuple) / gen_step) or 1)
    chart = chart_population_distribution(dist_df, value_col, sampled_generations)
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"{len(sampled_generations)} of {dist_df['generation'].nunique()} generations shown "
        f"({len(dist_df)} individual records scanned in total)."
    )


@st.fragment
def render_lineage_tracer(df: pd.DataFrame, selected_run_path: str):
    st.markdown("#### Lineage tracing")

    min_gen0 = int(df["generation"].min())
    max_gen0 = int(df["generation"].max())
    gen_sel_display = st.slider(
        "Generation whose best solution to trace back",
        min_value=display_gen(min_gen0),
        max_value=display_gen(max_gen0),
        value=display_gen(max_gen0),
        key="lineage_gen_slider",
    )
    gen_sel = gen_sel_display - 1

    sol_id = get_best_solution_id(selected_run_path, gen_sel)
    if sol_id is None:
        st.info(f"No best-solution record found for generation {gen_sel_display}.")
        return

    chain = trace_lineage(selected_run_path, gen_sel, sol_id)
    if not chain:
        st.info(f"Could not trace solution {sol_id} back through statistics/ records.")
        return

    lineage_df = pd.DataFrame(
        [
            {
                "generation": c["generation"],
                "id": c["record"]["id"],
                "fitness": c["record"]["fitness"],
                "hidden_fields": sum(1 for fg in c["record"]["field_genes"] if fg["type"] == "HIDDEN"),
                "enabled_connections": sum(1 for cg in c["record"]["connection_genes"] if cg["enabled"]),
            }
            for c in chain
        ]
    )

    root_gen = display_gen(int(lineage_df["generation"].iloc[0]))
    st.caption(
        f"Traced solution {sol_id} (generation {gen_sel_display}) back {len(chain)} generation(s) "
        f"to its bootstrap root at generation {root_gen}."
    )

    st.altair_chart(chart_lineage_fitness(lineage_df), width="stretch")

    display_df = lineage_df.copy()
    display_df["generation"] = display_df["generation"].apply(display_gen)
    lineage_column_config = {
        "generation": st.column_config.NumberColumn("generation", format="%d"),
        "id": st.column_config.NumberColumn("solution id", format="%d"),
        "fitness": st.column_config.NumberColumn("fitness", format="%.4f"),
        "hidden_fields": st.column_config.NumberColumn("hidden fields", format="%d"),
        "enabled_connections": st.column_config.NumberColumn("enabled connections", format="%d"),
    }
    st.dataframe(display_df, width="stretch", column_config=lineage_column_config, hide_index=True)


@st.fragment
def render_mutations_view(df: pd.DataFrame, gens_tuple: tuple, selected_run_path: str):
    st.divider()

    mut_events = compute_mutation_events(selected_run_path, gens_tuple)

    if mut_events.empty:
        st.info(
            "No mutation logs found in statistics/generation_X.txt "
            "(the 'last mutations{...}' field appears empty)."
        )
        return

    # --- Top: activity over time ---
    st.altair_chart(chart_mutations_per_generation(mut_events), width="stretch")
    st.caption("Total number of mutation events applied in each generation.")

    # --- Category breakdown: full width, horizontal bars (category names can be long) ---
    st.altair_chart(chart_mutation_categories(mut_events), width="stretch")
    st.caption("How mutation events are distributed across categories.")

    # --- Middle: most beneficial mutations (bar chart) ---
    st.divider()
    chart_eff = chart_mutation_effectiveness(mut_events)
    if chart_eff is not None:
        st.altair_chart(chart_eff, width="stretch")
        st.caption(
            "Mutations that, on average, appear in higher-fitness individuals "
            "compared to the global mean. Bar length shows the improvement."
        )

    # --- Tables: frequency + effectiveness ---
    st.divider()
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
        show_cols = crossing_df.sort_values(
            ["generation", "mean_fitness_in_gen"], ascending=[True, False]
        )
        crossing_column_config = {
            "generation": st.column_config.NumberColumn("generation", format="%d"),
            "mutation_raw": st.column_config.TextColumn("mutation", help="The mutation description as logged"),
            "category": st.column_config.TextColumn("category"),
            "occurrences_in_gen": st.column_config.NumberColumn(
                "count in gen", help="Times this mutation appears in that generation", format="%d"
            ),
            "mean_fitness_in_gen": st.column_config.NumberColumn(
                "mean fitness", help="Mean fitness of its carriers in that generation", format="%.4f"
            ),
        }
        st.dataframe(show_cols.head(20), width="stretch", column_config=crossing_column_config, hide_index=True)

    # --- Per-generation most impactful mutation timeline ---
    per_gen_best = compute_per_generation_best_mutation(mut_events)
    if not per_gen_best.empty:
        st.markdown("#### Per-generation most beneficial mutation")
        st.caption(
            "For each generation, this considers mutations that appear at least "
            "three times and selects the one whose carriers have the largest "
            "advantage over that generation's average fitness."
        )

        st.altair_chart(chart_best_mutation_timeline(per_gen_best), width="stretch")

        st.markdown("Top generations by mutation impact:")
        timeline_column_config = {
            "generation": st.column_config.NumberColumn("generation", format="%d"),
            "mutation": st.column_config.TextColumn("mutation", help="The mutation description as logged"),
            "category": st.column_config.TextColumn("category"),
            "occurrences": st.column_config.NumberColumn("count in gen", format="%d"),
            "mean_fitness": st.column_config.NumberColumn("mean fitness", format="%.4f"),
            "delta_vs_gen": st.column_config.NumberColumn(
                "Δ vs gen mean", help="Delta vs. that generation's mean fitness", format="%.4f"
            ),
        }
        st.dataframe(
            per_gen_best.sort_values("delta_vs_gen", ascending=False).head(10),
            width="stretch",
            column_config=timeline_column_config,
            hide_index=True,
        )


_NOT_RECORDED = "—"  # em dash


def _meta_field(meta: dict, section: str, key: str):
    """Value at meta[section][key], or None when either level is absent."""
    block = meta.get(section)
    return block.get(key) if isinstance(block, dict) else None


@st.fragment
def render_provenance_view(selected_run_path: str):
    """What built this run, and the machine it ran on -- from run_metadata.json.

    Every run directory predating neat-dnfs 0.1.0 has no such file, and an older
    build may have written only some of its sections, so every level here (file,
    section, field) has its own "not recorded" state rather than assuming the
    whole document is complete.
    """
    meta = load_run_metadata(selected_run_path)

    if meta is None:
        path = Path(selected_run_path) / "run_metadata.json"
        if path.exists():
            st.warning(
                f"`run_metadata.json` in this run folder could not be read -- "
                f"it is not valid JSON.\n\n{path}"
            )
        else:
            st.info(
                "No `run_metadata.json` in this run folder.\n\n"
                "Provenance recording was added in neat-dnfs 0.1.0; runs evolved before "
                "that did not write this file, and it cannot be reconstructed after the "
                "fact. Re-run this task with a current build to capture it."
            )
            st.caption(f"Looked for: {path}")
        return

    build = meta.get("build")
    dependencies = meta.get("dependencies")
    machine = meta.get("machine")
    run_parameters = meta.get("run_parameters")

    st.markdown("### Source")
    if isinstance(build, dict):
        git_sha = build.get("git_sha")
        col_sha, col_badge = st.columns([3, 2])
        with col_sha:
            st.code(git_sha or _NOT_RECORDED, language=None)
        with col_badge:
            if build.get("git_dirty") is True:
                st.badge("uncommitted changes", icon=":material/warning:", color="orange")
            elif build.get("git_dirty") is False:
                st.badge("clean tree", icon=":material/check:", color="green")
        if build.get("git_dirty") is True:
            st.caption(
                "This build included local edits that are not in any commit -- the "
                "exact source cannot be recovered from the SHA alone."
            )
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("neat-dnfs version", build.get("neat_dnfs_version") or _NOT_RECORDED, border=True)
        with metric_cols[1]:
            st.metric("build type", build.get("build_type") or _NOT_RECORDED, border=True)
    else:
        st.caption("Not recorded in this file.")

    st.markdown("### Build & machine")
    if isinstance(build, dict) or isinstance(machine, dict):
        build_col, machine_col = st.columns(2)
        with build_col:
            st.markdown("**Build**")
            compiler_id = _meta_field(meta, "build", "compiler_id")
            compiler_version = _meta_field(meta, "build", "compiler_version")
            compiler = f"{compiler_id} {compiler_version}".strip() if (compiler_id or compiler_version) else _NOT_RECORDED
            st.markdown(f"- Compiler: **{compiler}**")
            st.markdown(f"- CMake: **{_meta_field(meta, 'build', 'cmake_version') or _NOT_RECORDED}**")
            sanitizer = _meta_field(meta, "build", "sanitizer")
            st.markdown(f"- Sanitizer: **{sanitizer if sanitizer else 'none'}**")
        with machine_col:
            st.markdown("**Machine**")
            st.markdown(f"- OS: **{_meta_field(meta, 'machine', 'os') or _NOT_RECORDED}**")
            cpu_model = _meta_field(meta, "machine", "cpu_model")
            cpu_display = cpu_model.strip() if isinstance(cpu_model, str) and cpu_model.strip() else _NOT_RECORDED
            st.markdown(f"- CPU: **{cpu_display}**")
            cores = _meta_field(meta, "machine", "logical_cores")
            st.markdown(f"- Logical cores: **{cores if cores else _NOT_RECORDED}**")
            ram = format_ram_bytes(_meta_field(meta, "machine", "total_ram_bytes"))
            st.markdown(f"- RAM: **{ram or _NOT_RECORDED}**")
    else:
        st.caption("Not recorded in this file.")

    st.markdown("### Dependencies")
    if isinstance(dependencies, dict):
        dep_cols = st.columns(2)
        with dep_cols[0]:
            st.markdown("**imgui-platform-kit**")
            st.code(dependencies.get("imgui_platform_kit_sha") or _NOT_RECORDED, language=None)
        with dep_cols[1]:
            st.markdown("**dynamic-neural-field-composer**")
            st.code(dependencies.get("dynamic_neural_field_composer_sha") or _NOT_RECORDED, language=None)

        vcpkg_blob = dependencies.get("vcpkg_packages")
        pkg_df = (
            parse_vcpkg_package_list(vcpkg_blob)
            if isinstance(vcpkg_blob, str) and vcpkg_blob
            else pd.DataFrame()
        )
        if not pkg_df.empty:
            with st.expander(f"vcpkg packages ({len(pkg_df)})", expanded=False):
                st.dataframe(
                    pkg_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "package": st.column_config.TextColumn("package"),
                        "version": st.column_config.TextColumn("version"),
                        "description": st.column_config.TextColumn("description"),
                    },
                )
        else:
            st.caption("No vcpkg package list recorded in this file.")
    else:
        st.caption("Not recorded in this file.")

    st.markdown("### Run parameters")
    if isinstance(run_parameters, dict):
        parallel = run_parameters.get("parallel_evolution")
        st.metric(
            "parallel evolution",
            "yes" if parallel is True else ("no" if parallel is False else _NOT_RECORDED),
            border=True,
        )
    else:
        st.caption("Not recorded in this file.")
    st.caption(
        "Seed and resolved config are not recorded yet -- they depend on ongoing "
        "reproducibility work, so this section is expected to grow."
    )


def render_topology_frequency(all_run_metrics: list):
    st.markdown("#### Topology-frequency map")

    freq_df = compute_topology_frequency(all_run_metrics)
    if freq_df.empty:
        st.info("No successful runs to build a topology-frequency map from.")
        return

    st.altair_chart(chart_topology_frequency_heatmap(freq_df), width="stretch")

    top_row = freq_df.loc[freq_df["count"].idxmax()]
    st.caption(
        f"Most frequent topology: **{int(top_row['hidden_fields'])} hidden field(s), "
        f"{int(top_row['enabled_connections'])} enabled connection(s)** "
        f"({int(top_row['count'])} of {int(freq_df['count'].sum())} successful runs)."
    )

    successful = [m for m in all_run_metrics if m.get("success") and m.get("generation_to_threshold") is not None]
    if len(successful) >= 3:
        complexity = [m["hidden_fields_count"] + m["enabled_connections_count"] for m in successful]
        gens_to_threshold = [m["generation_to_threshold"] for m in successful]
        rho, p = spearman_correlation(complexity, gens_to_threshold)
        if rho == rho:  # not NaN
            st.caption(
                f"Spearman correlation between topology complexity (hidden fields + enabled "
                f"connections) and generations-to-threshold: r_s = {rho:.3f}, p = {p:.4f} "
                f"(n = {len(successful)}). p-value uses the standard t-distribution "
                f"approximation; treat as indicative only for small n."
            )

    st.markdown("##### Distance to a reference architecture")
    st.caption(
        "Compare every successful run's final topology against a reference, e.g. an ablation "
        "condition's expected topology."
    )
    rc1, rc2 = st.columns(2)
    with rc1:
        ref_hidden = st.number_input("Reference hidden fields", min_value=0, value=1, key="ref_arch_hidden")
    with rc2:
        ref_conn = st.number_input("Reference enabled connections", min_value=0, value=5, key="ref_arch_conn")

    successful_runs = [m for m in all_run_metrics if m.get("success")]
    if successful_runs:
        dist_rows = [
            {
                "run": m.get("run_dir", "?"),
                "hidden_fields": m["hidden_fields_count"],
                "enabled_connections": m["enabled_connections_count"],
                "distance": topology_distance(
                    m["hidden_fields_count"], m["enabled_connections_count"], int(ref_hidden), int(ref_conn)
                ),
            }
            for m in successful_runs
        ]
        dist_df = pd.DataFrame(dist_rows).sort_values("distance")
        n_exact = int((dist_df["distance"] == 0).sum())
        st.caption(
            f"{n_exact} of {len(dist_df)} successful runs match the reference architecture exactly "
            f"({int(ref_hidden)} hidden, {int(ref_conn)} connections)."
        )
        dist_column_config = {
            "run": st.column_config.TextColumn("run"),
            "hidden_fields": st.column_config.NumberColumn("hidden fields", format="%d"),
            "enabled_connections": st.column_config.NumberColumn("enabled connections", format="%d"),
            "distance": st.column_config.NumberColumn(
                "distance", help="Manhattan distance to the reference architecture", format="%d"
            ),
        }
        st.dataframe(dist_df, width="stretch", column_config=dist_column_config, hide_index=True)
    else:
        st.info("No successful runs to compare against a reference architecture.")


def render_partial_component_failure_rates(all_run_metrics: list, partial_targets: dict):
    st.markdown("#### Partial-component bottlenecks across runs")
    st.caption(
        "For each partial-fitness component, the fraction of runs whose best solution failed "
        "that component's target -- which components most often gate success in this experiment."
    )
    rates_df = compute_partial_component_failure_rates(all_run_metrics, partial_targets)
    if rates_df.empty:
        st.info("No partial-component failure data available.")
        return
    st.altair_chart(chart_partial_component_failure_rates(rates_df), width="stretch")
    rates_column_config = {
        "component": st.column_config.NumberColumn("component", format="%d"),
        "failure_count": st.column_config.NumberColumn("failures", format="%d"),
        "total_runs": st.column_config.NumberColumn("total runs", format="%d"),
        "failure_rate": st.column_config.ProgressColumn(
            "failure rate", help="Fraction of runs whose best solution failed this component's target", min_value=0.0, max_value=1.0, format="percent"
        ),
    }
    st.dataframe(rates_df, width="stretch", column_config=rates_column_config, hide_index=True)


def _cross_run_partial_bounds(parsed_runs: list, components: list) -> dict:
    """Data-derived (min, max) bounds per partial-fitness component, computed across every
    generation of every run already sitting in `parsed_runs` -- no new I/O (parsed_runs is
    already loaded by the caller), matching the Fitness view's per-run bound computation
    instead of a hardcoded range."""
    bounds = {}
    for comp in components:
        values = [0.0, 1.0]
        for run in parsed_runs:
            for parts in run["partial_vectors"]:
                if len(parts) >= comp:
                    values.append(parts[comp - 1])
        bounds[comp] = (float(min(values)), float(max(values)))
    return bounds


@st.fragment
def render_experiment_view(base_dir_str: str):
    st.divider()

    st.markdown("### Experiment-level statistics across runs")

    # Success criterion (refactor): run is successful if there exists at least one generation
    # where *all* partial fitness components meet/exceed their targets simultaneously.
    if "partial_targets" not in st.session_state:
        st.session_state["partial_targets"] = {}

    partial_targets = dict(st.session_state["partial_targets"])

    # Always fetch -- cheap after the first call (@st.cache_data + on-disk cache, per
    # _load_experiment_runs_parsed's own docstring). Used both to infer how many partial
    # components exist when targets aren't set yet, and to compute data-derived per-component
    # bounds for the number_inputs below.
    #
    # This used to test `"partial_best" in df_first.columns` -- a column
    # load_overview never produces -- so it always fell through to a full
    # compute_partial_fitness() statistics/ scan just to count components.
    try:
        parsed_runs = _load_experiment_runs_parsed(base_dir_str)
    except OSError as e:
        st.warning(f"Could not read runs in {base_dir_str}: {e}")
        parsed_runs = []

    if not partial_targets and parsed_runs:
        n_parts = max(
            (len(v) for v in parsed_runs[0]["partial_vectors"]),
            default=0,
        )
        default_thr = float(st.session_state.get("target_fitness", 0.9))
        for i in range(1, n_parts + 1):
            partial_targets[i] = default_thr

    with st.expander("Partial fitness targets (success criteria)", expanded=True):
        if not partial_targets:
            st.info(
                "No partial fitness components detected yet. "
                "Make sure your runs have statistics/generation_X.txt files."
            )
        else:
            comp_bounds = _cross_run_partial_bounds(parsed_runs, sorted(partial_targets.keys()))
            cols = st.columns(min(6, max(1, len(partial_targets))))
            for idx, comp in enumerate(sorted(partial_targets.keys())):
                with cols[idx % len(cols)]:
                    col_min, col_max = comp_bounds.get(comp, (-10.0, 10.0))
                    partial_targets[comp] = _clamp(partial_targets[comp], col_min, col_max)
                    partial_targets[comp] = float(
                        st.number_input(
                            f"Target p{comp}",
                            value=float(partial_targets[comp]),
                            min_value=col_min,
                            max_value=col_max,
                            step=0.01,
                            key=f"exp_target_p{comp}",
                        )
                    )

        st.session_state["partial_targets"] = partial_targets

    # convergence / architecture (across runs)
    targets_items = tuple(sorted((int(k), float(v)) for k, v in partial_targets.items()))
    conv = compute_experiment_convergence(base_dir_str, targets_items)
    render_experiment_convergence(conv, partial_targets)

    st.divider()
    all_run_metrics = conv.get("all_run_metrics", [])
    if all_run_metrics:
        chart_overlay = chart_all_runs_overlay(all_run_metrics, float(st.session_state.get("target_fitness", 0.9)))
        st.altair_chart(chart_overlay, width="stretch")

        st.divider()
        render_topology_frequency(all_run_metrics)

        if partial_targets:
            st.divider()
            render_partial_component_failure_rates(all_run_metrics, partial_targets)

    st.divider()

    # runtime + total mutation statistics (statistics/*_total.txt etc.)
    agg_totals, df_totals = compute_experiment_totals(base_dir_str)
    render_experiment_totals(agg_totals, df_totals)


@st.fragment
def render_cross_experiment_view(data_root_str: str):
    st.divider()
    st.markdown("### Cross-experiment comparison")
    st.caption(
        "Compare success rate, convergence speed, and architecture across several experiment "
        "folders under data/ -- e.g. different tasks or different ablation conditions. Uses the "
        "data root selected in the sidebar."
    )

    data_root = Path(data_root_str).expanduser()
    if not data_root.exists() or not data_root.is_dir():
        st.error(f"Data root does not exist or is not a directory:\n{data_root}")
        return

    experiment_dirs = find_experiment_dirs(data_root)
    if not experiment_dirs:
        st.warning("No experiment folders with runs found under this data root.")
        return

    names = [name for name, _ in experiment_dirs]
    selected_names = st.multiselect("Experiments to compare", names, default=names[: min(3, len(names))])
    if len(selected_names) < 2:
        st.info("Select at least 2 experiments to compare.")
        return

    experiment_paths = dict(experiment_dirs)
    partial_targets = st.session_state.get("partial_targets", {})
    targets_items = tuple(sorted((int(k), float(v)) for k, v in partial_targets.items()))

    summary_rows = []
    gens_by_experiment = {}
    for name in selected_names:
        conv = compute_experiment_convergence(str(experiment_paths[name]), targets_items)
        if not conv:
            continue
        all_metrics = conv.get("all_run_metrics", [])
        successful = [m for m in all_metrics if m.get("success")]
        gens_by_experiment[name] = [
            m["generation_to_threshold"] for m in successful if m.get("generation_to_threshold") is not None
        ]
        summary_rows.append(
            {
                "experiment": name,
                "n runs": conv.get("total_runs", 0),
                "success rate": conv.get("success_rate", 0.0),
                "median gens to threshold": conv.get("median_generations_to_threshold"),
                "hidden fields (mean)": conv.get("mean_hidden_fields"),
                "enabled connections (mean)": conv.get("mean_enabled_connections"),
            }
        )

    if not summary_rows:
        st.info("No data available for the selected experiments.")
        return

    summary_column_config = {
        "experiment": st.column_config.TextColumn("experiment"),
        "n runs": st.column_config.NumberColumn("runs", format="%d"),
        "success rate": st.column_config.NumberColumn("success rate", format="percent"),
        "median gens to threshold": st.column_config.NumberColumn("median gens to threshold", format="%.1f"),
        "hidden fields (mean)": st.column_config.NumberColumn("hidden fields (mean)", format="%.2f"),
        "enabled connections (mean)": st.column_config.NumberColumn("enabled connections (mean)", format="%.2f"),
    }
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", column_config=summary_column_config, hide_index=True)

    plottable = {name: vals for name, vals in gens_by_experiment.items() if vals}
    if plottable:
        st.altair_chart(chart_cross_experiment_boxplot(plottable), width="stretch")

    if len(plottable) >= 2:
        st.markdown("#### Mann-Whitney U vs. reference")
        reference_name = st.selectbox("Reference experiment", list(plottable.keys()), key="compare_reference")
        ref_vals = plottable[reference_name]
        rows = []
        for name, vals in plottable.items():
            if name == reference_name:
                continue
            u, p = mann_whitney_u(ref_vals, vals)
            rows.append({"experiment": name, "n": len(vals), "U": u, "p-value": p})
        if rows:
            mw_column_config = {
                "experiment": st.column_config.TextColumn("experiment"),
                "n": st.column_config.NumberColumn("n", format="%d"),
                "U": st.column_config.NumberColumn("U statistic", format="%.2f"),
                "p-value": st.column_config.NumberColumn("p-value", format="%.4f"),
            }
            st.dataframe(pd.DataFrame(rows), width="stretch", column_config=mw_column_config, hide_index=True)
            st.caption(
                f"Generations-to-threshold vs. **{reference_name}** (n={len(ref_vals)}). "
                "Normal-approximation p-value; treat as indicative only for small n."
            )
