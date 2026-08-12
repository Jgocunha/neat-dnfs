import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .genome import build_topology_graph, compute_kernel_usage_stats, summarize_best_solution_genome
from .stats import compute_fitness_stats, compute_species_stats, compute_topology_stats, display_gen
from .plots import plot_best_mutation_timeline, plot_genome_topology_curves, plot_innovation_growth, plot_kernel_usage_time, plot_mutation_categories, plot_mutation_effectiveness, plot_mutations_per_generation, plot_partial_fitness_grid, plot_species_counts, plot_topology_graph, plot_total_fitness, show_fig
from .parsing import compute_mutation_events, compute_partial_fitness, compute_per_generation_best_mutation, compute_population_kernel_usage, compute_species_meta, compute_target_crossing_mutations, generations_all_partial_meet_targets, get_species_for_generation, load_best_solution_architecture
from .experiment import _load_experiment_runs_parsed, compute_experiment_convergence, compute_experiment_totals

def render_fitness_stats(
    df: pd.DataFrame,
    target_fitness: float,
    partial_threshold_generation: int | None = None,
):
    s = compute_fitness_stats(df, target_fitness)

    st.markdown("#### Statistics")
    st.markdown(
        f"""
        **Final generation (g = {display_gen(s.final_gen0)})**  
        • Best fitness: **{s.best_final:.4f}**  
        • Average fitness: **{s.avg_final:.4f}**  

        **Overall**  
        • Max best fitness: **{s.max_best:.4f}** (reached at generation {display_gen(s.gen_max_best0)})  
        • Mean best fitness over run (AUC): **{s.auc_best:.4f}**  
        • Mean average fitness over run (AUC): **{s.auc_avg:.4f}**  
        • Longest stagnation (best fitness not improving): **{s.longest_stagnation}** generations  
        """,
    )

    if partial_threshold_generation is not None:
        st.success(f"✅ All partial fitness targets met simultaneously at generation **{partial_threshold_generation}**.")
    else:
        st.warning("❌ All partial fitness targets were **never** met simultaneously in this run.")

    if s.total_cross_gen0 is not None:
        st.caption(f"Reference: total best fitness first crossed the chosen overall target at generation {display_gen(s.total_cross_gen0)}.")


def render_species_stats(df: pd.DataFrame, species_meta: dict):
    s = compute_species_stats(df, species_meta)

    st.markdown("#### Species statistics")
    st.markdown(
        f"""
        **Final generation (g = {display_gen(s.final_gen0)})**  
        • Species: **{s.final_species}**  
        • Active species: **{s.final_active}**  

        **Across run**  
        • Total distinct species ever created: **{s.total_species}**  
        • Species that went extinct by final generation: **{s.extinct_species}**  
        • Average number of species per generation: **{s.avg_species:.2f}**  
        • Average number of active species per generation: **{s.avg_active:.2f}**  
        • Max number of **active** species in a generation: **{s.max_active_species}** (at generation {display_gen(s.gen_max_active0)})  
        """
    )

    st.markdown(
        f"""
        **Species lifetime & size**  
        • Average species lifespan: **{s.avg_lifespan:.2f}** generations  
        • Longest-lived species: **{s.max_life_sid}** (lifespan {s.max_lifespan} generations)  
        • Average max members per species: **{s.avg_max_members:.2f}**  
        • Average total offspring assigned per species: **{s.avg_offspring:.2f}**  
        """
    )


def render_topology_stats(df: pd.DataFrame):
    s = compute_topology_stats(df)

    st.markdown("#### Topology statistics")
    st.markdown(
        f"""
        **Final generation (g = {display_gen(s.final_gen0)})**  
        • Avg genome size: **{s.genome_size_final:.2f}**  
        • Avg field genes: **{s.field_genes_final:.2f}**  
        • Avg connection genes: **{s.conn_genes_final:.2f}**  

        **Growth over run**  
        • Genome size change: **{s.genome_size_delta:+.2f}** (≈ {s.genome_size_per_gen:+.3f} per generation)  
        • Field genes change: **{s.field_genes_delta:+.2f}** (≈ {s.field_genes_per_gen:+.3f} per generation)  
        • Connection genes change: **{s.conn_genes_delta:+.2f}** (≈ {s.conn_genes_per_gen:+.3f} per generation)  

        **Ratios**  
        • Avg connections per field at final gen: **{s.avg_conn_per_field_final:.2f}**  
        • Avg genome size / population size is accessible from per-generation statistics if needed.  
        """
    )


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


@st.fragment
def render_species_view(df: pd.DataFrame, gens_tuple: tuple, selected_run_path: str):
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


@st.fragment
def render_topology_view(df: pd.DataFrame, selected_run_path: str):
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


@st.fragment
def render_mutations_view(df: pd.DataFrame, gens_tuple: tuple, selected_run_path: str):
    st.markdown("---")

    mut_events = compute_mutation_events(selected_run_path, gens_tuple)

    if mut_events.empty:
        st.info(
            "No mutation logs found in statistics/generation_X.txt "
            "(the 'last mutations{...}' field appears empty)."
        )
        return

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
        show_cols = crossing_df.sort_values(
            ["generation", "mean_fitness_in_gen"], ascending=[True, False]
        )
        show_cols = show_cols.rename(
            columns={
                "generation": "generation",
                "mutation_raw": "mutation description",
                "category": "mutation category",
                "occurrences_in_gen": "times this mutation appears in that generation",
                "mean_fitness_in_gen": "mean fitness of its carriers in that generation",
            }
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


@st.fragment
def render_experiment_view(base_dir_str: str):
    st.markdown("---")

    st.markdown("### Experiment-level statistics across runs")

    # Success criterion (refactor): run is successful if there exists at least one generation
    # where *all* partial fitness components meet/exceed their targets simultaneously.
    if "partial_targets" not in st.session_state:
        st.session_state["partial_targets"] = {}

    partial_targets = dict(st.session_state["partial_targets"])

    # If not set yet, infer how many partial components exist from the first run we find.
    # Default each partial target to the current global target_fitness.
    #
    # This used to test `"partial_best" in df_first.columns` -- a column
    # load_overview never produces -- so it always fell through to a full
    # compute_partial_fitness() statistics/ scan just to count components.
    # _load_experiment_runs_parsed is already cached and target-independent,
    # and its partial_vectors give us the same count for free.
    if not partial_targets:
        try:
            parsed_runs = _load_experiment_runs_parsed(base_dir_str)
        except OSError as e:
            st.warning(f"Could not read runs in {base_dir_str}: {e}")
            parsed_runs = []

        if parsed_runs:
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
    conv = compute_experiment_convergence(base_dir_str, targets_items)
    render_experiment_convergence(conv, partial_targets)

    st.markdown("---")

    # runtime + total mutation statistics (statistics/*_total.txt etc.)
    agg_totals, df_totals = compute_experiment_totals(base_dir_str)
    render_experiment_totals(agg_totals, df_totals)
