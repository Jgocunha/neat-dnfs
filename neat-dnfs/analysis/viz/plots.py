import math
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx

from . import theme
from .theme import CATEGORICAL_CYCLE, COLOR_AVG, COLOR_BEST, COLOR_FAILURE, COLOR_STRUCTURAL_CHANGE, COLOR_SUCCESS, COLOR_TARGET, DIVERGING_CMAP, FIG_GRID_CELL, FIG_HALF, FIG_SQUARE, FIG_WIDE

theme.apply_plot_style()

# All-runs overlay can exceed Altair's default 5000-row safety limit for larger experiments
# (e.g. 40 runs x 200 generations = 8000 rows) -- disabled deliberately rather than downsampling,
# since downsampling would defeat the chart's purpose (seeing every individual run's curve).
alt.data_transformers.disable_max_rows()


def show_fig(fig):
    """Render a figure and close it, so it doesn't leak into Matplotlib's global
    figure registry. Every plot_*() helper creates a new figure via plt.subplots()
    and nothing ever closed them, so figures accumulated on every rerun (e.g. every
    slider drag)."""
    st.pyplot(fig)
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    x = df["generation"] + 1
    ax.plot(x, df["avg_fitness"], label="avg. fitness")
    ax.plot(x, df["best_fitness"], label="best fitness")
    ax.axhline(target_fitness, linestyle="--", color=COLOR_TARGET, label=f"target ({target_fitness:.3f})")

    # Mark ALL generations where all partial targets were met (simultaneously).
    if success_generations:
        # groupby/max (not set_index) so a run whose overview logged the same generation twice
        # (e.g. an interrupted/restarted run) still yields one scalar per generation, not a Series.
        best_by_gen = df.assign(gen_display=df["generation"] + 1).groupby("gen_display")["best_fitness"].max()
        xs = [g for g in success_generations if g in best_by_gen.index]
        ys = [float(best_by_gen.loc[g]) for g in xs]

        if xs:
            label = f'{success_label} (g={", ".join(map(str, xs))})'
            ax.scatter(xs, ys, marker="D", s=40, label=label, zorder=5)

    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("Total fitness")
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
            if best_col not in partial_df.columns:
                comp_index += 1
                continue

            target = partial_targets.get(comp_index)

            with col:
                fig, ax = plt.subplots(figsize=FIG_GRID_CELL)
                if avg_col in partial_df.columns:
                    ax.plot(gen, partial_df[avg_col], label="avg.")
                ax.plot(gen, partial_df[best_col], label="best")
                ax.axhline(target, linestyle="--", color=COLOR_TARGET, label=f"target ({target:.3f})")

                reached = partial_df[partial_df[best_col] >= target]
                if not reached.empty:
                    row = reached.iloc[0]
                    ax.scatter(row["generation"] + 1, row[best_col], marker="o", zorder=5, label="target reached")

                ax.set_xlabel("generation")
                ax.set_ylabel("fitness")
                ax.set_title(f"partial fitness {comp_index}")
                ax.legend(fontsize="x-small")
                fig.tight_layout()
                show_fig(fig)

            comp_index += 1


def plot_all_runs_overlay(all_run_metrics: list, target_fitness: float):
    """Every run's best-fitness-vs-generation curve overlaid, colored by success/failure, with
    a median line and IQR band across runs. Runs of different lengths are aligned by generation
    index (position 0, 1, 2, ... from each run's own start) and padded with NaN past their own
    length, so the band narrows to only the still-running runs at the tail."""
    fig, ax = plt.subplots(figsize=(9, 4))
    if not all_run_metrics:
        ax.set_title("All-runs fitness overlay (no data)")
        fig.tight_layout()
        return fig

    max_len = max(len(m["fitness_values"]) for m in all_run_metrics)
    matrix = np.full((len(all_run_metrics), max_len), np.nan)

    success_label_used = False
    failure_label_used = False
    for i, m in enumerate(all_run_metrics):
        vals = m["fitness_values"]
        matrix[i, : len(vals)] = vals
        is_success = bool(m.get("success"))
        color = COLOR_SUCCESS if is_success else COLOR_FAILURE
        label = None
        if is_success and not success_label_used:
            label, success_label_used = "successful run", True
        elif not is_success and not failure_label_used:
            label, failure_label_used = "unsuccessful run", True
        ax.plot(range(1, len(vals) + 1), vals, color=color, alpha=0.25, linewidth=1, label=label)

    xs_all = range(1, max_len + 1)
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)

    ax.plot(xs_all, median, color="black", linewidth=2, label="median")
    ax.fill_between(xs_all, q25, q75, color="black", alpha=0.15, label="IQR")
    ax.axhline(target_fitness, linestyle="--", color=COLOR_TARGET, label=f"target ({target_fitness:.3f})")

    n_success = sum(1 for m in all_run_metrics if m.get("success"))
    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness")
    ax.set_title(f"All-runs fitness overlay ({len(all_run_metrics)} runs, {n_success} successful)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def plot_topology_frequency_heatmap(freq_df: pd.DataFrame):
    """Topology-frequency map (RAS paper Fig. 4): hidden-field count x enabled-connection
    count, marker size/annotation = number of successful runs converging to that exact
    topology. A sized scatter reads better than a dense grid heatmap at the run counts this
    tool typically has on disk (tens, not the paper's 97) -- most cells would otherwise be
    empty."""
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    if freq_df is None or freq_df.empty:
        ax.set_title("Topology-frequency map (no successful runs)")
        fig.tight_layout()
        return fig

    max_count = freq_df["count"].max()
    sizes = 200 + 1800 * (freq_df["count"] / max_count)
    ax.scatter(
        freq_df["enabled_connections"],
        freq_df["hidden_fields"],
        s=sizes,
        alpha=0.5,
        edgecolors="black",
    )
    for _, row in freq_df.iterrows():
        ax.annotate(
            str(int(row["count"])),
            (row["enabled_connections"], row["hidden_fields"]),
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("enabled connections")
    ax.set_ylabel("hidden fields")
    ax.set_title(f"Topology-frequency map ({int(freq_df['count'].sum())} successful runs)")
    ax.set_ylim(bottom=-0.5)
    ax.set_xlim(left=-0.5)
    fig.tight_layout()
    return fig


def plot_partial_component_heatmap(best_df: pd.DataFrame, partial_targets: dict):
    """Heatmap of (best_p_i - target_i) per generation, one row per component with a target
    set, diverging around 0 (green = above target, red = below). Encoding the value relative to
    its own target directly in color avoids needing a separate per-row threshold overlay, which
    a raw-value heatmap with a different target per row could not show cleanly."""
    comps = sorted(int(c) for c in partial_targets)
    fig, ax = plt.subplots(figsize=(9, max(2.5, len(comps) * 0.4)))
    if not comps or best_df is None or best_df.empty:
        ax.set_title("Partial-component heatmap (no data)")
        fig.tight_layout()
        return fig

    gens = best_df["generation"] + 1
    rows = []
    for c in comps:
        col = f"best_p{c}"
        target = float(partial_targets[c])
        if col in best_df.columns:
            rows.append((best_df[col] - target).to_numpy())
        else:
            rows.append(np.full(len(best_df), np.nan))
    matrix = np.array(rows)

    finite = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=DIVERGING_CMAP,
        vmin=-vmax,
        vmax=vmax,
        extent=[gens.iloc[0], gens.iloc[-1], len(comps) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([f"p{c}" for c in comps])
    ax.set_xlabel("generation")
    ax.set_title("Partial-component value vs. target (purple = above, orange = below)")
    fig.colorbar(im, ax=ax, label="best value - target")
    fig.tight_layout()
    return fig


def plot_first_crossing_per_component(crossing: dict):
    comps = sorted(crossing.keys())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if not comps:
        ax.set_title("First crossing per component (no data)")
        fig.tight_layout()
        return fig

    values = [crossing[c] for c in comps]
    reached = [v for v in values if v is not None]
    never_height = (max(reached) * 1.15 + 1) if reached else 1
    heights = [v if v is not None else never_height for v in values]
    colors = [COLOR_SUCCESS if v is not None else COLOR_FAILURE for v in values]

    ax.bar([f"p{c}" for c in comps], heights, color=colors)
    for i, v in enumerate(values):
        if v is None:
            ax.text(i, heights[i], "never", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_ylabel("first generation crossing target")
    ax.set_title("First-crossing generation per partial-fitness component")
    fig.tight_layout()
    return fig


def plot_partial_component_failure_rates(rates_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if rates_df is None or rates_df.empty:
        ax.set_title("Partial-component failure rates (no data)")
        fig.tight_layout()
        return fig

    ax.bar([f"p{c}" for c in rates_df["component"]], rates_df["failure_rate"] * 100)
    ax.set_ylabel("% of runs failing this component")
    ax.set_title("Which components most often gate success")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


def plot_species_counts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(df["generation"] + 1, df["num_species"], label="species")
    ax.plot(df["generation"] + 1, df["num_active_species"], label="active species")
    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.set_title("Species count evolution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_innovation_growth(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(df["generation"] + 1, df["innovation_number"], label="innovation number")
    ax.set_xlabel("generation")
    ax.set_ylabel("innovation number")
    ax.set_title("Innovation numbers growth")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_genome_topology_curves(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(df["generation"] + 1, df["avg_genome_size"], label="avg genome size")
    ax.plot(df["generation"] + 1, df["avg_field_genes"], label="avg field genes")
    ax.plot(df["generation"] + 1, df["avg_conn_genes"], label="avg connection genes")
    ax.set_xlabel("generation")
    ax.set_ylabel("genes")
    ax.set_title("Genome topology")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_topology_trajectory(traj_df: pd.DataFrame):
    """Best-solution hidden-field/enabled-connection counts across generations, with a marker on
    every generation where the hidden-field count changed from the previous one (a structural
    change on the winning lineage)."""
    fig, ax = plt.subplots(figsize=FIG_HALF)
    gens = traj_df["generation"] + 1
    ax.plot(gens, traj_df["hidden_fields"], label="hidden fields", marker="", color=CATEGORICAL_CYCLE[1])
    ax.plot(gens, traj_df["enabled_connections"], label="enabled connections", color=CATEGORICAL_CYCLE[0])

    changed = traj_df["hidden_fields"].diff().fillna(0) != 0
    if changed.any():
        ax.scatter(
            gens[changed],
            traj_df.loc[changed, "hidden_fields"],
            color=COLOR_STRUCTURAL_CHANGE,
            zorder=5,
            label="hidden-field count changed",
        )

    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.set_title("Best-solution topology over time")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_lineage_fitness(lineage_df: pd.DataFrame):
    """Fitness of a traced lineage (one ancestor per generation it spans), with markers where
    the lineage's own hidden-field count changed from its previous ancestor."""
    fig, ax = plt.subplots(figsize=FIG_HALF)
    gens = lineage_df["generation"] + 1
    ax.plot(gens, lineage_df["fitness"], marker="o", markersize=3, label="lineage fitness")

    changed = lineage_df["hidden_fields"].diff().fillna(0) != 0
    if changed.any():
        ax.scatter(
            gens[changed],
            lineage_df.loc[changed, "fitness"],
            color=COLOR_STRUCTURAL_CHANGE,
            zorder=5,
            label="hidden-field count changed",
        )

    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("Traced lineage fitness")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_population_distribution(dist_df: pd.DataFrame, value_col: str, sampled_generations: list):
    """Box plot of `value_col` (fitness / genome_size / age) across the population, one box
    per generation in `sampled_generations` (a subset of dist_df["generation"] values, chosen
    by the caller for readability -- 1000 individuals x 200 generations would otherwise be
    unreadable as a single figure)."""
    labels = []
    data = []
    for g in sampled_generations:
        vals = dist_df.loc[dist_df["generation"] == g, value_col]
        if vals.empty:
            continue
        labels.append(str(g + 1))
        data.append(vals.values)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), 3.5))
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_xlabel("generation")
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(f"Population {value_col.replace('_', ' ')} distribution")
    if len(labels) > 20:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(90)
    fig.tight_layout()
    return fig


def plot_species_membership_stacked(species_meta: dict, top_n: int = 15):
    """Stacked-area membership of the top_n species (by peak membership), with the rest of the
    population collapsed into a single 'other' band."""
    fig, ax = plt.subplots(figsize=(9, 4))
    all_gens = sorted({g for m in species_meta.values() for g in m["members_by_gen"]})
    if not species_meta or not all_gens:
        ax.set_title("Species membership over time (no data)")
        fig.tight_layout()
        return fig

    ranked = sorted(species_meta.items(), key=lambda kv: kv[1]["max_members"], reverse=True)
    top_ids = [sid for sid, _ in ranked[:top_n]]
    other_ids = [sid for sid, _ in ranked[top_n:]]

    gen_index = {g: i for i, g in enumerate(all_gens)}
    series = {}
    for sid in top_ids:
        arr = [0] * len(all_gens)
        for g, count in species_meta[sid]["members_by_gen"].items():
            arr[gen_index[g]] = count
        series[str(sid)] = arr

    if other_ids:
        other_arr = [0] * len(all_gens)
        for sid in other_ids:
            for g, count in species_meta[sid]["members_by_gen"].items():
                other_arr[gen_index[g]] += count
        series["other"] = other_arr

    xs = [g + 1 for g in all_gens]
    ax.stackplot(xs, *series.values(), labels=list(series.keys()))
    ax.set_xlabel("generation")
    ax.set_ylabel("members")
    ax.set_title(f"Species membership over time (top {len(top_ids)} of {len(species_meta)} species)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_species_lifespans(species_meta: dict, top_n: int = 30):
    """Horizontal bar per species spanning [first_gen, last_gen], sorted by first-seen
    generation. Capped to top_n species (earliest first) since a run can log thousands."""
    ranked = sorted(species_meta.items(), key=lambda kv: kv[1]["first_gen"])[:top_n]

    fig, ax = plt.subplots(figsize=(FIG_HALF[0], max(3, len(ranked) * 0.25)))
    if not ranked:
        ax.set_title("Species lifespans (no data)")
        fig.tight_layout()
        return fig

    for i, (sid, m) in enumerate(ranked):
        start = m["first_gen"] + 1
        end = m["last_gen"] + 1
        ax.barh(i, end - start + 1, left=start, height=0.6)

    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([str(sid) for sid, _ in ranked])
    ax.set_xlabel("generation")
    ax.set_ylabel("species id")
    ax.set_title(f"Species lifespans (first {len(ranked)} of {len(species_meta)} species, by first-seen generation)")
    fig.tight_layout()
    return fig


def plot_species_champion_trajectory(traj_df: pd.DataFrame, species_id: int):
    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(traj_df["generation"] + 1, traj_df["fitness"], marker="o", markersize=3)
    ax.set_xlabel("generation")
    ax.set_ylabel("champion fitness")
    ax.set_title(f"Species {species_id} champion fitness over time")
    fig.tight_layout()
    return fig


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

    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(df_usage["generation"] + 1, df_usage[y1], label="Gaussian")
    ax.plot(df_usage["generation"] + 1, df_usage[y2], label="Mexican-hat")
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mutations_per_generation(mut_events: pd.DataFrame):
    """Line plot: how many mutations occurred each generation."""
    per_gen = (
        mut_events.groupby("generation")["mutation_raw"]
        .count()
        .reset_index(name="num_mutations")
    )
    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(per_gen["generation"] + 1, per_gen["num_mutations"])
    ax.set_xlabel("generation")
    ax.set_ylabel("number of mutations")
    ax.set_title("Mutation activity per generation")
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


def plot_mutation_categories(mut_events: pd.DataFrame):
    """Bar chart: how many mutation events per high-level category."""
    per_cat = (
        mut_events.groupby("category")["mutation_raw"]
        .count()
        .reset_index(name="num_events")
        .sort_values("num_events", ascending=False)
    )
    fig, ax = plt.subplots(figsize=FIG_HALF)
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


def plot_best_mutation_timeline(per_gen_df: pd.DataFrame):
    """Line plot: per-generation most beneficial mutation (delta vs gen mean)."""
    fig, ax = plt.subplots(figsize=FIG_HALF)
    ax.plot(per_gen_df["generation"] + 1, per_gen_df["delta_vs_gen"])
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("generation")
    ax.set_ylabel("Δ best mutation vs gen mean")
    ax.set_title("Per-generation most beneficial mutation (approx.)")
    fig.tight_layout()
    return fig


def plot_run_duration_histogram(durations: pd.Series):
    fig, ax = plt.subplots(figsize=FIG_GRID_CELL)
    ax.hist(durations.dropna(), bins=10)
    ax.set_xlabel("duration (hours)")
    ax.set_ylabel("runs")
    ax.set_title("Run duration")
    fig.tight_layout()
    return fig


def plot_seconds_per_generation_histogram(values: pd.Series):
    fig, ax = plt.subplots(figsize=FIG_GRID_CELL)
    ax.hist(values.dropna(), bins=10)
    ax.set_xlabel("sec / generation")
    ax.set_ylabel("runs")
    ax.set_title("Time per generation")
    fig.tight_layout()
    return fig


def plot_convergence_generations_histogram(generations_to_threshold: list):
    fig, ax = plt.subplots(figsize=FIG_GRID_CELL)
    ax.hist(generations_to_threshold, bins=min(10, len(generations_to_threshold)))
    ax.set_xlabel("generations to success")
    ax.set_ylabel("runs")
    ax.set_title("How many generations runs need\n to meet all partial targets")
    fig.tight_layout()
    return fig


def plot_architecture_complexity_scatter(hidden: list, connections: list, labels: list):
    fig, ax = plt.subplots(figsize=FIG_GRID_CELL)
    ax.scatter(hidden, connections)

    for h, c, label in zip(hidden, connections, labels):
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
    return fig


def plot_cross_experiment_boxplot(plottable: dict):
    """plottable: {experiment_name: [generations_to_threshold, ...]}."""
    fig, ax = plt.subplots(figsize=(FIG_HALF[0], 4))
    ax.boxplot(plottable.values(), tick_labels=list(plottable.keys()), showfliers=False)
    ax.set_ylabel("generations to threshold")
    ax.set_title("Convergence speed across experiments")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(20)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive Altair charts for the five highest-value views. These coexist
# with their matplotlib plot_* counterparts above (which stay in use for the
# markdown exporter and every other chart) -- both engines read colors from
# viz/theme.py so the two don't visually drift from each other. Altair does
# not inherit matplotlib's rcParams, so each function below applies theme
# tokens explicitly via alt.Scale(range=[...]).
# ---------------------------------------------------------------------------


def chart_total_fitness(
    df: pd.DataFrame,
    target_fitness: float,
    success_generations: list[int] | None = None,
    success_label: str = "All partial targets met",
):
    """Interactive counterpart to plot_total_fitness: hover tooltips show the exact
    generation/fitness value instead of requiring the reader to eyeball a static PNG."""
    plot_df = df.assign(gen_display=df["generation"] + 1)
    long_df = plot_df.melt(
        id_vars=["gen_display"],
        value_vars=["avg_fitness", "best_fitness"],
        var_name="series",
        value_name="fitness",
    )
    series_labels = {"avg_fitness": "avg. fitness", "best_fitness": "best fitness"}
    long_df["series_label"] = long_df["series"].map(series_labels)

    line = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("fitness:Q", title="fitness"),
            color=alt.Color(
                "series_label:N",
                title=None,
                scale=alt.Scale(domain=["avg. fitness", "best fitness"], range=[COLOR_AVG, COLOR_BEST]),
            ),
            tooltip=[
                alt.Tooltip("gen_display:Q", title="generation"),
                alt.Tooltip("series_label:N", title="series"),
                alt.Tooltip("fitness:Q", format=".4f"),
            ],
        )
    )

    target_rule = (
        alt.Chart(pd.DataFrame({"target": [target_fitness]}))
        .mark_rule(strokeDash=[4, 4], color=COLOR_TARGET)
        .encode(y="target:Q", tooltip=[alt.Tooltip("target:Q", title="target", format=".4f")])
    )

    layers = [line, target_rule]

    if success_generations:
        best_by_gen = plot_df.groupby("gen_display")["best_fitness"].max()
        xs = [g for g in success_generations if g in best_by_gen.index]
        if xs:
            points_df = pd.DataFrame({"gen_display": xs, "fitness": [float(best_by_gen.loc[g]) for g in xs]})
            points = (
                alt.Chart(points_df)
                .mark_point(shape="diamond", size=90, filled=True, color=COLOR_SUCCESS)
                .encode(
                    x="gen_display:Q",
                    y="fitness:Q",
                    tooltip=[
                        alt.Tooltip("gen_display:Q", title="generation"),
                        alt.Tooltip("fitness:Q", title=success_label, format=".4f"),
                    ],
                )
            )
            layers.append(points)

    return alt.layer(*layers).properties(title="Total fitness").interactive()


def chart_partial_fitness_grid(partial_df: pd.DataFrame, partial_targets: dict):
    """Interactive counterpart to plot_partial_fitness_grid. Keeps the same st.columns grid
    shell and per-panel st.info empty-state handling as the matplotlib version -- this function
    isn't a pure fig-returning helper either, it draws directly, matching the original."""
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
    gen = partial_df["generation"] + 1

    comp_index = 1
    for _ in range(rows):
        cols = st.columns(per_row)
        for col in cols:
            if comp_index > num_partial:
                break

            best_col = f"best_p{comp_index}"
            avg_col = f"avg_p{comp_index}"
            if best_col not in partial_df.columns:
                comp_index += 1
                continue

            target = partial_targets.get(comp_index)

            with col:
                series = {"best": partial_df[best_col]}
                if avg_col in partial_df.columns:
                    series["avg."] = partial_df[avg_col]
                cell_df = pd.DataFrame({"gen_display": gen, **series}).melt(
                    id_vars=["gen_display"], var_name="series", value_name="fitness"
                )
                line = (
                    alt.Chart(cell_df)
                    .mark_line()
                    .encode(
                        x=alt.X("gen_display:Q", title="generation"),
                        y=alt.Y("fitness:Q", title="fitness"),
                        color=alt.Color("series:N", title=None),
                        tooltip=[
                            alt.Tooltip("gen_display:Q", title="generation"),
                            alt.Tooltip("series:N"),
                            alt.Tooltip("fitness:Q", format=".4f"),
                        ],
                    )
                )
                target_rule = (
                    alt.Chart(pd.DataFrame({"target": [target]}))
                    .mark_rule(strokeDash=[4, 4], color=COLOR_TARGET)
                    .encode(y="target:Q")
                )
                layers = [line, target_rule]

                reached = partial_df[partial_df[best_col] >= target]
                if not reached.empty:
                    row = reached.iloc[0]
                    reached_df = pd.DataFrame({"gen_display": [row["generation"] + 1], "fitness": [row[best_col]]})
                    layers.append(
                        alt.Chart(reached_df)
                        .mark_point(shape="circle", size=70, filled=True, color=COLOR_SUCCESS)
                        .encode(x="gen_display:Q", y="fitness:Q", tooltip=[alt.Tooltip("gen_display:Q", title="target reached at gen")])
                    )

                chart = alt.layer(*layers).properties(title=f"partial fitness {comp_index}", height=180)
                st.altair_chart(chart, width="stretch")

            comp_index += 1


def chart_all_runs_overlay(all_run_metrics: list, target_fitness: float):
    """Interactive counterpart to plot_all_runs_overlay. Individual run lines are colored by
    success/failure with click-to-isolate via the legend; hovering a line shows exact values."""
    if not all_run_metrics:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().properties(title="All-runs fitness overlay (no data)")

    max_len = max(len(m["fitness_values"]) for m in all_run_metrics)
    matrix = np.full((len(all_run_metrics), max_len), np.nan)

    run_rows = []
    for i, m in enumerate(all_run_metrics):
        vals = m["fitness_values"]
        matrix[i, : len(vals)] = vals
        is_success = bool(m.get("success"))
        run_label = "successful run" if is_success else "unsuccessful run"
        for gen0, v in enumerate(vals):
            run_rows.append({"run": m.get("run_dir", i), "generation": gen0 + 1, "fitness": v, "outcome": run_label})
    runs_df = pd.DataFrame(run_rows)

    xs_all = np.arange(1, max_len + 1)
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    band_df = pd.DataFrame({"generation": xs_all, "median": median, "q25": q25, "q75": q75})

    legend_selection = alt.selection_point(fields=["outcome"], bind="legend")

    run_lines = (
        alt.Chart(runs_df)
        .mark_line(opacity=0.25)
        .encode(
            x=alt.X("generation:Q", title="generation"),
            y=alt.Y("fitness:Q", title="best fitness"),
            color=alt.Color(
                "outcome:N",
                title=None,
                scale=alt.Scale(domain=["successful run", "unsuccessful run"], range=[COLOR_SUCCESS, COLOR_FAILURE]),
            ),
            detail="run:N",
            opacity=alt.condition(legend_selection, alt.value(0.25), alt.value(0.03)),
            tooltip=[
                alt.Tooltip("run:N"),
                alt.Tooltip("generation:Q"),
                alt.Tooltip("fitness:Q", format=".4f"),
                alt.Tooltip("outcome:N"),
            ],
        )
        .add_params(legend_selection)
    )

    band = alt.Chart(band_df).mark_area(opacity=0.15, color="black").encode(
        x="generation:Q", y="q25:Q", y2="q75:Q"
    )
    median_line = alt.Chart(band_df).mark_line(color="black", strokeWidth=2).encode(
        x="generation:Q", y=alt.Y("median:Q"), tooltip=[alt.Tooltip("generation:Q"), alt.Tooltip("median:Q", format=".4f")]
    )
    target_rule = (
        alt.Chart(pd.DataFrame({"target": [target_fitness]}))
        .mark_rule(strokeDash=[4, 4], color=COLOR_TARGET)
        .encode(y="target:Q")
    )

    n_success = sum(1 for m in all_run_metrics if m.get("success"))
    title = f"All-runs fitness overlay ({len(all_run_metrics)} runs, {n_success} successful)"
    return alt.layer(band, run_lines, median_line, target_rule).properties(title=title).interactive()


def chart_topology_trajectory(traj_df: pd.DataFrame):
    """Interactive counterpart to plot_topology_trajectory."""
    plot_df = traj_df.assign(gen_display=traj_df["generation"] + 1)
    long_df = plot_df.melt(
        id_vars=["gen_display"],
        value_vars=["hidden_fields", "enabled_connections"],
        var_name="series",
        value_name="count",
    )
    series_labels = {"hidden_fields": "hidden fields", "enabled_connections": "enabled connections"}
    long_df["series_label"] = long_df["series"].map(series_labels)

    line = (
        alt.Chart(long_df)
        .mark_line()
        .encode(
            x=alt.X("gen_display:Q", title="generation"),
            y=alt.Y("count:Q", title="count"),
            color=alt.Color(
                "series_label:N",
                title=None,
                scale=alt.Scale(
                    domain=["hidden fields", "enabled connections"],
                    range=[CATEGORICAL_CYCLE[1], CATEGORICAL_CYCLE[0]],
                ),
            ),
            tooltip=[
                alt.Tooltip("gen_display:Q", title="generation"),
                alt.Tooltip("series_label:N", title="series"),
                alt.Tooltip("count:Q"),
            ],
        )
    )

    layers = [line]
    changed = plot_df["hidden_fields"].diff().fillna(0) != 0
    if changed.any():
        changed_df = plot_df.loc[changed, ["gen_display", "hidden_fields"]]
        points = (
            alt.Chart(changed_df)
            .mark_point(shape="circle", size=80, filled=True, color=COLOR_STRUCTURAL_CHANGE)
            .encode(
                x="gen_display:Q",
                y="hidden_fields:Q",
                tooltip=[alt.Tooltip("gen_display:Q", title="generation"), alt.Tooltip("hidden_fields:Q", title="hidden fields")],
            )
        )
        layers.append(points)

    return alt.layer(*layers).properties(title="Best-solution topology over time").interactive()


def chart_species_membership(species_meta: dict, top_n: int = 15):
    """Interactive counterpart to plot_species_membership_stacked: legend entries can be
    clicked to isolate a single species' band."""
    all_gens = sorted({g for m in species_meta.values() for g in m["members_by_gen"]})
    if not species_meta or not all_gens:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_area().properties(title="Species membership over time (no data)")

    ranked = sorted(species_meta.items(), key=lambda kv: kv[1]["max_members"], reverse=True)
    top_ids = [sid for sid, _ in ranked[:top_n]]
    other_ids = [sid for sid, _ in ranked[top_n:]]

    rows = []
    for g in all_gens:
        for sid in top_ids:
            count = species_meta[sid]["members_by_gen"].get(g, 0)
            rows.append({"generation": g + 1, "species": str(sid), "members": count})
        if other_ids:
            other_count = sum(species_meta[sid]["members_by_gen"].get(g, 0) for sid in other_ids)
            rows.append({"generation": g + 1, "species": "other", "members": other_count})

    long_df = pd.DataFrame(rows)
    legend_selection = alt.selection_point(fields=["species"], bind="legend")

    chart = (
        alt.Chart(long_df)
        .mark_area()
        .encode(
            x=alt.X("generation:Q", title="generation"),
            y=alt.Y("members:Q", title="members", stack="zero"),
            color=alt.Color("species:N", title="species", scale=alt.Scale(range=CATEGORICAL_CYCLE)),
            opacity=alt.condition(legend_selection, alt.value(0.85), alt.value(0.08)),
            tooltip=[alt.Tooltip("generation:Q"), alt.Tooltip("species:N"), alt.Tooltip("members:Q")],
        )
        .add_params(legend_selection)
        .properties(title=f"Species membership over time (top {len(top_ids)} of {len(species_meta)} species)")
    )
    return chart
