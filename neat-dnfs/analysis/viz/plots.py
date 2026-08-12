import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import networkx as nx

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
