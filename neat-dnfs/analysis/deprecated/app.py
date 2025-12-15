#!/usr/bin/env python3
import re
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ---------- PARSING LOGIC (from previous script, slightly adapted) ----------

def parse_overview_line(line: str):
    """
    Parse a single line of per_generation_overview.txt.
    Returns a dict with all numeric fields (or None if parsing fails).
    """

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


def load_overview(overview_path: Path) -> pd.DataFrame:
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
    """
    Scan base_dir for subdirectories containing per_generation_overview.txt
    Return list of (run_name, path) tuples.
    """
    runs = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            overview = child / "per_generation_overview.txt"
            if overview.exists():
                runs.append((child.name, child))
    return runs


# ---------- STREAMLIT APP ----------

def main():
    st.set_page_config(page_title="NEAT-DNFs Evolution Overview", layout="wide")

    st.title("📈 NEAT-DNFs Evolution Overview Dashboard")

    # Sidebar: base directory selection
    st.sidebar.header("Settings")

    default_base = Path(".").absolute()
    base_dir_str = st.sidebar.text_input(
        "Base directory with run folders",
        value=str(default_base),
        help="Directory that contains your run folders (each run has per_generation_overview.txt).",
    )

    base_dir = Path(base_dir_str).expanduser()

    if not base_dir.exists() or not base_dir.is_dir():
        st.error(f"Base directory does not exist or is not a directory: {base_dir}")
        st.stop()

    # Find runs
    runs = find_runs_with_overview(base_dir)
    if not runs:
        st.warning("No subfolders with per_generation_overview.txt found.")
        st.stop()

    run_names = [name for name, _ in runs]
    selected_run_name = st.sidebar.selectbox("Select run", run_names)

    # Get path to selected run
    selected_run_path = dict(runs)[selected_run_name]
    overview_path = selected_run_path / "per_generation_overview.txt"

    st.sidebar.markdown(f"**Selected run folder:** `{selected_run_path}`")
    st.sidebar.markdown(f"`{overview_path.name}` will be parsed.")

    # Load data
    try:
        df = load_overview(overview_path)
    except Exception as e:
        st.error(f"Error parsing {overview_path}: {e}")
        st.stop()

    # Show basic info
    st.subheader("Run summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Generations", int(df["generation"].max()))
    with col2:
        st.metric("Best fitness (final)", f"{df['best_fitness'].iloc[-1]:.4f}")
    with col3:
        st.metric("Species (final)", int(df["num_species"].iloc[-1]))

    st.write("Preview of parsed per-generation data:")
    st.dataframe(df.head())

    # Plots: 3 rows, each using full width
    # 1) Fitness
    st.markdown("---")
    st.subheader("Fitness over generations")
    fig1, ax1 = plt.subplots()
    ax1.plot(df["generation"], df["avg_fitness"], label="Average fitness")
    ax1.plot(df["generation"], df["best_fitness"], label="Best fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # 2) Species
    st.markdown("---")
    st.subheader("Species count over generations")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["generation"], df["num_species"], label="Total species")
    ax2.plot(df["generation"], df["num_active_species"], label="Active species")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Number of species")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # 3) Genome complexity
    st.markdown("---")
    st.subheader("Genome complexity over generations")
    fig3, ax3 = plt.subplots()
    ax3.plot(df["generation"], df["avg_genome_size"], label="Avg genome size")
    ax3.plot(df["generation"], df["avg_conn_genes"], label="Avg connection genes")
    ax3.plot(df["generation"], df["avg_field_genes"], label="Avg field genes")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Genes")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # Optional: show raw table expandable
    with st.expander("Show full per-generation table"):
        st.dataframe(df)


if __name__ == "__main__":
    main()
