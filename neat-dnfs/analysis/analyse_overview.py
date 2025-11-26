#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------- CONFIG ----------
# Either pass the run folder as an argument, or hardcode it here.
# Example:
#   python analyze_overview.py "/path/to/run/2025-11-26 11h59m25s"
# If no argument is given, it uses the current directory.
# ----------------------------

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

    # Convert numeric fields
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


def load_overview(run_dir: Path) -> pd.DataFrame:
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
            else:
                # If needed, you can print or log problematic lines here
                # print("Could not parse line:", line[:120], "...")
                pass

    if not rows:
        raise RuntimeError("No lines could be parsed from per_generation_overview.txt")

    df = pd.DataFrame(rows)
    df.sort_values("generation", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def plot_fitness(df: pd.DataFrame):
    plt.figure()
    plt.plot(df["generation"], df["avg_fitness"], label="Average fitness")
    plt.plot(df["generation"], df["best_fitness"], label="Best fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_species(df: pd.DataFrame):
    plt.figure()
    plt.plot(df["generation"], df["num_species"], label="Total species")
    plt.plot(df["generation"], df["num_active_species"], label="Active species")
    plt.xlabel("Generation")
    plt.ylabel("Number of species")
    plt.title("Species count over generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_genome_size(df: pd.DataFrame):
    plt.figure()
    plt.plot(df["generation"], df["avg_genome_size"], label="Avg genome size")
    plt.plot(df["generation"], df["avg_conn_genes"], label="Avg connection genes")
    plt.plot(df["generation"], df["avg_field_genes"], label="Avg field genes")
    plt.xlabel("Generation")
    plt.ylabel("Genes")
    plt.title("Genome complexity over generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = Path(".")

    df = load_overview(run_dir)
    print("Parsed per_generation_overview:")
    print(df.head())

    plot_fitness(df)
    plot_species(df)
    plot_genome_size(df)

    plt.show()


if __name__ == "__main__":
    main()
