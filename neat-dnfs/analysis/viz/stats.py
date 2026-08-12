import pandas as pd
from dataclasses import dataclass

def display_gen(gen0: int) -> int:
    """The one place a stored (0-based) generation index becomes the 1-based
    number shown to the user. per_generation_overview.txt stores generations
    0-based; every other generation field in this module keeps a `0` suffix and
    stays 0-based right up until it is rendered, specifically to make this
    conversion impossible to apply inconsistently."""
    return gen0 + 1


def longest_non_improving_run(values: list, eps: float = 1e-9) -> int:
    """Longest streak of consecutive values that do not improve on the previous
    one, where "improve" means a gain larger than `eps` (guards against
    floating-point noise between equal-looking values counting as progress)."""
    longest = current = 0
    for i in range(1, len(values)):
        if values[i] - values[i - 1] > eps:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


@dataclass
class FitnessStats:
    final_gen0: int
    best_final: float
    avg_final: float
    max_best: float
    gen_max_best0: int
    total_cross_gen0: int | None  # generation total best fitness first crossed target
    best_at_target: float | None
    longest_stagnation: int
    auc_best: float
    auc_avg: float


def compute_fitness_stats(df: pd.DataFrame, target_fitness: float) -> FitnessStats:
    final_row = df.iloc[-1]
    max_best = float(df["best_fitness"].max())
    gen_max_best0 = int(df.loc[df["best_fitness"].idxmax(), "generation"])

    reached = df[df["best_fitness"] >= target_fitness]
    if not reached.empty:
        total_cross_gen0 = int(reached["generation"].iloc[0])
        best_at_target = float(reached["best_fitness"].iloc[0])
    else:
        total_cross_gen0 = None
        best_at_target = None

    return FitnessStats(
        final_gen0=int(final_row["generation"]),
        best_final=float(final_row["best_fitness"]),
        avg_final=float(final_row["avg_fitness"]),
        max_best=max_best,
        gen_max_best0=gen_max_best0,
        total_cross_gen0=total_cross_gen0,
        best_at_target=best_at_target,
        longest_stagnation=longest_non_improving_run(df["best_fitness"].astype(float).tolist()),
        auc_best=float(df["best_fitness"].mean()),
        auc_avg=float(df["avg_fitness"].mean()),
    )


@dataclass
class SpeciesStats:
    final_gen0: int
    final_species: int
    final_active: int
    avg_species: float
    avg_active: float
    max_active_species: int
    gen_max_active0: int
    total_species: int
    extinct_species: int
    avg_lifespan: float
    max_lifespan: int
    max_life_sid: int | None
    avg_max_members: float
    avg_offspring: float


def compute_species_stats(df: pd.DataFrame, species_meta: dict) -> SpeciesStats:
    last = df.iloc[-1]
    final_gen0 = int(last["generation"])

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
        if m["last_gen"] == final_gen0 and not m["last_extinct"]:
            active_final += 1

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

    return SpeciesStats(
        final_gen0=final_gen0,
        final_species=int(last["num_species"]),
        final_active=int(last["num_active_species"]),
        avg_species=float(df["num_species"].mean()),
        avg_active=float(df["num_active_species"].mean()),
        max_active_species=int(df["num_active_species"].max()),
        gen_max_active0=int(df.loc[df["num_active_species"].idxmax(), "generation"]),
        total_species=total_species,
        extinct_species=total_species - active_final,
        avg_lifespan=avg_lifespan,
        max_lifespan=max_lifespan,
        max_life_sid=max_life_sid,
        avg_max_members=sum(max_members_list) / len(max_members_list) if max_members_list else 0.0,
        avg_offspring=sum(offspring_list) / len(offspring_list) if offspring_list else 0.0,
    )


@dataclass
class TopologyStats:
    final_gen0: int
    genome_size_final: float
    field_genes_final: float
    conn_genes_final: float
    genome_size_delta: float
    field_genes_delta: float
    conn_genes_delta: float
    genome_size_per_gen: float
    field_genes_per_gen: float
    conn_genes_per_gen: float
    avg_conn_per_field_final: float


def compute_topology_stats(df: pd.DataFrame) -> TopologyStats:
    first = df.iloc[0]
    last = df.iloc[-1]

    g0, gN = float(first["avg_genome_size"]), float(last["avg_genome_size"])
    f0, fN = float(first["avg_field_genes"]), float(last["avg_field_genes"])
    c0, cN = float(first["avg_conn_genes"]), float(last["avg_conn_genes"])

    gens = last["generation"] - first["generation"]
    gens = gens if gens > 0 else 1

    return TopologyStats(
        final_gen0=int(last["generation"]),
        genome_size_final=gN,
        field_genes_final=fN,
        conn_genes_final=cN,
        genome_size_delta=gN - g0,
        field_genes_delta=fN - f0,
        conn_genes_delta=cN - c0,
        genome_size_per_gen=(gN - g0) / gens,
        field_genes_per_gen=(fN - f0) / gens,
        conn_genes_per_gen=(cN - c0) / gens,
        avg_conn_per_field_final=(cN / fN) if fN > 0 else 0.0,
    )
