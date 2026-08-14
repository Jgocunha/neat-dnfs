import math
import pandas as pd
from dataclasses import dataclass

def _rank(values: list) -> list:
    """Average-rank ranking (ties share the mean of the ranks they span), 1-based."""
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg_rank = sum(range(i, j + 1)) / (j - i + 1) + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued-fraction term of the regularized incomplete beta function (Lentz's method,
    as in Press et al., Numerical Recipes ch. 6.4)."""
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b), via the standard continued-fraction
    algorithm. No scipy dependency in this project (see analysis/requirements.txt) -- this is
    the piece needed to get an honest p-value out of a t-statistic without one."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(x, a, b) / a
    return 1.0 - math.exp(lbeta) * _betacf(1 - x, b, a) / b


def _student_t_two_tailed_pvalue(t: float, df: int) -> float:
    """Two-tailed p-value for a Student's t statistic with `df` degrees of freedom, via the
    standard identity p = I_{df/(df+t^2)}(df/2, 1/2). Verified against the textbook critical
    value t=2.228, df=10 -> p=0.05."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _incomplete_beta(x, df / 2.0, 0.5)


def spearman_correlation(x: list, y: list) -> tuple[float, float]:
    """Tie-corrected Spearman rank correlation (rho, p-value), with no scipy dependency.

    The p-value uses the standard t-distribution approximation (rho converted to a t-statistic
    with n-2 degrees of freedom), which is the same approximation scipy.stats.spearmanr uses
    for n >= ~10. For smaller n, treat the p-value as indicative only -- this is a property of
    the approximation itself, not a bug in this implementation (an exact p-value would need a
    permutation test).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return float("nan"), float("nan")

    rx = _rank(x)
    ry = _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")

    rho = max(-1.0, min(1.0, cov / (sx * sy)))
    if abs(rho) >= 1.0:
        return rho, 0.0

    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return rho, _student_t_two_tailed_pvalue(t, n - 2)


def display_gen(gen0: int) -> int:
    """The one place a stored (0-based) generation index becomes the 1-based
    number shown to the user. per_generation_overview.txt stores generations
    0-based; every other generation field in this module keeps a `0` suffix and
    stays 0-based right up until it is rendered, specifically to make this
    conversion impossible to apply inconsistently."""
    return gen0 + 1


def compute_topology_frequency(all_metrics: list) -> pd.DataFrame:
    """How many successful runs converged to each (hidden_fields, enabled_connections)
    topology -- the data behind a topology-frequency heatmap (RAS paper Fig. 4). Failed runs
    are excluded, matching the paper's own figure, which counts only successful runs.

    Returns a DataFrame with columns: hidden_fields, enabled_connections, count.
    """
    counts: dict[tuple[int, int], int] = {}
    for m in all_metrics:
        if not m.get("success"):
            continue
        key = (int(m["hidden_fields_count"]), int(m["enabled_connections_count"]))
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["hidden_fields", "enabled_connections", "count"])

    rows = [
        {"hidden_fields": h, "enabled_connections": c, "count": n}
        for (h, c), n in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(["hidden_fields", "enabled_connections"]).reset_index(drop=True)


def _standard_normal_two_tailed_pvalue(z: float) -> float:
    """Two-tailed p-value from a standard-normal z statistic, via the complementary error
    function (math.erfc is in the stdlib -- no scipy needed for this one)."""
    return math.erfc(abs(z) / math.sqrt(2))


def mann_whitney_u(x: list, y: list) -> tuple[float, float]:
    """Mann-Whitney U test (rank-sum test), tie-corrected, with a normal-approximation
    p-value -- no scipy dependency (see analysis/requirements.txt). The normal approximation is
    standard practice for this test once both groups have a handful of observations each
    (commonly cited threshold: both n >= ~8); for very small samples treat the p-value as
    indicative only, same caveat as spearman_correlation's t-approximation.

    Returns (U, p-value) where U is the smaller of the two one-sided U statistics.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    combined = list(x) + list(y)
    ranks = _rank(combined)
    r1 = sum(ranks[:n1])

    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # tie correction for the normal approximation's variance term
    tie_groups: dict[float, int] = {}
    for r in ranks:
        tie_groups[r] = tie_groups.get(r, 0) + 1
    n = n1 + n2
    tie_sum = sum(t**3 - t for t in tie_groups.values())

    mean_u = n1 * n2 / 2.0
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_sum / (n * (n - 1))) if n > 1 else 0.0
    if var_u <= 0:
        return u, float("nan")

    z = (u - mean_u) / math.sqrt(var_u)
    return u, _standard_normal_two_tailed_pvalue(z)


def compute_partial_component_failure_rates(all_metrics: list, partial_targets: dict) -> pd.DataFrame:
    """For each partial-fitness component, the fraction of runs where that component was among
    the best solution's failed targets -- the RAS paper's "f4/f5/f6 remained below threshold"
    failure attribution, generalised across every run in an experiment instead of 3 hand-picked
    ones. Reads the `failed_partial_components` list already computed per run by
    viz.experiment._evaluate_run_targets.

    Returns a DataFrame with columns: component, failure_count, total_runs, failure_rate.
    """
    total = len(all_metrics)
    if total == 0 or not partial_targets:
        return pd.DataFrame(columns=["component", "failure_count", "total_runs", "failure_rate"])

    fail_counts = {int(c): 0 for c in partial_targets}
    for m in all_metrics:
        for c in m.get("failed_partial_components", []):
            if c in fail_counts:
                fail_counts[c] += 1

    rows = [
        {"component": c, "failure_count": n, "total_runs": total, "failure_rate": n / total}
        for c, n in sorted(fail_counts.items())
    ]
    return pd.DataFrame(rows)


def topology_distance(hidden: int, connections: int, ref_hidden: int, ref_connections: int) -> int:
    """Manhattan distance from (hidden, connections) to a reference architecture -- used both
    to rank runs by closeness to a paper's reported minimal architecture and, at distance 0, to
    identify exact matches."""
    return abs(hidden - ref_hidden) + abs(connections - ref_connections)


def find_invariant_violations(
    trajectory_df: pd.DataFrame, target_hidden: int, target_connections: int, tolerance: int = 0
) -> list[int]:
    """Generations (0-based) where a topology trajectory (see
    viz.parsing.compute_topology_trajectory) departs from an exact target by more than
    `tolerance`. tolerance=0 is a hard invariant check -- e.g. the ablation study's A1/A2
    conditions, which must hold an exact field/connection count on every single generation."""
    violations = []
    for _, row in trajectory_df.iterrows():
        dist = topology_distance(
            int(row["hidden_fields"]), int(row["enabled_connections"]), target_hidden, target_connections
        )
        if dist > tolerance:
            violations.append(int(row["generation"]))
    return violations


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
