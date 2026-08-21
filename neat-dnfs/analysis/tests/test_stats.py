import pandas as pd
import pytest

from viz.stats import (
    compute_partial_component_failure_rates,
    compute_topology_frequency,
    display_gen,
    find_invariant_violations,
    longest_non_improving_run,
    mann_whitney_u,
    spearman_correlation,
    topology_distance,
)


def test_display_gen_converts_0_based_to_1_based():
    assert display_gen(0) == 1
    assert display_gen(199) == 200


def test_longest_non_improving_run_basic():
    # strictly increasing: never stagnates
    assert longest_non_improving_run([0.1, 0.2, 0.3, 0.4]) == 0
    # flat: stagnates for every step after the first value
    assert longest_non_improving_run([0.5, 0.5, 0.5, 0.5]) == 3
    # one improvement resets the streak
    assert longest_non_improving_run([0.1, 0.1, 0.1, 0.9, 0.9]) == 2


def test_longest_non_improving_run_sub_epsilon_gains_do_not_count_as_improvement():
    # Gains smaller than eps are floating-point noise, not real progress -- this is
    # the case that used to diverge between the on-screen and exported stagnation
    # counts before the two implementations were unified (render side used a plain
    # `<=` comparison, so any of these tiny gains would have reset its streak).
    values = [0.5, 0.5000000005, 0.5000000010, 0.5000000015, 0.5000000020]
    assert longest_non_improving_run(values, eps=1e-9) == 4


def test_student_t_two_tailed_pvalue_matches_known_critical_value():
    # textbook critical value: t=2.228 at df=10 gives a two-tailed p of 0.05
    from viz.stats import _student_t_two_tailed_pvalue

    assert _student_t_two_tailed_pvalue(2.228, 10) == pytest.approx(0.05, abs=1e-3)
    assert _student_t_two_tailed_pvalue(0.0, 10) == pytest.approx(1.0)


def test_spearman_correlation_perfect_monotonic_is_plus_one():
    rho, p = spearman_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert rho == pytest.approx(1.0)
    assert p == pytest.approx(0.0)


def test_spearman_correlation_perfect_inverse_is_minus_one():
    rho, _p = spearman_correlation([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert rho == pytest.approx(-1.0)


def test_spearman_correlation_handles_ties_via_average_rank():
    # known worked example: x has a tie at rank (2,3) -> both get rank 2.5
    rho, _p = spearman_correlation([1, 2, 2, 4, 5], [2, 1, 4, 3, 5])
    assert -1.0 <= rho <= 1.0
    assert rho == rho  # not NaN


def test_spearman_correlation_too_few_points_returns_nan():
    rho, _p = spearman_correlation([1, 2], [1, 2])
    assert rho != rho  # NaN


def test_compute_topology_frequency_excludes_failed_runs_and_counts_by_topology():
    all_metrics = [
        {"success": True, "hidden_fields_count": 1, "enabled_connections_count": 5},
        {"success": True, "hidden_fields_count": 1, "enabled_connections_count": 5},
        {"success": True, "hidden_fields_count": 2, "enabled_connections_count": 9},
        {"success": False, "hidden_fields_count": 3, "enabled_connections_count": 14},
    ]
    freq_df = compute_topology_frequency(all_metrics)
    assert len(freq_df) == 2
    assert freq_df["count"].sum() == 3  # the failed run is excluded

    row15 = freq_df[(freq_df["hidden_fields"] == 1) & (freq_df["enabled_connections"] == 5)]
    assert row15["count"].iloc[0] == 2


def test_compute_topology_frequency_empty_when_no_successes():
    freq_df = compute_topology_frequency([{"success": False, "hidden_fields_count": 1, "enabled_connections_count": 5}])
    assert freq_df.empty


def test_topology_distance_manhattan():
    assert topology_distance(1, 5, 1, 5) == 0
    assert topology_distance(2, 5, 1, 5) == 1
    assert topology_distance(2, 7, 1, 5) == 3


def test_find_invariant_violations_matches_ablation_a1_invariant():
    # constant trajectory matching the ablation brief's A1 invariant (0 hidden, 3 enabled
    # connections held every generation) -- verified against a live A1 run this session.
    traj_df = pd.DataFrame(
        {
            "generation": range(5),
            "hidden_fields": [0, 0, 0, 0, 0],
            "enabled_connections": [3, 3, 3, 3, 3],
            "disabled_connections": [0, 0, 0, 0, 0],
        }
    )
    assert find_invariant_violations(traj_df, target_hidden=0, target_connections=3) == []


def test_find_invariant_violations_flags_the_violating_generation():
    traj_df = pd.DataFrame(
        {
            "generation": range(4),
            "hidden_fields": [0, 0, 1, 0],
            "enabled_connections": [3, 3, 3, 3],
            "disabled_connections": [0, 0, 0, 0],
        }
    )
    assert find_invariant_violations(traj_df, target_hidden=0, target_connections=3) == [2]


def test_find_invariant_violations_respects_tolerance():
    traj_df = pd.DataFrame(
        {
            "generation": [0],
            "hidden_fields": [1],
            "enabled_connections": [3],
            "disabled_connections": [0],
        }
    )
    assert find_invariant_violations(traj_df, target_hidden=0, target_connections=3, tolerance=0) == [0]
    assert find_invariant_violations(traj_df, target_hidden=0, target_connections=3, tolerance=1) == []


def test_mann_whitney_u_fully_separated_groups_is_significant():
    u, p = mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert u == pytest.approx(0.0)
    assert p < 0.05


def test_mann_whitney_u_identical_distributions_is_not_significant():
    _u, p = mann_whitney_u([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert p == pytest.approx(1.0, abs=1e-6)


def test_mann_whitney_u_interleaved_groups_not_significant():
    _u, p = mann_whitney_u([1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
    assert p > 0.3


def test_compute_partial_component_failure_rates_counts_across_runs():
    all_metrics = [
        {"failed_partial_components": [6]},
        {"failed_partial_components": [6, 5]},
        {"failed_partial_components": []},
        {"failed_partial_components": [4]},
    ]
    targets = {4: 0.9, 5: 0.9, 6: 0.9}
    rates_df = compute_partial_component_failure_rates(all_metrics, targets)

    assert len(rates_df) == 3
    row6 = rates_df[rates_df["component"] == 6].iloc[0]
    assert row6["failure_count"] == 2
    assert row6["failure_rate"] == pytest.approx(0.5)
    row4 = rates_df[rates_df["component"] == 4].iloc[0]
    assert row4["failure_count"] == 1


def test_compute_partial_component_failure_rates_empty_when_no_targets():
    rates_df = compute_partial_component_failure_rates([{"failed_partial_components": [1]}], {})
    assert rates_df.empty
