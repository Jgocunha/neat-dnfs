from viz.stats import display_gen, longest_non_improving_run


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
