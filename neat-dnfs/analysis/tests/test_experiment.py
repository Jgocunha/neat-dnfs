import pytest

from viz.experiment import compute_partial_fitness_best_only
from viz.parsing import compute_partial_fitness


def _overview_line(gen0, sol_id, fitness, partials):
    parts_text = ", ".join(str(p) for p in partials) + ", "
    return (
        f"Current generation: {gen0} Number of solutions: 2 Number of species: 1 "
        f"Number of active species: 1 Has fitness improved: yes "
        f"Number of generations without improvement: 0 Average fitness: {fitness} "
        f"Best fitness: {fitness} Innovation number: 0 Average genome size: 4.0 "
        f"Average connection genes: 0.0 Average field genes: 4.0 "
        f"Best solution: [solution {sol_id} [ fit.: {fitness}, part.: ({parts_text}), spec.: 0, "
        f"adj.fit.: 0.0, age: 1, parents (0, 0), genome ( 4 field genes, 3 connection genes ) "
        f"field genes {{fg (id: 1, type: INPUT), }} connection genes {{}}, last mutations{{}}]]"
    )


def _stats_line(sol_id, fitness, partials):
    parts_text = ", ".join(str(p) for p in partials) + ", "
    return (
        f"solution {sol_id} [ fit.: {fitness}, part.: ({parts_text}), spec.: 0, adj.fit.: 0.0, "
        f"age: 1, parents (0, 0), genome ( 4 field genes, 3 connection genes ) "
        f"field genes {{fg (id: 1, type: INPUT), }} connection genes {{}}, last mutations{{}}]\n"
    )


def test_compute_partial_fitness_best_only_agrees_with_full_scan(tmp_path):
    run_dir = tmp_path / "run"
    stats_dir = run_dir / "statistics"
    stats_dir.mkdir(parents=True)

    overview_lines = [
        _overview_line(0, 1, 0.5, [0.5, 0.6]),
        _overview_line(1, 5, 0.9, [0.9, 0.8]),
    ]
    (run_dir / "per_generation_overview.txt").write_text("\n".join(overview_lines) + "\n")

    (stats_dir / "generation_1.txt").write_text(
        _stats_line(1, 0.5, [0.5, 0.6]) + _stats_line(2, 0.3, [0.3, 0.3])
    )
    (stats_dir / "generation_2.txt").write_text(
        _stats_line(5, 0.9, [0.9, 0.8]) + _stats_line(6, 0.7, [0.7, 0.6])
    )

    fast_df = compute_partial_fitness_best_only(str(run_dir), (0, 1))
    full_df = compute_partial_fitness(str(run_dir), (0, 1))

    assert list(fast_df["best_p1"]) == pytest.approx([0.5, 0.9])
    assert list(fast_df["best_p2"]) == pytest.approx([0.6, 0.8])

    # load-bearing equivalence: the fast path's best_p* must match the full scan's exactly,
    # since both are defined as "this generation's best-total-fitness individual's partials"
    assert list(fast_df["best_p1"]) == pytest.approx(list(full_df["best_p1"]))
    assert list(fast_df["best_p2"]) == pytest.approx(list(full_df["best_p2"]))

    # and the full scan alone provides avg_p*, which genuinely differs from best_p*
    assert "avg_p1" in full_df.columns
    assert list(full_df["avg_p1"]) == pytest.approx([0.4, 0.8])


def test_compute_partial_fitness_best_only_missing_overview_returns_empty(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    df = compute_partial_fitness_best_only(str(run_dir), (0, 1))
    assert df.empty
