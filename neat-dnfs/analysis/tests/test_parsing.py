import pandas as pd
import pytest

from viz.parsing import (
    _extract_mutation_events,
    _sample_evenly,
    categorize_mutation,
    generations_all_partial_meet_targets,
    generations_meeting_targets,
    parse_overview_line,
    parse_species_header,
)

# Real line from data/HRI Packaging Task C/.../per_generation_overview.txt
# (the "Best solution: [...]" tail is irrelevant to this parser and trimmed).
OVERVIEW_LINE = (
    "Current generation: 0 Number of solutions: 1000 Number of species: 66 "
    "Number of active species: 66 Has fitness improved: no "
    "Number of generations without improvement: 0 Average fitness: 0.213299 "
    "Best fitness: 0.794035 Innovation number: 42 Average genome size: 10.028000 "
    "Average connection genes: 3.570000 Average field genes: 6.458000 "
    "Best solution: [solution 110 [ fit.: 0.794035, part.: (0.925407, ...), ...]]"
)

# Real line from data/HRI Packaging Task C/.../species/generation_1.txt
# (the rep./champ. solution bodies are trimmed; only the header is parsed).
SPECIES_LINE = (
    "species 0 [ age: 1, extinct: no, improved: yes, gens. since imp.: 0  "
    "offs.: 0, mem: 638 rep.: {solution 913 [ fit.: 0.241126, ...]} "
    "champ.: {solution 110 [ fit.: 0.794035, ...]}]"
)


def test_parse_overview_line_real_sample():
    parsed = parse_overview_line(OVERVIEW_LINE)
    assert parsed == {
        "generation": 0,
        "num_solutions": 1000,
        "num_species": 66,
        "num_active_species": 66,
        "fitness_improved": False,
        "gens_without_improvement": 0,
        "avg_fitness": pytest.approx(0.213299),
        "best_fitness": pytest.approx(0.794035),
        "innovation_number": 42,
        "avg_genome_size": pytest.approx(10.028),
        "avg_conn_genes": pytest.approx(3.570),
        "avg_field_genes": pytest.approx(6.458),
    }


def test_parse_overview_line_unmatched_returns_none():
    assert parse_overview_line("not an overview line") is None


def test_parse_species_header_real_sample():
    parsed = parse_species_header(SPECIES_LINE)
    assert parsed["id"] == 0
    assert parsed["age"] == 1
    assert parsed["extinct"] is False
    assert parsed["improved"] is True
    assert parsed["gens_since_improvement"] == 0
    assert parsed["offspring"] == 0
    assert parsed["members"] == 638
    assert parsed["rep_raw"].startswith("{solution 913")
    assert parsed["champ_raw"].startswith("{solution 110")


def test_categorize_mutation_structural():
    assert categorize_mutation("toggle cg 5-4 enabled.").startswith("Structural")
    assert categorize_mutation("added fg 5").startswith("Structural")
    assert categorize_mutation("added cg 5-4").startswith("Structural")


def test_categorize_mutation_amp_glob_checked_before_amp():
    # "amp. glob" is a more specific match than "amp" and must be tried first,
    # or every global-amplitude mutation would be misclassified as a plain
    # amplitude mutation.
    assert (
        categorize_mutation("fg gk amp. glob.-0.100000", gene_type="fg")
        == "Field kernel – gk global amplitude"
    )
    assert (
        categorize_mutation("fg gk amp.-0.500000", gene_type="fg")
        == "Field kernel – gk amplitude"
    )
    assert (
        categorize_mutation("cg gk amp. glob.-0.100000", gene_type="cg")
        == "Connection kernel – gk global amplitude"
    )
    assert (
        categorize_mutation("cg gk amp.0.500000", gene_type="cg")
        == "Connection kernel – gk amplitude"
    )


def test_extract_mutation_events_real_sample():
    # Real "last mutations{...}" block (braces stripped, as the scanner does)
    # from data/HRI Packaging Task C/.../statistics/generation_20.txt.
    block = "[fg 3 (fg gk amp.-0.500000)] [fg 4 (fg gk width -0.250000)(fg gk amp.-0.500000)]  (mutated)"
    records = []
    _extract_mutation_events(5, 42, 0.75, block, records)

    assert [r["category"] for r in records] == [
        "Field kernel – gk amplitude",
        "Field kernel – gk width",
        "Field kernel – gk amplitude",
    ]
    assert [r["gene_ref"] for r in records] == ["3", "4", "4"]
    assert all(r["generation"] == 5 and r["solution_id"] == 42 and r["fitness"] == 0.75 for r in records)


def test_extract_mutation_events_structural_added_cg():
    block = "(added cg 6-4 innov.19)(added cg 3-5 innov.21)"
    records = []
    _extract_mutation_events(0, 110, 0.794035, block, records)
    assert len(records) == 2
    assert all(r["category"].startswith("Structural") for r in records)


def test_sample_evenly_no_op_when_under_limit():
    assert _sample_evenly([1, 2, 3], 5) == [1, 2, 3]


def test_sample_evenly_downsamples_preserving_order():
    items = list(range(10))
    result = _sample_evenly(items, 5)
    assert len(result) == 5
    assert result == sorted(result)
    assert result[0] == 0


def test_generations_meeting_targets_agrees_with_dataframe_variant():
    df = pd.DataFrame(
        {
            "generation": [0, 1, 2],
            "best_p1": [0.5, 0.9, 0.95],
            "best_p2": [0.4, 0.6, 0.92],
        }
    )
    targets = {1: 0.9, 2: 0.9}

    from_df = generations_all_partial_meet_targets(df, targets)

    generations = df["generation"].tolist()
    partial_vectors = [[row.best_p1, row.best_p2] for row in df.itertuples()]
    from_vectors = generations_meeting_targets(generations, partial_vectors, targets)

    assert from_df == from_vectors == [3]
