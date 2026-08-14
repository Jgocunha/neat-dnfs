import pandas as pd
import pytest

from viz.parsing import (
    _extract_mutation_events,
    _sample_evenly,
    categorize_mutation,
    compute_partial_fitness,
    compute_population_distributions,
    compute_species_meta,
    compute_topology_trajectory,
    find_experiment_dirs,
    first_crossing_per_component,
    generations_all_partial_meet_targets,
    generations_meeting_targets,
    get_best_solution_id,
    list_champion_generations,
    parse_overview_line,
    parse_species_header,
    species_champion_fitness_trajectory,
    trace_lineage,
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


def test_compute_topology_trajectory_extracts_and_sorts_per_generation(tmp_path):
    gen0_line = (
        "Current generation: 0 Number of solutions: 1000 Number of species: 1 "
        "Number of active species: 1 Has fitness improved: no "
        "Number of generations without improvement: 0 Average fitness: 0.5 "
        "Best fitness: 0.5 Innovation number: 0 Average genome size: 4.0 "
        "Average connection genes: 0.0 Average field genes: 4.0 "
        "Best solution: [solution 1 [ fit.: 0.5, part.: (0.5,), spec.: 0, adj.fit.: 0.0, "
        "age: 1, parents (0, 0), genome ( 4 field genes, 3 connection genes ) "
        "field genes {fg (id: 1, type: INPUT), fg (id: 2, type: INPUT), "
        "fg (id: 3, type: INPUT), fg (id: 4, type: OUTPUT), } "
        "connection genes {cg (1-4, innov: 0, enabled: true), cg (2-4, innov: 1, enabled: true), "
        "cg (3-4, innov: 2, enabled: true), }, last mutations{}]]"
    )
    gen1_line = (
        "Current generation: 1 Number of solutions: 1000 Number of species: 1 "
        "Number of active species: 1 Has fitness improved: yes "
        "Number of generations without improvement: 0 Average fitness: 0.6 "
        "Best fitness: 0.6 Innovation number: 4 Average genome size: 5.0 "
        "Average connection genes: 1.0 Average field genes: 5.0 "
        "Best solution: [solution 2 [ fit.: 0.6, part.: (0.6,), spec.: 0, adj.fit.: 0.0, "
        "age: 1, parents (1, 1), genome ( 5 field genes, 4 connection genes ) "
        "field genes {fg (id: 1, type: INPUT), fg (id: 2, type: INPUT), "
        "fg (id: 3, type: INPUT), fg (id: 4, type: OUTPUT), fg (id: 5, type: HIDDEN), } "
        "connection genes {cg (1-4, innov: 0, enabled: true), cg (2-4, innov: 1, enabled: false), "
        "cg (3-5, innov: 3, enabled: true), cg (5-4, innov: 4, enabled: true), }, "
        "last mutations{}]]"
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # written out of order to verify the function sorts by generation
    (run_dir / "per_generation_overview.txt").write_text(gen1_line + "\n" + gen0_line + "\n")

    df = compute_topology_trajectory(str(run_dir))
    assert list(df["generation"]) == [0, 1]
    assert list(df["hidden_fields"]) == [0, 1]
    assert list(df["enabled_connections"]) == [3, 3]
    assert list(df["disabled_connections"]) == [0, 1]


def _write_solution_line(path, sol_id, fitness, age, parents, field_genes, connection_genes):
    fg_text = "".join(f"fg (id: {i}, type: {t}), " for i, t in field_genes)
    cg_text = "".join(
        f"cg ({s}-{t}, innov: {innov}, enabled: {'true' if en else 'false'}), "
        for s, t, innov, en in connection_genes
    )
    line = (
        f"solution {sol_id} [ fit.: {fitness}, part.: ({fitness},), spec.: 0, adj.fit.: 0.0, "
        f"age: {age}, parents ({parents[0]}, {parents[1]}), "
        f"genome ( {len(field_genes)} field genes, {len(connection_genes)} connection genes ) "
        f"field genes {{{fg_text}}} connection genes {{{cg_text}}}, last mutations{{}}]\n"
    )
    with open(path, "a") as f:
        f.write(line)


def _overview_line(gen0, sol_id, fitness):
    return (
        f"Current generation: {gen0} Number of solutions: 10 Number of species: 1 "
        f"Number of active species: 1 Has fitness improved: yes "
        f"Number of generations without improvement: 0 Average fitness: {fitness} "
        f"Best fitness: {fitness} Innovation number: 0 Average genome size: 4.0 "
        f"Average connection genes: 0.0 Average field genes: 4.0 "
        f"Best solution: [solution {sol_id} [ fit.: {fitness}, part.: ({fitness},), spec.: 0, "
        f"adj.fit.: 0.0, age: 1, parents (0, 0), genome ( 4 field genes, 3 connection genes ) "
        f"field genes {{fg (id: 1, type: INPUT), }} connection genes {{}}, last mutations{{}}]]"
    )


def test_trace_lineage_walks_parents_back_to_bootstrap_root(tmp_path):
    run_dir = tmp_path / "run"
    stats_dir = run_dir / "statistics"
    stats_dir.mkdir(parents=True)

    fields4 = [(1, "INPUT"), (2, "INPUT"), (3, "INPUT"), (4, "OUTPUT")]
    conns3 = [(1, 4, 0, True), (2, 4, 1, True), (3, 4, 2, True)]
    fields5 = fields4 + [(5, "HIDDEN")]
    conns4 = conns3 + [(5, 4, 4, True)]

    _write_solution_line(stats_dir / "generation_1.txt", 1, 0.5, 1, (0, 0), fields4, conns3)
    _write_solution_line(stats_dir / "generation_1.txt", 2, 0.4, 1, (0, 0), fields4, conns3)
    _write_solution_line(stats_dir / "generation_2.txt", 5, 0.6, 2, (1, 2), fields4, conns3)
    _write_solution_line(stats_dir / "generation_3.txt", 9, 0.7, 3, (5, 6), fields5, conns4)

    chain = trace_lineage(str(run_dir), start_generation0=2, start_solution_id=9)

    assert [c["generation"] for c in chain] == [0, 1, 2]
    assert [c["record"]["id"] for c in chain] == [1, 5, 9]
    assert chain[0]["record"]["parent_ids"] == (0, 0)
    assert chain[-1]["record"]["num_field_genes"] == 5


def test_trace_lineage_missing_parent_stops_gracefully(tmp_path):
    run_dir = tmp_path / "run"
    stats_dir = run_dir / "statistics"
    stats_dir.mkdir(parents=True)

    fields4 = [(1, "INPUT"), (2, "INPUT"), (3, "INPUT"), (4, "OUTPUT")]
    conns3 = [(1, 4, 0, True), (2, 4, 1, True), (3, 4, 2, True)]
    # parent 999 does not exist anywhere -- chain should stop at generation 1, not crash
    _write_solution_line(stats_dir / "generation_2.txt", 5, 0.6, 2, (999, 2), fields4, conns3)

    chain = trace_lineage(str(run_dir), start_generation0=1, start_solution_id=5)
    assert [c["record"]["id"] for c in chain] == [5]


def test_get_best_solution_id_reads_overview_blob(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lines = [_overview_line(0, 42, 0.5), _overview_line(1, 43, 0.6)]
    (run_dir / "per_generation_overview.txt").write_text("\n".join(lines) + "\n")

    assert get_best_solution_id(str(run_dir), 0) == 42
    assert get_best_solution_id(str(run_dir), 1) == 43
    assert get_best_solution_id(str(run_dir), 5) is None


def test_compute_population_distributions_and_partial_fitness_share_one_scan(tmp_path):
    run_dir = tmp_path / "run"
    stats_dir = run_dir / "statistics"
    stats_dir.mkdir(parents=True)

    fields4 = [(1, "INPUT"), (2, "INPUT"), (3, "INPUT"), (4, "OUTPUT")]
    conns3 = [(1, 4, 0, True), (2, 4, 1, True), (3, 4, 2, True)]
    fields5 = fields4 + [(5, "HIDDEN")]

    _write_solution_line(stats_dir / "generation_1.txt", 1, 0.5, 3, (0, 0), fields4, conns3)
    _write_solution_line(stats_dir / "generation_1.txt", 2, 0.7, 5, (0, 0), fields5, conns3)

    dist_df = compute_population_distributions(str(run_dir), (0,))
    assert len(dist_df) == 2
    assert sorted(dist_df["age"].tolist()) == [3, 5]
    assert sorted(dist_df["genome_size"].tolist()) == [7, 8]  # 4+3 and 5+3
    assert set(dist_df["fitness"].round(1)) == {0.5, 0.7}

    # the shared single-pass scan must still produce correct best_p* for the same generations
    partial_df = compute_partial_fitness(str(run_dir), (0,))
    assert partial_df is not None
    assert partial_df.loc[0, "best_p1"] == pytest.approx(0.7)


def _species_line(sid, members, champ_id, champ_fitness, extinct="no"):
    champ_blob = (
        f"solution {champ_id} [ fit.: {champ_fitness}, part.: ({champ_fitness},), spec.: {sid}, "
        f"adj.fit.: 0.0, age: 1, parents (0, 0), genome ( 4 field genes, 3 connection genes ) "
        f"field genes {{fg (id: 1, type: INPUT), }} connection genes {{}}, last mutations{{}}]"
    )
    return (
        f"species {sid} [ age: 1, extinct: {extinct}, improved: yes, gens. since imp.: 0  "
        f"offs.: 0, mem: {members} rep.: {{none}} champ.: {{{champ_blob}}}]"
    )


def test_compute_species_meta_tracks_membership_and_champion_per_generation(tmp_path):
    run_dir = tmp_path / "run"
    species_dir = run_dir / "species"
    species_dir.mkdir(parents=True)

    (species_dir / "generation_1.txt").write_text(_species_line(0, 10, 100, 0.5) + "\n")
    (species_dir / "generation_2.txt").write_text(_species_line(0, 20, 101, 0.6) + "\n")

    meta = compute_species_meta(str(run_dir), (0, 1))
    assert 0 in meta
    assert meta[0]["members_by_gen"] == {0: 10, 1: 20}
    assert meta[0]["first_gen"] == 0
    assert meta[0]["last_gen"] == 1
    assert meta[0]["max_members"] == 20

    traj = species_champion_fitness_trajectory(str(run_dir), (0, 1), 0)
    assert list(traj["generation"]) == [0, 1]
    assert list(traj["fitness"]) == [0.5, 0.6]


def test_species_champion_fitness_trajectory_unknown_species_returns_empty(tmp_path):
    run_dir = tmp_path / "run"
    species_dir = run_dir / "species"
    species_dir.mkdir(parents=True)
    (species_dir / "generation_1.txt").write_text(_species_line(0, 10, 100, 0.5) + "\n")

    traj = species_champion_fitness_trajectory(str(run_dir), (0,), 999)
    assert traj.empty


def test_find_experiment_dirs_only_returns_folders_with_runs(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()

    # a real experiment: a folder whose child has per_generation_overview.txt
    exp_a = data_root / "Experiment A" / "2026-01-01 00h00m00s"
    exp_a.mkdir(parents=True)
    (exp_a / "per_generation_overview.txt").write_text(_overview_line(0, 1, 0.5))

    # not an experiment: no run subfolder has the overview file
    (data_root / "Not An Experiment").mkdir()
    (data_root / "Not An Experiment" / "some_dir").mkdir()

    # a stray file directly under data_root, not a directory -- must be skipped
    (data_root / "readme.txt").write_text("hello")

    found = find_experiment_dirs(data_root)
    assert [name for name, _ in found] == ["Experiment A"]


def test_find_experiment_dirs_missing_root_returns_empty(tmp_path):
    assert find_experiment_dirs(tmp_path / "does_not_exist") == []


def test_first_crossing_per_component_tracks_each_component_independently():
    # same fixture as the generations_meeting_targets/dataframe-variant agreement test, but
    # here each component's OWN first-crossing generation matters, not the joint one
    generations = [0, 1, 2]
    partial_vectors = [[0.5, 0.4], [0.9, 0.6], [0.95, 0.92]]
    targets = {1: 0.9, 2: 0.9}

    crossing = first_crossing_per_component(generations, partial_vectors, targets)
    assert crossing == {1: 2, 2: 3}  # component 1 crosses at gen1 (1-based 2), component 2 at gen2 (1-based 3)


def test_first_crossing_per_component_never_crossing_is_none():
    generations = [0, 1, 2]
    partial_vectors = [[0.1], [0.2], [0.3]]
    targets = {1: 0.9}

    crossing = first_crossing_per_component(generations, partial_vectors, targets)
    assert crossing == {1: None}


def test_list_champion_generations_filters_by_species_and_converts_to_0_based(tmp_path):
    run_dir = tmp_path / "run"
    champ_dir = run_dir / "champions" / "prev_generations"
    champ_dir.mkdir(parents=True)

    (champ_dir / "solution 100 generation 1 species 1 fitness 0.5").mkdir()
    (champ_dir / "solution 200 generation 2 species 1 fitness 0.6").mkdir()
    # digit-prefix collision case: "species 14" must not match species_id=1 via substring match
    (champ_dir / "solution 300 generation 1 species 14 fitness 0.9").mkdir()

    result = list_champion_generations(str(run_dir), 1)
    gens = [g for g, _ in result]
    assert gens == [0, 1]  # 1-based generation 1,2 -> 0-based 0,1


def test_list_champion_generations_no_directory_returns_empty(tmp_path):
    assert list_champion_generations(str(tmp_path / "run"), 0) == []


def test_list_champion_generations_picks_highest_fitness_on_duplicate(tmp_path):
    run_dir = tmp_path / "run"
    champ_dir = run_dir / "champions" / "prev_generations"
    champ_dir.mkdir(parents=True)
    (champ_dir / "solution 100 generation 1 species 0 fitness 0.3").mkdir()
    (champ_dir / "solution 101 generation 1 species 0 fitness 0.8").mkdir()

    result = list_champion_generations(str(run_dir), 0)
    assert len(result) == 1
    gen0, path = result[0]
    assert gen0 == 0
    assert "fitness 0.8" in path.name


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
