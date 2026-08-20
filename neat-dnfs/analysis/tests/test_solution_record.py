from viz.solution_record import find_solution_blob, parse_solution_blob, parse_solution_mutation_events

# Real line from data/.../statistics/generation_100.txt (current log format).
BLOB_NEW = (
    "solution 99001 [ fit.: 0.808721, part.: (0.917139, 0.920018, 0.803287, 0.858307, 0.920024, "
    "0.480993, 0.912083, 0.487273, 0.930490, 0.482627, 0.930475, 0.482627, ), spec.: 3626, "
    "adj.fit.: 0.004961, age: 1, parents (98347, 98190), genome ( 8 field genes, 20 connection genes ) "
    "field genes {fg (id: 1, type: INPUT), fg (id: 2, type: INPUT), fg (id: 3, type: INPUT), "
    "fg (id: 4, type: OUTPUT), fg (id: 5, type: HIDDEN), fg (id: 6, type: HIDDEN), "
    "fg (id: 7, type: HIDDEN), fg (id: 8, type: HIDDEN), } "
    "connection genes {cg (3-5, innov: 21, enabled: true), cg (1-4, innov: 4, enabled: true), "
    "cg (1-7, innov: 2252, enabled: true), }, "
    "last mutations{[fg 3 (fg gk amp.-0.500000)] [fg 4 (fg gk width -0.250000)(fg gk amp.-0.500000)]  (mutated)}]"
)

# Real line from an older run (data/Selection Instability/.../statistics/generation_10.txt) --
# no 'parents' label, no comma after 'part.: (...)'.
BLOB_OLD = (
    "solution 9001 [ fit.: 0.7445097975358732, part.: (0.9858036911120149, 0, 0.9941333779261721, "
    "0.9981021211053056, ) spec.: 0, adj.fit.: 0.0010868756168406909, age: 1, (8005, 8072), "
    "genome ( 2 field genes, 1 connection genes ) field genes {fg (id: 1, type: INPUT), "
    "fg (id: 2, type: OUTPUT), } connection genes {cg (1-2, innov: 0, enabled: false), }, "
    "last mutations{[cg 1-2 (cg gk amp. 0.5)]  (mutated)}]"
)

SPECIES_LINE = (
    "species 28 [ age: 50, extinct: no, improved: yes, gens. since imp.: 0  offs.: 4, mem: 78 "
    "rep.: {solution 49267 [ fit.: 0.687183, part.: (0.921063, 0.055000,), spec.: 28, "
    "adj.fit.: 0.008810, age: 1, parents (48317, 48317), genome ( 8 field genes, 19 connection genes ) "
    "field genes {fg (id: 1, type: INPUT), } connection genes {cg (3-5, innov: 21, enabled: true), }, "
    "last mutations{[fg 3 (fg gk amp.-0.5)] (mutated)}]} "
    "champ.: {solution 50123 [ fit.: 0.9, part.: (0.9,), spec.: 28, adj.fit.: 0.01, age: 2, "
    "parents (1,2), genome ( 8 field genes, 19 connection genes ) "
    "field genes {fg (id: 1, type: INPUT), } connection genes {cg (3-5, innov: 21, enabled: true), }, "
    "last mutations{}]}]"
)


def test_parse_solution_blob_current_format():
    r = parse_solution_blob(BLOB_NEW)
    assert r["id"] == 99001
    assert r["fitness"] == 0.808721
    assert len(r["partials"]) == 12
    assert r["species_id"] == 3626
    assert r["adjusted_fitness"] == 0.004961
    assert r["age"] == 1
    assert r["parent_ids"] == (98347, 98190)
    assert r["num_field_genes"] == 8
    assert r["num_connection_genes"] == 20
    assert len(r["field_genes"]) == 8
    assert r["field_genes"][4] == {"id": 5, "type": "HIDDEN"}
    assert len(r["connection_genes"]) == 3
    assert r["connection_genes"][0] == {"src": 3, "tgt": 5, "innov": 21, "enabled": True}
    assert "fg gk amp" in r["mutations_raw"]


def test_parse_solution_blob_older_format_no_parents_label():
    r = parse_solution_blob(BLOB_OLD)
    assert r["id"] == 9001
    assert r["parent_ids"] == (8005, 8072)
    assert r["connection_genes"] == [{"src": 1, "tgt": 2, "innov": 0, "enabled": False}]
    assert r["field_genes"] == [{"id": 1, "type": "INPUT"}, {"id": 2, "type": "OUTPUT"}]


def test_parse_solution_blob_unmatched_returns_none():
    assert parse_solution_blob("not a solution blob") is None


def test_find_solution_blob_handles_nested_brackets_in_mutation_block():
    blob = find_solution_blob(BLOB_NEW)
    assert blob is not None
    assert blob.startswith("solution 99001 [")
    assert blob.endswith("]")
    # the balanced scan must not stop at the first ']' inside the mutation block
    assert "(mutated)}]" in blob


def test_find_solution_blob_isolates_rep_from_champ_in_species_line():
    rep_blob = find_solution_blob(SPECIES_LINE)
    assert rep_blob is not None
    rep = parse_solution_blob(rep_blob)
    assert rep["id"] == 49267

    rep_end = SPECIES_LINE.find(rep_blob) + len(rep_blob)
    champ_blob = find_solution_blob(SPECIES_LINE, start_index=rep_end)
    assert champ_blob is not None
    champ = parse_solution_blob(champ_blob)
    assert champ["id"] == 50123


def test_find_solution_blob_returns_none_without_marker():
    assert find_solution_blob("no solution here") is None


def test_parse_solution_mutation_events_reuses_taxonomy():
    r = parse_solution_blob(BLOB_NEW)
    events = parse_solution_mutation_events(r, generation=5)
    assert len(events) == 3
    assert all(e["generation"] == 5 and e["solution_id"] == 99001 for e in events)
    assert [e["category"] for e in events] == [
        "Field kernel – gk amplitude",
        "Field kernel – gk width",
        "Field kernel – gk amplitude",
    ]


def test_parse_solution_mutation_events_empty_when_no_mutations():
    r = parse_solution_blob(
        "solution 1 [ fit.: 0.5, part.: (0.5,), spec.: 0, adj.fit.: 0.0, age: 1, parents (0, 0), "
        "genome ( 1 field genes, 0 connection genes ) field genes {fg (id: 1, type: INPUT), } "
        "connection genes {}, last mutations{}]"
    )
    assert parse_solution_mutation_events(r) == []
