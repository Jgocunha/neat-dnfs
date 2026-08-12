from viz.genome import build_genome_graph

# A minimal synthetic genome: field_in --(k1)--> field_out, plus a self-loop
# kernel on field_out, plus a stimulus element that only exists to exercise
# the exclude_stimuli flag.
ELEMENTS = [
    {"uniqueName": "field_in", "label": ["Neural field", "neural field"], "inputs": []},
    {"uniqueName": "k1", "label": ["Gaussian kernel", "gauss kernel"], "inputs": [["field_in", 0]]},
    {
        "uniqueName": "field_out",
        "label": ["Neural field", "neural field"],
        "inputs": [["k1", 0], ["k_self", 0]],
    },
    {"uniqueName": "k_self", "label": ["Gaussian kernel", "gauss kernel"], "inputs": [["field_out", 0]]},
    {"uniqueName": "stim1", "label": ["Gauss stimulus", "gauss stimulus"], "inputs": []},
]


def test_build_genome_graph_roles_interactions_and_self_loop():
    gg = build_genome_graph(ELEMENTS, exclude_stimuli=False)

    assert set(gg.field_nodes) == {"field_in", "field_out"}
    assert gg.field_roles["field_in"] == "Input"
    assert gg.field_roles["field_out"] == "Output"

    # only the field_in -> k1 -> field_out edge is a real interaction; k_self's
    # field_out -> k_self -> field_out loop must be excluded
    assert gg.interactions == [("k1", "field_in", "field_out")]
    assert gg.interaction_kernels == {"k1"}
    assert "k_self" not in gg.interaction_kernels

    assert gg.primary_kernel_by_field["field_out"] == "k1"
    assert "stim1" in gg.by_name


def test_build_genome_graph_exclude_stimuli_drops_stimulus_elements():
    gg = build_genome_graph(ELEMENTS, exclude_stimuli=True)
    assert "stim1" not in gg.by_name
    # dropping the stimulus doesn't change the field/kernel interaction graph
    assert set(gg.field_nodes) == {"field_in", "field_out"}
    assert gg.interactions == [("k1", "field_in", "field_out")]


def test_build_genome_graph_empty_input():
    assert build_genome_graph(None, exclude_stimuli=False).by_name == {}
    assert build_genome_graph([], exclude_stimuli=False).by_name == {}
