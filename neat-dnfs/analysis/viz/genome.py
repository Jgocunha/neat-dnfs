from collections import Counter
import pandas as pd
import networkx as nx
from dataclasses import dataclass, field

def get_element_label(el) -> str:
    if el is None:
        return ""
    lab = el.get("label", ["", ""])
    return lab[1] if isinstance(lab, list) and len(lab) > 1 else str(lab)


@dataclass
class GenomeGraph:
    by_name: dict = field(default_factory=dict)
    field_nodes: list = field(default_factory=list)
    kernel_nodes_all: list = field(default_factory=list)
    # (kernel, src_field, tgt_field) triples; self-loops (src == tgt) excluded.
    interactions: list = field(default_factory=list)
    interaction_kernels: set = field(default_factory=set)
    field_roles: dict = field(default_factory=dict)  # field -> "Input" / "Hidden" / "Output"
    # first kernel feeding each field, per its "inputs" order (may be a self-loop kernel).
    primary_kernel_by_field: dict = field(default_factory=dict)


def build_genome_graph(elements, *, exclude_stimuli: bool) -> GenomeGraph:
    """
    Walk a DNF-composer genome (a solution's "elements" list) once and derive
    everything the topology graph, kernel-usage stats, and genome tables need:
    field/kernel nodes, field-to-field interactions (kernels that mediate a
    connection between *different* fields; same-field self-loops are treated as
    part of that field's own dynamics, not an interaction), field roles inferred
    from that connectivity, and each field's primary (first) feeding kernel.

    exclude_stimuli=True drops "gauss stimulus"/"normal noise" elements entirely
    (used for the interaction-graph view); other callers keep them since a field's
    primary kernel can legitimately be a stimulus/noise source.
    """
    if elements is None:
        return GenomeGraph()

    by_name = {}
    for el in elements:
        name = el.get("uniqueName")
        if not name:
            continue
        if exclude_stimuli and get_element_label(el) in {"gauss stimulus", "normal noise"}:
            continue
        by_name[name] = el

    if not by_name:
        return GenomeGraph()

    field_nodes = [n for n, el in by_name.items() if get_element_label(el) == "neural field"]
    kernel_nodes_all = [n for n, el in by_name.items() if "kernel" in get_element_label(el)]

    inputs_by_target = {n: (el.get("inputs") or []) for n, el in by_name.items()}

    # --- determine which kernels actually mediate field-to-field interactions (no self-loops) ---
    interactions = []
    interaction_kernels = set()

    for k in kernel_nodes_all:
        k_el = by_name[k]
        src_fields = [src for src, _ in (k_el.get("inputs") or []) if src in field_nodes]

        tgt_fields = []
        for f in field_nodes:
            for src, _ in inputs_by_target.get(f, []):
                if src == k:
                    tgt_fields.append(f)
                    break

        for s in src_fields:
            for t in tgt_fields:
                if s == t:
                    # self-loop: belongs to field dynamics, not an interaction
                    continue
                interactions.append((k, s, t))
                interaction_kernels.add(k)

    # --- field connectivity graph (ignoring self-loops) -> Input/Hidden/Output ---
    field_graph = {f: set() for f in field_nodes}
    for _, s, t in interactions:
        field_graph[s].add(t)

    indeg = {f: 0 for f in field_nodes}
    outdeg = {f: 0 for f in field_nodes}
    for s, outs in field_graph.items():
        outdeg[s] += len(outs)
        for t in outs:
            indeg[t] += 1

    field_roles = {}
    for f in field_nodes:
        if indeg[f] == 0 and outdeg[f] > 0:
            field_roles[f] = "Input"
        elif indeg[f] > 0 and outdeg[f] == 0:
            field_roles[f] = "Output"
        else:
            field_roles[f] = "Hidden"

    # --- each field's primary (first) feeding kernel ---
    primary_kernel_by_field = {}
    for f in field_nodes:
        f_el = by_name[f]
        k_name = None
        for src, _ in f_el.get("inputs") or []:
            if src in kernel_nodes_all:
                k_name = src
                break
        primary_kernel_by_field[f] = k_name

    return GenomeGraph(
        by_name=by_name,
        field_nodes=field_nodes,
        kernel_nodes_all=kernel_nodes_all,
        interactions=interactions,
        interaction_kernels=interaction_kernels,
        field_roles=field_roles,
        primary_kernel_by_field=primary_kernel_by_field,
    )


def build_topology_graph(elements):
    """
    Build a clean left-to-right interaction graph for the best solution:

      - Nodes: neural fields + kernels that mediate interactions between
        *different* fields (no self-loops).
      - Excludes gauss stimulus and normal noise.
      - Excludes kernels that only form self-loops (nf -> nf).
      - Layout: Inputs on the left, Outputs on the right, Hidden in the middle.
        Kernels are positioned between the fields they connect.

    Returns:
      g          : networkx.DiGraph
      pos        : dict node -> (x, y)
      field_nodes: list of field node names
      kernel_nodes: list of kernel node names (that are in the graph)
    """
    g = nx.DiGraph()
    gg = build_genome_graph(elements, exclude_stimuli=True)
    if not gg.by_name:
        return g, {}, [], []

    kernel_nodes = sorted(gg.interaction_kernels)

    for f in gg.field_nodes:
        g.add_node(f, kind="field", role=gg.field_roles.get(f, ""))
    for k in kernel_nodes:
        g.add_node(k, kind="kernel")

    for k, s, t in gg.interactions:
        if s in g.nodes and k in g.nodes and t in g.nodes:
            g.add_edge(s, k)
            g.add_edge(k, t)

    # --- layout: left→right ---
    pos = {}

    def assign_layer(nodes, x):
        nodes = list(nodes)
        k = len(nodes)
        for i, n in enumerate(sorted(nodes)):
            y = 1.0 - (i + 1) / (k + 1) if k > 0 else 0.5
            pos[n] = (x, y)

    inputs_set = [f for f in gg.field_nodes if gg.field_roles.get(f) == "Input"]
    hidden_set = [f for f in gg.field_nodes if gg.field_roles.get(f) == "Hidden"]
    outputs_set = [f for f in gg.field_nodes if gg.field_roles.get(f) == "Output"]

    assign_layer(inputs_set, 0.1)
    assign_layer(hidden_set, 0.5)
    assign_layer(outputs_set, 0.9)

    # Kernels: between their fields (average x/y of connected fields)
    kernel_y_fallback = 0.5
    for idx, k in enumerate(kernel_nodes):
        connected_fields = []
        for u, v in g.in_edges(k):
            if u in gg.field_nodes and u in pos:
                connected_fields.append(u)
        for u, v in g.out_edges(k):
            if v in gg.field_nodes and v in pos:
                connected_fields.append(v)

        if connected_fields:
            xs = [pos[f][0] for f in connected_fields]
            ys = [pos[f][1] for f in connected_fields]
            x_k = sum(xs) / len(xs)
            y_k = sum(ys) / len(ys)
        else:
            x_k = 0.5
            y_k = 1.0 - (idx + 1) / (len(kernel_nodes) + 1) if kernel_nodes else kernel_y_fallback

        pos[k] = (x_k, y_k)

    return g, pos, gg.field_nodes, kernel_nodes


def classify_kernel_kind(el) -> str:
    """Return 'Gaussian', 'Mexican-hat', or 'Other' based on the label."""
    low = get_element_label(el).lower()
    if "mexican" in low:
        return "Mexican-hat"
    if "gauss" in low:
        return "Gaussian"
    return "Other"


def _kernel_param_str(k_el) -> str:
    lab = get_element_label(k_el)
    lab_lower = lab.lower()

    def num(*keys, default=None):
        for key in keys:
            val = k_el.get(key)
            if val is not None:
                return float(val)
        return default

    def fmt(val):
        return f"{val:.2f}" if val is not None else "n/a"

    if "mexican" in lab_lower:
        Ae = num("amplitudeExc")
        se = num("widthExc")
        Ai = num("amplitudeInh")
        si = num("widthInh")
        Ag = num("amplitudeGlobal", default=0.0)
        return (
            "Mexican-hat kernel: "
            f"A_exc = {fmt(Ae)}, σ_exc = {fmt(se)}, "
            f"A_inh = {fmt(Ai)}, σ_inh = {fmt(si)}, "
            f"A_glob = {fmt(Ag)}"
        )

    if "gauss" in lab_lower:
        A = num("amplitude", "amplitudeExc")
        s = num("width", "widthExc")
        Ag = num("amplitudeGlobal", default=0.0)
        return f"Gaussian kernel: A = {fmt(A)}, σ = {fmt(s)}, A_glob = {fmt(Ag)}"

    return lab


def compute_kernel_usage_stats(elements):
    """
    For a given best-solution genome (elements list) compute:

      - field_kernel_kinds: one entry per field that has a primary kernel
      - interaction_kernel_kinds: one entry per kernel that mediates
        a field-to-field interaction (no self-loops)

    Returns (field_counts, field_percent, inter_counts, inter_percent)
    where each *_counts is a dict(kind -> count)
          each *_percent is a dict(kind -> % of total)
    """
    gg = build_genome_graph(elements, exclude_stimuli=False)
    if not gg.by_name:
        return {}, {}, {}, {}

    field_kinds = []
    for f in gg.field_nodes:
        k_name = gg.primary_kernel_by_field.get(f)
        if k_name:
            field_kinds.append(classify_kernel_kind(gg.by_name[k_name]))

    inter_kinds = [classify_kernel_kind(gg.by_name[k]) for k in gg.interaction_kernels]

    def counts_and_perc(kinds):
        c = Counter(kinds)
        total = sum(c.values())
        if total == 0:
            return {}, {}
        perc = {k: 100.0 * v / total for k, v in c.items()}
        return dict(c), perc

    field_counts, field_perc = counts_and_perc(field_kinds)
    inter_counts, inter_perc = counts_and_perc(inter_kinds)
    return field_counts, field_perc, inter_counts, inter_perc


def collect_parameter_values(elements) -> dict:
    """Every numeric field/kernel parameter present in a genome's elements list, keyed by
    parameter name -- the raw material for population-level parameter-distribution charts.
    A neural field contributes tau/restingLevel; a Gaussian kernel contributes
    amplitude/width/amplitudeGlobal; a Mexican-hat kernel contributes
    amplitudeExc/widthExc/amplitudeInh/widthInh/amplitudeGlobal. Stimulus/noise elements have no
    field/kernel parameters of interest and are skipped.

    Returns dict[str, list[float]].
    """
    values: dict = {}

    def add(name, val):
        if val is not None:
            values.setdefault(name, []).append(float(val))

    for el in elements or []:
        label = get_element_label(el)
        if label == "neural field":
            add("tau", el.get("tau"))
            add("restingLevel", el.get("restingLevel"))
        elif label == "gauss kernel":
            add("amplitude", el.get("amplitude"))
            add("width", el.get("width"))
            add("amplitudeGlobal", el.get("amplitudeGlobal"))
        elif label == "mexican hat kernel":
            add("amplitudeExc", el.get("amplitudeExc"))
            add("widthExc", el.get("widthExc"))
            add("amplitudeInh", el.get("amplitudeInh"))
            add("widthInh", el.get("widthInh"))
            add("amplitudeGlobal", el.get("amplitudeGlobal"))

    return values


def kernel_kinds_for_solution(elements):
    """
    Helper used by compute_population_kernel_usage.

    For a single solution (its JSON 'elements' list), return:
      - field_kinds:  list with one entry per field's primary kernel
      - inter_kinds:  list with one entry per field–field interaction kernel

    We simply reuse compute_kernel_usage_stats and expand its counts
    into repeated labels, so the population-level code only has to
    aggregate simple lists.
    """
    field_counts, _, inter_counts, _ = compute_kernel_usage_stats(elements)

    field_kinds = []
    for kind, cnt in field_counts.items():
        field_kinds.extend([kind] * cnt)

    inter_kinds = []
    for kind, cnt in inter_counts.items():
        inter_kinds.extend([kind] * cnt)

    return field_kinds, inter_kinds


def summarize_best_solution_genome(elements):
    """
    Build two tables (as DataFrames) that describe the best solution's genome:

      - Field genes: one row per neural field
      - Interaction genes: one row per (kernel, from-field -> to-field) pair

    Field roles (Input / Hidden / Output) are inferred from connectivity:
      - Build a field->field graph via kernels, ignoring self-loops.
      - Input:    no incoming edges from other fields.
      - Output:   no outgoing edges to other fields.
      - Hidden:   everything else.

    Self-loop kernels (nf -> nf) are treated as part of that field's own
    dynamics and are NOT listed as separate interaction genes.
    """
    if elements is None:
        return None, None

    gg = build_genome_graph(elements, exclude_stimuli=False)
    if not gg.by_name:
        return None, None

    # ---------- Field genes table ----------
    field_rows = []
    for f in sorted(gg.field_nodes):
        f_el = gg.by_name[f]
        k_name = gg.primary_kernel_by_field.get(f)
        k_el = gg.by_name.get(k_name) if k_name else None

        role = gg.field_roles.get(f, "")
        kernel_type = get_element_label(k_el) if k_el else ""

        h = f_el.get("restingLevel")
        tau = f_el.get("tau")
        if h is not None and tau is not None:
            field_params = f"h = {h:.2f}, τ = {tau:.2f}"
        else:
            field_params = ""

        kernel_params = _kernel_param_str(k_el) if k_el else ""

        field_rows.append(
            {
                "Field": f,
                "Role": role,
                "Kernel type": kernel_type,
                "Field parameters": field_params,
                "Kernel parameters": kernel_params,
            }
        )

    df_fields = pd.DataFrame(field_rows)

    # ---------- Interaction genes table (exclude self-loops) ----------
    inter_rows = [
        {
            "Interaction gene": k,
            "From → To": f"{s} → {t}",
            "Kernel parameters": _kernel_param_str(gg.by_name[k]),
        }
        for k, s, t in gg.interactions
    ]

    df_inter = pd.DataFrame(inter_rows)
    return df_fields, df_inter
