#!/usr/bin/env python3
from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import networkx as nx  # for topology graphs

# =========================
# Matplotlib global style
# =========================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Garamond", "Times New Roman", "DejaVu Serif"]
plt.rcParams["figure.dpi"] = 150


# =========================
# Parsing helpers: overview
# =========================

def parse_overview_line(line: str):
    pattern = (
        r"Current generation:\s*(?P<generation>\d+)"
        r".*?Number of solutions:\s*(?P<num_solutions>\d+)"
        r".*?Number of species:\s*(?P<num_species>\d+)"
        r".*?Number of active species:\s*(?P<num_active_species>\d+)"
        r".*?Has fitness improved:\s*(?P<fitness_improved>yes|no)"
        r".*?Number of generations without improvement:\s*(?P<gens_without_improvement>\d+)"
        r".*?Average fitness:\s*(?P<avg_fitness>[0-9eE\.\+\-]+)"
        r".*?Best fitness:\s*(?P<best_fitness>[0-9eE\.\+\-]+)"
        r".*?Innovation number:\s*(?P<innovation_number>\d+)"
        r".*?Average genome size:\s*(?P<avg_genome_size>[0-9eE\.\+\-]+)"
        r".*?Average connection genes:\s*(?P<avg_conn_genes>[0-9eE\.\+\-]+)"
        r".*?Average field genes:\s*(?P<avg_field_genes>[0-9eE\.\+\-]+)"
    )
    m = re.search(pattern, line)
    if not m:
        return None

    g = m.groupdict()

    def to_int(name):
        return int(g[name])

    def to_float(name):
        return float(g[name])

    return {
        "generation": to_int("generation"),
        "num_solutions": to_int("num_solutions"),
        "num_species": to_int("num_species"),
        "num_active_species": to_int("num_active_species"),
        "fitness_improved": g["fitness_improved"] == "yes",
        "gens_without_improvement": to_int("gens_without_improvement"),
        "avg_fitness": to_float("avg_fitness"),
        "best_fitness": to_float("best_fitness"),
        "innovation_number": to_int("innovation_number"),
        "avg_genome_size": to_float("avg_genome_size"),
        "avg_conn_genes": to_float("avg_conn_genes"),
        "avg_field_genes": to_float("avg_field_genes"),
    }


@st.cache_data
def load_overview(run_dir_str: str) -> pd.DataFrame:
    run_dir = Path(run_dir_str)
    overview_path = run_dir / "per_generation_overview.txt"
    if not overview_path.exists():
        raise FileNotFoundError(f"Could not find {overview_path}")

    rows = []
    with overview_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_overview_line(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise RuntimeError(f"No parsable lines in {overview_path}")

    df = pd.DataFrame(rows)
    df.sort_values("generation", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def find_runs_with_overview(base_dir: Path):
    runs = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            overview = child / "per_generation_overview.txt"
            if overview.exists():
                runs.append((child.name, child))
    return runs


# =========================
# Parsing helpers: partial fitness (statistics/)
# =========================

@st.cache_data
def compute_partial_fitness(run_dir_str: str, generations: tuple):
    """
    For each generation, read statistics/generation_X.txt and compute:
      - best partial fitness vector (from best total fitness individual)
      - average partial fitness over all individuals
    Returns a DataFrame with columns:
      generation, best_p1..N, avg_p1..N
    """
    run_dir = Path(run_dir_str)
    stats_dir = run_dir / "statistics"
    if not stats_dir.exists():
        return None

    records = []
    for g in generations:
        stats_path = stats_dir / f"generation_{g}.txt"
        if not stats_path.exists():
            continue

        best_fit = -1e9
        best_parts = None
        sum_parts = None
        count = 0

        with stats_path.open("r") as f:
            for line in f:
                m = re.search(
                    r"fit\.\:\s*([0-9eE\.\+\-]+).*?part\.\:\s*\(([^)]*)\)",
                    line,
                )
                if not m:
                    continue

                fit = float(m.group(1))
                parts_str = m.group(2)
                parts = []
                for token in parts_str.split(","):
                    token = token.strip()
                    if token:
                        try:
                            parts.append(float(token))
                        except ValueError:
                            pass

                if not parts:
                    continue

                if sum_parts is None:
                    sum_parts = [0.0] * len(parts)

                for i, v in enumerate(parts):
                    sum_parts[i] += v

                count += 1

                if fit > best_fit:
                    best_fit = fit
                    best_parts = parts

        if count == 0 or best_parts is None:
            continue

        avg_parts = [s / count for s in sum_parts]
        rec = {"generation": g}
        for i, v in enumerate(best_parts, start=1):
            rec[f"best_p{i}"] = v
        for i, v in enumerate(avg_parts, start=1):
            rec[f"avg_p{i}"] = v

        records.append(rec)

    if not records:
        return None

    df = pd.DataFrame(records)
    df.sort_values("generation", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# =========================
# Parsing helpers: species/ files
# =========================

def parse_species_header(line: str):
    """
    Parse one 'species ...' line from species/generation_X.txt.
    """
    header_pattern = (
        r"species\s+(?P<id>\d+)\s+\["
        r"\s*age:\s*(?P<age>\d+),\s*extinct:\s*(?P<extinct>yes|no),"
        r"\s*improved:\s*(?P<improved>yes|no),\s*gens\. since imp\.\:\s*(?P<since>\d+)"
        r"\s*offs\.\:\s*(?P<offs>\d+),\s*mem:\s*(?P<mem>\d+)"
    )
    m = re.search(header_pattern, line)
    if not m:
        return None

    g = m.groupdict()
    tail = line[m.end():].strip()

    rep_text = ""
    champ_text = ""

    rep_idx = tail.find("rep.:")
    champ_idx = tail.find("champ.:")
    if rep_idx != -1 and champ_idx != -1:
        rep_text = tail[rep_idx + len("rep.:"):champ_idx].strip()
        champ_text = tail[champ_idx + len("champ.:"):].strip()
    elif rep_idx != -1:
        rep_text = tail[rep_idx + len("rep.:"):].strip()
    elif champ_idx != -1:
        champ_text = tail[champ_idx + len("champ.:"):].strip()

    return {
        "id": int(g["id"]),
        "age": int(g["age"]),
        "extinct": g["extinct"] == "yes",
        "improved": g["improved"] == "yes",
        "gens_since_improvement": int(g["since"]),
        "offspring": int(g["offs"]),
        "members": int(g["mem"]),
        "rep_raw": rep_text,
        "champ_raw": champ_text,
    }


@st.cache_data
def compute_species_meta(run_dir_str: str, generations: tuple):
    run_dir = Path(run_dir_str)
    species_dir = run_dir / "species"
    if not species_dir.exists():
        return {}

    meta = {}
    gens_sorted = sorted(generations)
    final_gen = gens_sorted[-1] if gens_sorted else None

    for g in gens_sorted:
        path = species_dir / f"generation_{g}.txt"
        if not path.exists():
            continue
        with path.open("r") as f:
            for line in f:
                if "species" not in line:
                    continue
                parsed = parse_species_header(line)
                if not parsed:
                    continue
                sid = parsed["id"]
                m = meta.setdefault(
                    sid,
                    {
                        "first_gen": g,
                        "last_gen": g,
                        "max_members": parsed["members"],
                        "total_offspring": parsed["offspring"],
                        "last_extinct": parsed["extinct"],
                    },
                )
                m["first_gen"] = min(m["first_gen"], g)
                m["last_gen"] = max(m["last_gen"], g)
                m["max_members"] = max(m["max_members"], parsed["members"])
                m["total_offspring"] += parsed["offspring"]
                if g == final_gen:
                    m["last_extinct"] = parsed["extinct"]

    return meta


@st.cache_data
def get_species_for_generation(run_dir_str: str, generation: int):
    run_dir = Path(run_dir_str)
    species_dir = run_dir / "species"
    path = species_dir / f"generation_{generation}.txt"
    if not path.exists():
        return []

    result = []
    with path.open("r") as f:
        for line in f:
            if "species" not in line:
                continue
            parsed = parse_species_header(line)
            if parsed:
                result.append(parsed)
    result.sort(key=lambda d: d["id"])
    return result


# =========================
# Parsing helpers: best_solutions JSON (topology)
# =========================

@st.cache_data
def load_best_solution_architecture(run_dir_str: str, generation: int):
    """
    Load the JSON architecture for the best solution of a given generation
    from best_solutions/prev_generations.
    Returns list of element dicts (DNF composer JSON) or None.
    """
    run_dir = Path(run_dir_str)
    bs_dir = run_dir / "best_solutions" / "prev_generations"
    if not bs_dir.exists():
        return None

    pattern = f"solution * generation {generation} *.json"
    candidates = list(bs_dir.glob(pattern))
    if not candidates:
        return None

    def fitness_from_name(p: Path):
        # Extract numeric fitness before ".json"
        m = re.search(r"fitness ([0-9eE+\-\.]+)\.json", p.name)
        if not m:
            return -1e9
        return float(m.group(1))

    best_path = max(candidates, key=fitness_from_name)

    import json
    with best_path.open("r") as f:
        data = json.load(f)
    return data

def build_topology_graph(elements):
    """
    Simple version:
      - show all elements (stimuli, noise, kernels, fields, etc.)
      - use a generic graph with spring layout (no left/right classification)
    """
    g = nx.DiGraph()
    if elements is None:
        return g, {}

    # Add nodes
    for el in elements:
        name = el.get("uniqueName")
        if not name:
            continue
        label = el.get("label", ["", ""])
        label_text = label[1] if isinstance(label, list) and len(label) > 1 else str(label)
        node_label = f"{name}\n{label_text}"
        g.add_node(name, label=node_label)

    # Add edges from inputs
    for el in elements:
        target = el.get("uniqueName")
        if not target:
            continue
        inputs = el.get("inputs") or []  # handle null -> []
        for src, port in inputs:
            if src in g.nodes:
                g.add_edge(src, target)

    labels = {n: d.get("label", n) for n, d in g.nodes(data=True)}
    return g, labels



def plot_topology_graph(g, labels):
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(g.nodes) == 0:
        ax.text(0.5, 0.5, "No topology data available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    pos = nx.spring_layout(g, seed=42)
    nx.draw_networkx(g, pos=pos, labels=labels, ax=ax, font_size=8)
    ax.axis("off")
    fig.tight_layout()
    return fig





# =========================
# Plotting helpers
# =========================

def plot_total_fitness(df: pd.DataFrame, target_fitness: float):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df["generation"], df["avg_fitness"], label="avg. fitness")
    ax.plot(df["generation"], df["best_fitness"], label="best fitness")
    ax.axhline(target_fitness, linestyle="--", label=f"target ({target_fitness:.3f})")

    reached = df[df["best_fitness"] >= target_fitness]
    if not reached.empty:
        row = reached.iloc[0]
        ax.scatter(row["generation"], row["best_fitness"], marker="o", zorder=5, label="target reached")

    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_partial_fitness_grid(partial_df: pd.DataFrame, partial_targets: dict):
    if partial_df is None or partial_df.empty:
        st.info("No partial fitness statistics found (statistics/generation_X.txt missing?).")
        return

    num_partial = 0
    for col in partial_df.columns:
        if col.startswith("best_p"):
            idx = int(col.replace("best_p", ""))
            num_partial = max(num_partial, idx)

    if num_partial == 0:
        st.info("No partial fitness columns could be parsed.")
        return

    st.markdown("#### Partial fitness components")

    per_row = 4
    rows = math.ceil(num_partial / per_row)
    gen = partial_df["generation"]

    comp_index = 1
    for _ in range(rows):
        cols = st.columns(per_row)
        for col in cols:
            if comp_index > num_partial:
                break

            best_col = f"best_p{comp_index}"
            avg_col = f"avg_p{comp_index}"
            if best_col not in partial_df.columns or avg_col not in partial_df.columns:
                comp_index += 1
                continue

            target = partial_targets.get(comp_index)

            with col:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(gen, partial_df[avg_col], label="avg.")
                ax.plot(gen, partial_df[best_col], label="best")
                ax.axhline(target, linestyle="--", label=f"target ({target:.3f})")

                reached = partial_df[partial_df[best_col] >= target]
                if not reached.empty:
                    row = reached.iloc[0]
                    ax.scatter(row["generation"], row[best_col], marker="o", zorder=5, label="target reached")

                ax.set_xlabel("generation")
                ax.set_ylabel("fitness")
                ax.set_title(f"partial fitness {comp_index}")
                ax.legend(fontsize="x-small")
                ax.grid(True)
                fig.tight_layout()
                st.pyplot(fig)

            comp_index += 1


def plot_species_counts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["generation"], df["num_species"], label="species")
    ax.plot(df["generation"], df["num_active_species"], label="active species")
    ax.set_xlabel("generation")
    ax.set_ylabel("count")
    ax.set_title("Species count evolution")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_innovation_growth(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["generation"], df["innovation_number"], label="innovation number")
    ax.set_xlabel("generation")
    ax.set_ylabel("innovation number")
    ax.set_title("Innovation numbers growth")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_genome_topology_curves(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["generation"], df["avg_genome_size"], label="avg genome size")
    ax.plot(df["generation"], df["avg_field_genes"], label="avg field genes")
    ax.plot(df["generation"], df["avg_conn_genes"], label="avg connection genes")
    ax.set_xlabel("generation")
    ax.set_ylabel("genes")
    ax.set_title("Genome topology")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


# =========================
# Statistics helpers
# =========================

def render_fitness_stats(df: pd.DataFrame, target_fitness: float):
    final_row = df.iloc[-1]
    final_gen = int(final_row["generation"])
    best_final = final_row["best_fitness"]
    avg_final = final_row["avg_fitness"]

    max_best = df["best_fitness"].max()
    gen_max_best = int(df.loc[df["best_fitness"].idxmax(), "generation"])

    reached = df[df["best_fitness"] >= target_fitness]
    if not reached.empty:
        gen_target = int(reached["generation"].iloc[0])
        best_at_target = reached["best_fitness"].iloc[0]
    else:
        gen_target = None
        best_at_target = None

    best_series = df["best_fitness"]
    improved = best_series.diff().fillna(0) > 1e-9
    longest_stagnation = 0
    current = 0
    for imp in improved[1:]:
        if imp:
            longest_stagnation = max(longest_stagnation, current)
            current = 0
        else:
            current += 1
    longest_stagnation = max(longest_stagnation, current)

    auc_best = best_series.mean()
    auc_avg = df["avg_fitness"].mean()

    st.markdown("#### Statistics")
    st.markdown(
        f"""
        **Final generation (g = {final_gen})**  
        • Best fitness: **{best_final:.4f}**  
        • Average fitness: **{avg_final:.4f}**  

        **Overall**  
        • Max best fitness: **{max_best:.4f}** (reached at generation {gen_max_best})  
        • Mean best fitness over run (AUC): **{auc_best:.4f}**  
        • Mean average fitness over run (AUC): **{auc_avg:.4f}**  
        • Longest stagnation period (no improvement in best fitness): **{longest_stagnation} generations**  
        """
    )

    if gen_target is not None:
        st.markdown(
            f"• Target fitness **{target_fitness:.3f}** first reached at generation **{gen_target}** "
            f"(best fitness ≈ **{best_at_target:.4f}**)."
        )
    else:
        st.markdown(
            f"• Target fitness **{target_fitness:.3f}** was **not reached** by the best fitness."
        )


def render_species_stats(df: pd.DataFrame, species_meta: dict):
    last = df.iloc[-1]
    final_gen = int(last["generation"])
    final_species = int(last["num_species"])
    final_active = int(last["num_active_species"])

    avg_species = df["num_species"].mean()
    avg_active = df["num_active_species"].mean()
    max_species = df["num_species"].max()
    gen_max_species = int(df.loc[df["num_species"].idxmax(), "generation"])

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
        if m["last_gen"] == final_gen and not m["last_extinct"]:
            active_final += 1

    extinct_species = total_species - active_final

    if lifespans:
        avg_lifespan = sum(lifespans) / len(lifespans)
        max_lifespan = max(lifespans)
        max_life_sid = [
            sid for sid, m in species_meta.items()
            if m["last_gen"] - m["first_gen"] + 1 == max_lifespan
        ][0]
    else:
        avg_lifespan = 0.0
        max_lifespan = 0
        max_life_sid = None

    avg_max_members = sum(max_members_list) / len(max_members_list) if max_members_list else 0.0
    avg_offspring = sum(offspring_list) / len(offspring_list) if offspring_list else 0.0

    st.markdown("#### Species statistics")
    st.markdown(
        f"""
        **Final generation (g = {final_gen})**  
        • Species: **{final_species}**  
        • Active species: **{final_active}**  

        **Across run**  
        • Total distinct species ever created: **{total_species}**  
        • Species that went extinct by final generation: **{extinct_species}**  
        • Average number of species per generation: **{avg_species:.2f}**  
        • Average number of active species per generation: **{avg_active:.2f}**  
        • Max number of species in a generation: **{max_species}** (at generation {gen_max_species})  
        """
    )

    st.markdown(
        f"""
        **Species lifetime & size**  
        • Average species lifespan: **{avg_lifespan:.2f}** generations  
        • Longest-lived species: **{max_life_sid}** (lifespan {max_lifespan} generations)  
        • Average max members per species: **{avg_max_members:.2f}**  
        • Average total offspring assigned per species: **{avg_offspring:.2f}**  
        """
    )


def render_topology_stats(df: pd.DataFrame):
    first = df.iloc[0]
    last = df.iloc[-1]

    g0 = first["avg_genome_size"]
    gN = last["avg_genome_size"]
    f0 = first["avg_field_genes"]
    fN = last["avg_field_genes"]
    c0 = first["avg_conn_genes"]
    cN = last["avg_conn_genes"]

    gens = last["generation"] - first["generation"]
    gens = gens if gens > 0 else 1

    st.markdown("#### Topology statistics")
    st.markdown(
        f"""
        **Final generation (g = {int(last['generation'])})**  
        • Avg genome size: **{gN:.2f}**  
        • Avg field genes: **{fN:.2f}**  
        • Avg connection genes: **{cN:.2f}**  

        **Growth over run**  
        • Genome size change: **{gN - g0:+.2f}** (≈ {(gN - g0)/gens:+.3f} per generation)  
        • Field genes change: **{fN - f0:+.2f}** (≈ {(fN - f0)/gens:+.3f} per generation)  
        • Connection genes change: **{cN - c0:+.2f}** (≈ {(cN - c0)/gens:+.3f} per generation)  

        **Ratios**  
        • Avg connections per field at final gen: **{(cN / fN) if fN > 0 else 0.0:.2f}**  
        • Avg genome size / population size is accessible from per-generation statistics if needed.  
        """
    )

# =========================
# Mutation helpers
# =========================

def categorize_mutation(mut_str: str) -> str:
    """
    Heuristic categorisation of a mutation description string.
    Adjust these rules as you refine your logging format.
    """
    s = mut_str.lower()

    # ---- structural: field / connection changes ----
    if "add fg" in s or "added fg" in s or "new fg" in s or "add field" in s:
        return "structural – add field"
    if "add cg" in s or "added cg" in s or "new cg" in s or "add conn" in s:
        return "structural – add connection"
    if "del. fg" in s or "del. cg" in s or "remove fg" in s or "remove cg" in s:
        return "structural – delete"
    if "enabled" in s or "disabled" in s:
        return "structural – enable/disable connection"

    # ---- type / sign changes ----
    if "to inhibitory" in s or "to excitatory" in s:
        return "type / sign change – field"
    if "to gk" in s or "to mhk" in s or "to kernel" in s:
        return "type / sign change – kernel"

    # ---- field parameters ----
    if "rest. lvl" in s or "rest level" in s:
        return "field parameter – resting level"
    if "tau" in s or "time const" in s:
        return "field parameter – dynamics"

    # ---- kernel parameters ----
    if "gk width" in s or "width" in s:
        return "kernel parameter – width"
    if "gk amp" in s or "amp." in s or "ampl." in s:
        return "kernel parameter – amplitude"

    return "other / uncategorised"


@st.cache_data
@st.cache_data
def compute_mutation_events(run_dir_str: str, generations: tuple):
    """
    Parse statistics/generation_X.txt files and extract mutation events.

    Now we treat entries like:
      fg 2 (fg gk width -1.000000)(fg gk amp.-1.000000)
    as TWO distinct mutations on the same field gene.

    Returns a DataFrame with columns:
      generation, solution_id, fitness,
      gene_type ('fg' or 'cg'), gene_ref ('2', '1-3', ...),
      mutation_inner (text inside a single (...) ),
      mutation_raw  (gene + inner, e.g. 'fg 2: fg gk width -1.000000'),
      category      (from mutation_inner).
    One row per (solution, sub-mutation) pair.
    """
    run_dir = Path(run_dir_str)
    stats_dir = run_dir / "statistics"
    if not stats_dir.exists():
        return pd.DataFrame()

    records = []

    # solution line pattern with last mutations{...}
    sol_pattern = re.compile(
        r"solution\s+(?P<id>\d+)\s+\[\s*fit\.\:\s*(?P<fit>[0-9eE\.\+\-]+).*?"
        r"last mutations\{(?P<muts>.*?)\}\]",
        re.UNICODE,
    )

    for g in generations:
        path = stats_dir / f"generation_{g}.txt"
        if not path.exists():
            continue

        with path.open("r") as f:
            for line in f:
                m = sol_pattern.search(line)
                if not m:
                    continue

                sol_id = int(m.group("id"))
                fit = float(m.group("fit"))
                muts_block = m.group("muts").strip()
                if not muts_block:
                    continue

                # first split into [ ... ] blocks -> one per gene mutation
                gene_mutations = re.findall(r"\[([^\]]+)\]", muts_block)
                for gm in gene_mutations:
                    gm = gm.strip()
                    if not gm:
                        continue

                    # gm looks like: "fg 2 (fg gk width -1.000000)(fg gk amp.-1.000000)"
                    m_head = re.match(
                        r"(?P<gene_type>[fc]g)\s+(?P<ref>[^\s(]+)\s*(?P<rest>.*)", gm
                    )
                    if m_head:
                        gene_type = m_head.group("gene_type")
                        gene_ref = m_head.group("ref")
                        rest = m_head.group("rest") or ""
                    else:
                        gene_type = ""
                        gene_ref = ""
                        rest = gm

                    # extract each (...) as one sub-mutation
                    inners = re.findall(r"\(([^)]+)\)", rest)
                    if not inners:
                        # fallback: treat the whole string as one mutation
                        mut_inner = rest.strip() or gm
                        mut_full = gm
                        category = categorize_mutation(mut_inner)
                        records.append(
                            {
                                "generation": g,
                                "solution_id": sol_id,
                                "fitness": fit,
                                "gene_type": gene_type,
                                "gene_ref": gene_ref,
                                "mutation_inner": mut_inner,
                                "mutation_raw": mut_full,
                                "category": category,
                            }
                        )
                    else:
                        for inner in inners:
                            mut_inner = inner.strip()
                            mut_full = f"{gene_type} {gene_ref}: {mut_inner}".strip()
                            category = categorize_mutation(mut_inner)
                            records.append(
                                {
                                    "generation": g,
                                    "solution_id": sol_id,
                                    "fitness": fit,
                                    "gene_type": gene_type,
                                    "gene_ref": gene_ref,
                                    "mutation_inner": mut_inner,
                                    "mutation_raw": mut_full,
                                    "category": category,
                                }
                            )

    if not records:
        return pd.DataFrame()

    df_mut = pd.DataFrame(records)
    df_mut.sort_values(["generation", "solution_id"], inplace=True)
    df_mut.reset_index(drop=True, inplace=True)
    return df_mut


def plot_mutations_per_generation(mut_events: pd.DataFrame):
    """Line plot: how many mutations occurred each generation."""
    per_gen = (
        mut_events.groupby("generation")["mutation_raw"]
        .count()
        .reset_index(name="num_mutations")
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_gen["generation"], per_gen["num_mutations"])
    ax.set_xlabel("generation")
    ax.set_ylabel("number of mutations")
    ax.set_title("Mutation activity per generation")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_mutation_effectiveness(mut_events: pd.DataFrame, top_n: int = 10):
    """
    Horizontal bar plot of the top mutations with the highest
    positive Δ mean fitness vs the global mean.
    """
    if mut_events.empty:
        return None

    summary = (
        mut_events.groupby("mutation_raw")["fitness"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "occurrences", "mean": "mean_fitness"})
    )

    overall_mean = mut_events["fitness"].mean()
    summary["delta_vs_global"] = summary["mean_fitness"] - overall_mean

    # Top N beneficial
    top = (
        summary.sort_values("delta_vs_global", ascending=False)
        .head(top_n)
        .sort_values("delta_vs_global")  # so bars go from worst -> best
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["mutation_raw"], top["delta_vs_global"])
    ax.set_xlabel("Δ mean fitness vs global")
    ax.set_title(f"Most beneficial mutations (top {len(top)})")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def render_mutation_tables(mut_events: pd.DataFrame):
    """Tables: frequency + effectiveness, with clearer column names."""
    if mut_events.empty:
        st.info("No mutation information found in statistics/generation_X.txt.")
        return

    overall_mean = mut_events["fitness"].mean()

    summary = (
        mut_events.groupby(["mutation_raw", "category"])["fitness"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "occurrences", "mean": "mean_fitness"})
    )
    summary["delta_vs_global"] = summary["mean_fitness"] - overall_mean

    # Most frequent
    st.markdown("#### Most frequent mutations")
    st.caption(
        "How often each mutation appears, and how the average fitness of individuals "
        "with that mutation compares to the overall mean fitness of the run."
    )
    top_freq = summary.sort_values("occurrences", ascending=False).head(10)
    st.dataframe(
        top_freq.rename(
            columns={
                "mutation_raw": "mutation description",
                "category": "mutation category",
                "occurrences": "how many times this mutation was applied",
                "mean_fitness": "mean fitness of individuals with this mutation",
                "delta_vs_global": "Δ vs global mean fitness",
            }
        ),
        use_container_width=True,
    )

    # Most beneficial (positive delta)
    st.markdown("#### Mutations associated with higher fitness")
    st.caption(
        "Mutations whose carriers tend to have higher fitness than the global mean. "
        "Beware of very rare mutations (low counts) — they can look good just by chance."
    )
    top_good = (
        summary.sort_values("delta_vs_global", ascending=False)
        .head(10)
        .rename(
            columns={
                "mutation_raw": "mutation description",
                "category": "mutation category",
                "occurrences": "how many times this mutation was applied",
                "mean_fitness": "mean fitness of individuals with this mutation",
                "delta_vs_global": "Δ vs global mean fitness",
            }
        )
    )
    st.dataframe(top_good, use_container_width=True)


def plot_mutation_categories(mut_events: pd.DataFrame):
    """Bar chart: how many mutation events per high-level category."""
    per_cat = (
        mut_events.groupby("category")["mutation_raw"]
        .count()
        .reset_index(name="num_events")
        .sort_values("num_events", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(per_cat["category"], per_cat["num_events"])
    ax.set_ylabel("mutation events")
    ax.set_title("Mutation events by category")

    # rotate + right-align x tick labels for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    fig.tight_layout()
    return fig



def compute_target_crossing_mutations(
    mut_events: pd.DataFrame, overview_df: pd.DataFrame, target: float
):
    """
    Approximate 'mutations that pushed above target':

    For each generation where best_fitness crosses from <target to >=target,
    look at solutions in that generation with fitness >= target and collect
    their mutations.
    """
    if mut_events.empty or overview_df.empty:
        return pd.DataFrame()

    rows = []
    for i in range(1, len(overview_df)):
        prev = overview_df.iloc[i - 1]
        curr = overview_df.iloc[i]
        if prev["best_fitness"] < target <= curr["best_fitness"]:
            g = int(curr["generation"])
            events_g = mut_events[mut_events["generation"] == g]
            if events_g.empty:
                continue
            high = events_g[events_g["fitness"] >= target]
            if high.empty:
                continue

            summary = (
                high.groupby(["mutation_raw", "category"])["fitness"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "occurrences_in_gen", "mean": "mean_fitness_in_gen"})
            )
            summary["generation"] = g
            rows.append(summary)

    if not rows:
        return pd.DataFrame()

    df_cross = pd.concat(rows, ignore_index=True)
    return df_cross


def compute_per_generation_best_mutation(mut_events: pd.DataFrame):
    """
    For each generation, find the mutation with the highest average fitness
    advantage over that generation's mean fitness (using only mutations
    with at least 3 occurrences to reduce noise).
    """
    if mut_events.empty:
        return pd.DataFrame()

    rows = []
    for g, df_g in mut_events.groupby("generation"):
        gen_mean = df_g["fitness"].mean()
        summary = (
            df_g.groupby(["mutation_raw", "category"])["fitness"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "occurrences", "mean": "mean_fitness"})
        )
        summary = summary[summary["occurrences"] >= 3]
        if summary.empty:
            continue
        summary["delta_vs_gen"] = summary["mean_fitness"] - gen_mean
        best = summary.sort_values("delta_vs_gen", ascending=False).iloc[0]
        rows.append(
            {
                "generation": g,
                "mutation": best["mutation_raw"],
                "category": best["category"],
                "occurrences": best["occurrences"],
                "mean_fitness": best["mean_fitness"],
                "delta_vs_gen": best["delta_vs_gen"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df_pg = pd.DataFrame(rows).sort_values("generation").reset_index(drop=True)
    return df_pg


def plot_best_mutation_timeline(per_gen_df: pd.DataFrame):
    """Line plot: per-generation most beneficial mutation (delta vs gen mean)."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_gen_df["generation"], per_gen_df["delta_vs_gen"])
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("generation")
    ax.set_ylabel("Δ best mutation vs gen mean")
    ax.set_title("Per-generation most beneficial mutation (approx.)")
    ax.grid(True)
    fig.tight_layout()
    return fig


# =========================
# Main app
# =========================

def main():
    st.set_page_config(page_title="neat-dnfs evolution overview", layout="wide")

    if "view" not in st.session_state:
        st.session_state["view"] = "Fitness"
    if "target_fitness" not in st.session_state:
        st.session_state["target_fitness"] = 0.9
    if "partial_targets" not in st.session_state:
        st.session_state["partial_targets"] = {}

    left_col, main_col = st.columns([1, 5])

    # ---------- LEFT PANEL ----------
    with left_col:
        logo_candidate = Path("../resources/images/logo.png")
        if logo_candidate.exists():
            st.image(str(logo_candidate.resolve()), use_container_width=True)
        else:
            st.markdown("### neat-dnfs")

        st.markdown("**Base experiment directory**")
        default_base = Path("../data").resolve()
        base_dir_str = st.text_input(
            label="",
            value=str(default_base),
            help="Directory containing your run folders (each with per_generation_overview.txt).",
        )
        base_dir = Path(base_dir_str).expanduser()

        if not base_dir.exists() or not base_dir.is_dir():
            st.error(f"Base directory does not exist or is not a directory:\n{base_dir}")
            st.stop()

        runs = find_runs_with_overview(base_dir)
        if not runs:
            st.warning("No subfolders with per_generation_overview.txt found.")
            st.stop()

        run_names = [name for name, _ in runs]
        selected_run_name = st.selectbox("Selected run:", run_names)
        selected_run_path = str(dict(runs)[selected_run_name])

        st.markdown(f"<small>{selected_run_path}</small>", unsafe_allow_html=True)

    # ---------- MAIN PANEL ----------
    with main_col:
        h1_col, h2_col = st.columns([3, 2])
        with h1_col:
            st.markdown("## neat-dnfs evolution overview dashboard")
        with h2_col:
            bcols = st.columns(4)
            with bcols[0]:
                if st.button("Fitness", use_container_width=True):
                    st.session_state["view"] = "Fitness"
            with bcols[1]:
                if st.button("Species", use_container_width=True):
                    st.session_state["view"] = "Species"
            with bcols[2]:
                if st.button("Topology", use_container_width=True):
                    st.session_state["view"] = "Topology"
            with bcols[3]:
                if st.button("Mutations", use_container_width=True):
                    st.session_state["view"] = "Mutations"

        view = st.session_state["view"]
        df = load_overview(selected_run_path)
        gens_tuple = tuple(df["generation"].tolist())

        # ---------------- FITNESS ----------------
        if view == "Fitness":
            st.markdown("---")

            min_fit = float(min(df["avg_fitness"].min(), df["best_fitness"].min(), 0.0))
            max_fit = float(max(df["avg_fitness"].max(), df["best_fitness"].max(), 1.0))
            slider_min = min_fit
            slider_max = max_fit if max_fit > min_fit else min_fit + 1.0

            c1, c2 = st.columns([4, 1])
            with c1:
                target = st.slider(
                    "Overall target fitness (for main plot)",
                    min_value=slider_min,
                    max_value=slider_max,
                    value=float(st.session_state["target_fitness"]),
                    step=(slider_max - slider_min) / 100.0 if slider_max > slider_min else 0.01,
                )
            with c2:
                target = st.number_input(
                    "Target (manual)",
                    value=float(target),
                    min_value=slider_min,
                    max_value=slider_max,
                )

            st.session_state["target_fitness"] = float(target)

            st.markdown("### Total fitness")
            fig_total = plot_total_fitness(df, st.session_state["target_fitness"])
            st.pyplot(fig_total)

            st.markdown("---")
            stats_col, partial_col = st.columns([1, 3])

            with stats_col:
                render_fitness_stats(df, st.session_state["target_fitness"])

            with partial_col:
                partial_df = compute_partial_fitness(selected_run_path, gens_tuple)
                partial_targets = st.session_state["partial_targets"]

                if partial_df is not None and not partial_df.empty:
                    num_partial = 0
                    for col in partial_df.columns:
                        if col.startswith("best_p"):
                            idx = int(col.replace("best_p", ""))
                            num_partial = max(num_partial, idx)

                    if num_partial > 0:
                        with st.expander("Targets for partial fitness components", expanded=False):
                            rows = math.ceil(num_partial / 3)
                            comp = 1
                            for _ in range(rows):
                                cols = st.columns(3)
                                for c in cols:
                                    if comp > num_partial:
                                        break

                                    best_col = f"best_p{comp}"
                                    avg_col = f"avg_p{comp}"
                                    if best_col not in partial_df.columns or avg_col not in partial_df.columns:
                                        comp += 1
                                        continue

                                    if comp not in partial_targets:
                                        partial_targets[comp] = float(st.session_state["target_fitness"])

                                    col_min = float(
                                        min(partial_df[best_col].min(), partial_df[avg_col].min(), 0.0)
                                    )
                                    col_max = float(
                                        max(partial_df[best_col].max(), partial_df[avg_col].max(), 1.0)
                                    )

                                    with c:
                                        st.markdown(f"**partial {comp}**")
                                        val = st.number_input(
                                            f"target p{comp}",
                                            value=float(partial_targets[comp]),
                                            min_value=col_min,
                                            max_value=col_max,
                                            key=f"num_p{comp}",
                                        )
                                        partial_targets[comp] = float(val)

                                    comp += 1

                        st.session_state["partial_targets"] = partial_targets

                plot_partial_fitness_grid(partial_df, st.session_state["partial_targets"])

        # ---------------- SPECIES ----------------
        elif view == "Species":
            st.markdown("---")

            top_left, top_right = st.columns(2)
            with top_left:
                fig_sc = plot_species_counts(df)
                st.pyplot(fig_sc)
            with top_right:
                fig_innov = plot_innovation_growth(df)
                st.pyplot(fig_innov)

            st.markdown("---")
            bottom_left, bottom_right = st.columns([1, 3])

            species_meta = compute_species_meta(selected_run_path, gens_tuple)

            with bottom_left:
                render_species_stats(df, species_meta)

            with bottom_right:
                st.markdown("#### Species genome (per generation)")

                min_gen = int(df["generation"].min())
                max_gen = int(df["generation"].max())
                gen_sel = st.slider(
                    "Generation to inspect",
                    min_value=min_gen,
                    max_value=max_gen,
                    value=max_gen,
                )

                species_list = get_species_for_generation(selected_run_path, gen_sel)

                if not species_list:
                    st.info(f"No species file found or parsable for generation {gen_sel}.")
                else:
                    active_species = [
                        s for s in species_list if (not s["extinct"]) and s["members"] > 0
                    ]

                    st.markdown(
                        f"Generation **{gen_sel}** — active species: **{len(active_species)}** "
                        f"(total logged species in file: {len(species_list)})"
                    )

                    if not active_species:
                        st.info("No active species with members > 0 in this generation.")
                    else:
                        for s in active_species:
                            improved_str = "yes" if s["improved"] else "no"
                            header = (
                                f"Species {s['id']} "
                                f"(age {s['age']}, members {s['members']}, "
                                f"offspring {s['offspring']}, improved this gen: {improved_str}, "
                                f"gens since imp.: {s['gens_since_improvement']})"
                            )
                            with st.expander(header, expanded=False):
                                st.markdown("**Representative solution (raw log snippet):**")
                                if s["rep_raw"]:
                                    st.code(s["rep_raw"], language="text")
                                else:
                                    st.write("_none_")

                                st.markdown("**Champion solution (raw log snippet):**")
                                if s["champ_raw"]:
                                    st.code(s["champ_raw"], language="text")
                                else:
                                    st.write("_none_")

        # ---------------- TOPOLOGY ----------------
        elif view == "Topology":
            st.markdown("---")

            # Top: genome topology curves + stats
            top_left, top_right = st.columns(2)
            with top_left:
                fig_gen = plot_genome_topology_curves(df)
                st.pyplot(fig_gen)
            with top_right:
                render_topology_stats(df)

            st.markdown("---")

            # Bottom: best-solution topology over generations
            st.markdown("#### Topology (phenotype) of best solution across generations")

            min_gen = int(df["generation"].min())
            max_gen = int(df["generation"].max())
            gen_sel = st.slider(
                "Generation to inspect (best solution topology)",
                min_value=min_gen,
                max_value=max_gen,
                value=max_gen,
            )

            elements = load_best_solution_architecture(selected_run_path, gen_sel)
            if elements is None:
                st.info(
                    "No best-solution JSON found in "
                    "`best_solutions/prev_generations` for this generation."
                )
            else:
                g, labels = build_topology_graph(elements)
                fig_top = plot_topology_graph(g, labels)
                st.pyplot(fig_top)


        # ---------------- MUTATIONS (placeholder) ----------------
               # ---------------- MUTATIONS ----------------
        elif view == "Mutations":
            st.markdown("---")

            mut_events = compute_mutation_events(selected_run_path, gens_tuple)

            if mut_events.empty:
                st.info(
                    "No mutation logs found in statistics/generation_X.txt "
                    "(the 'last mutations{...}' field appears empty)."
                )
            else:
                # --- Top row: activity + category breakdown ---
                top_row_left, top_row_right = st.columns(2)
                with top_row_left:
                    fig_muts = plot_mutations_per_generation(mut_events)
                    st.pyplot(fig_muts)
                    st.caption("Total number of mutation events applied in each generation.")

                with top_row_right:
                    fig_cat = plot_mutation_categories(mut_events)
                    st.pyplot(fig_cat)
                    st.caption("How mutation events are distributed across high-level categories.")

                # --- Middle: most beneficial mutations (bar plot) ---
                st.markdown("---")
                fig_eff = plot_mutation_effectiveness(mut_events)
                if fig_eff is not None:
                    st.pyplot(fig_eff)
                    st.caption(
                        "Mutations that, on average, appear in higher-fitness individuals "
                        "compared to the global mean. Bar length shows the improvement."
                    )

                # --- Tables: frequency + effectiveness ---
                st.markdown("---")
                render_mutation_tables(mut_events)

                # --- Mutations involved when crossing the target fitness ---
                target = float(st.session_state.get("target_fitness", 0.9))
                crossing_df = compute_target_crossing_mutations(mut_events, df, target)

                if not crossing_df.empty:
                    st.markdown("#### Mutations present when best fitness crossed the target")
                    st.caption(
                        "Generations where best fitness first moved from below the selected "
                        f"target (currently {target:.3f}) to above it. "
                        "For those generations, this table shows which mutations were present "
                        "in above-target individuals. This is an approximation of "
                        "\"mutations that pushed solutions over the threshold\"."
                    )
                    show_cols = crossing_df.copy()
                    show_cols = show_cols.rename(
                        columns={
                            "generation": "generation",
                            "mutation_raw": "mutation description",
                            "category": "mutation category",
                            "occurrences_in_gen": "times this mutation appears in that generation",
                            "mean_fitness_in_gen": "mean fitness of its carriers in that generation",
                        }
                    )
                    show_cols = show_cols.sort_values(
                        ["generation", "mean_fitness_in_gen"], ascending=[True, False]
                    )
                    st.dataframe(show_cols.head(20), use_container_width=True)

                # --- Per-generation most impactful mutation timeline ---
                per_gen_best = compute_per_generation_best_mutation(mut_events)
                if not per_gen_best.empty:
                    st.markdown("#### Per-generation most beneficial mutation")
                    st.caption(
                        "For each generation, this considers mutations that appear at least "
                        "three times and selects the one whose carriers have the largest "
                        "advantage over that generation's average fitness."
                    )

                    fig_pg = plot_best_mutation_timeline(per_gen_best)
                    st.pyplot(fig_pg)

                    st.markdown("Top generations by mutation impact:")
                    st.dataframe(
                        per_gen_best.sort_values("delta_vs_gen", ascending=False)
                        .head(10)
                        .rename(
                            columns={
                                "generation": "generation",
                                "mutation": "mutation description",
                                "category": "mutation category",
                                "occurrences": "count in that generation",
                                "mean_fitness": "mean fitness of its carriers",
                                "delta_vs_gen": "Δ vs generation mean fitness",
                            }
                        ),
                        use_container_width=True,
                    )

if __name__ == "__main__":
    main()
