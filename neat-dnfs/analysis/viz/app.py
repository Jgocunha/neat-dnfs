from pathlib import Path
import streamlit as st

from .parsing import find_runs_with_overview, load_overview
from .views import render_experiment_view, render_fitness_view, render_mutations_view, render_species_view, render_topology_view
from .report import export_experiment_markdown, export_run_markdown

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
_VIEWS = ["Fitness", "Species", "Topology", "Mutations", "Experiment"]

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
        logo_candidate = _ANALYSIS_DIR.parent / "resources" / "images" / "logo.png"
        if logo_candidate.exists():
            st.image(str(logo_candidate.resolve()), width="stretch")
        else:
            st.markdown("### neat-dnfs")

        st.markdown("**Base experiment directory**")
        default_base = (_ANALYSIS_DIR.parent / "data").resolve()
        base_dir_str = st.text_input(
            label="Base experiment directory path",
            value=str(default_base),
            help="Directory containing your run folders (each with per_generation_overview.txt).",
            label_visibility="collapsed",  # hides the label visually but keeps it non-empty
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

        # --- Export controls (left side) ---
        ctrl_col1, ctrl_col2 = st.columns(2)

        with ctrl_col1:
            if st.button("Export run .md", width="stretch"):
                md_path = export_run_markdown(
                    selected_run_path,
                    target_fitness=float(st.session_state.get("target_fitness", 0.9)),
                )
                st.success(f"Run summary exported to:\n{md_path}")

        with ctrl_col2:
            if st.button("Export experiment .md", width="stretch"):
                partial_targets = st.session_state.get("partial_targets", {})
                targets_items = tuple(sorted((int(k), float(v)) for k, v in partial_targets.items()))
                md_path = export_experiment_markdown(
                    base_dir_str,
                    partial_targets_items=targets_items,
                )
                st.success(f"Experiment summary exported to:\n{md_path}")

    # ---------- MAIN PANEL ----------
    with main_col:
        h1_col, h2_col = st.columns([3, 2])
        with h1_col:
            st.markdown("## neat-dnfs evolution overview dashboard")
        with h2_col:
            view = st.radio(
                "View",
                _VIEWS,
                horizontal=True,
                key="view",
                label_visibility="collapsed",
            )

        df = load_overview(selected_run_path)
        gens_tuple = tuple(df["generation"].tolist())

        if view == "Fitness":
            render_fitness_view(df, gens_tuple, selected_run_path)
        elif view == "Species":
            render_species_view(df, gens_tuple, selected_run_path)
        elif view == "Topology":
            render_topology_view(df, selected_run_path)
        elif view == "Mutations":
            render_mutations_view(df, gens_tuple, selected_run_path)
        elif view == "Experiment":
            render_experiment_view(base_dir_str)
