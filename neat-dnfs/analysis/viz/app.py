from pathlib import Path
import streamlit as st

from . import theme
from .parsing import find_experiment_dirs, find_runs_with_overview, load_overview, run_picker_label
from .views import render_cross_experiment_view, render_experiment_view, render_fitness_view, render_mutations_view, render_species_view, render_topology_view
from .report import export_experiment_markdown, export_run_markdown

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
_VIEWS = ["Fitness", "Species", "Topology", "Mutations", "Experiment", "Compare"]

def main():
    icon_candidate = _ANALYSIS_DIR.parent / "resources" / "icons" / "icon.ico"
    st.set_page_config(
        page_title="neat-dnfs evolution overview",
        page_icon=str(icon_candidate.resolve()) if icon_candidate.exists() else None,
        layout="wide",
    )

    if "view" not in st.session_state:
        st.session_state["view"] = "Fitness"
    if "target_fitness" not in st.session_state:
        st.session_state["target_fitness"] = 0.9
    if "partial_targets" not in st.session_state:
        st.session_state["partial_targets"] = {}

    logo_candidate = _ANALYSIS_DIR.parent / "resources" / "images" / "logo.png"
    if logo_candidate.exists():
        st.logo(
            str(logo_candidate.resolve()),
            size="large",
            icon_image=str(icon_candidate.resolve()) if icon_candidate.exists() else None,
        )
        st.markdown(
            f"""
            <style>
            [data-testid="stSidebarHeader"] {{
                height: {theme.SIDEBAR_LOGO_HEIGHT_REM}rem !important;
            }}
            [data-testid="stSidebarLogo"], [data-testid="stLogo"] {{
                height: {theme.SIDEBAR_LOGO_HEIGHT_REM}rem !important;
                width: auto !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ---------- SIDEBAR ----------
    with st.sidebar:
        if not logo_candidate.exists():
            st.markdown("### neat-dnfs")

        default_data_root = str((_ANALYSIS_DIR.parent / "data").resolve())
        if "data_root" not in st.session_state:
            st.session_state["data_root"] = default_data_root

        st.caption(f"Data root: `{st.session_state['data_root']}`")
        with st.expander("Advanced: change data root", expanded=False):
            new_root = st.text_input(
                "Data root path",
                value=st.session_state["data_root"],
                help="Directory containing your experiment folders (each holding run folders "
                "with per_generation_overview.txt).",
            )
            st.session_state["data_root"] = new_root

        data_root = Path(st.session_state["data_root"]).expanduser()

        if not data_root.exists() or not data_root.is_dir():
            st.error(f"Data root does not exist or is not a directory:\n{data_root}")
            st.stop()

        experiments = find_experiment_dirs(data_root)
        if not experiments:
            st.warning("No experiment subfolders with runs found under this data root.")
            st.stop()

        experiment_names = [name for name, _ in experiments]
        prev_experiment = st.session_state.get("experiment")
        default_experiment_idx = (
            experiment_names.index(prev_experiment) if prev_experiment in experiment_names else 0
        )
        st.markdown("**Experiment**")
        selected_experiment_name = st.selectbox(
            "Experiment",
            experiment_names,
            index=default_experiment_idx,
            key="experiment_select",
            label_visibility="collapsed",
        )
        st.session_state["experiment"] = selected_experiment_name
        selected_experiment_dir = dict(experiments)[selected_experiment_name]

        runs = find_runs_with_overview(selected_experiment_dir)
        if not runs:
            st.warning("No subfolders with per_generation_overview.txt found in this experiment.")
            st.stop()
        st.caption(f"{len(runs)} run(s) in this experiment")

        run_paths = [path for _, path in runs]
        run_labels = [run_picker_label(str(path)) for path in run_paths]
        label_to_path = dict(zip(run_labels, run_paths))
        prev_run_path_str = st.session_state.get("selected_run_path")
        default_run_idx = 0
        for i, run_path in enumerate(run_paths):
            if str(run_path) == prev_run_path_str:
                default_run_idx = i
                break
        st.markdown("**Run**")
        selected_run_label = st.selectbox(
            "Run",
            run_labels,
            index=default_run_idx,
            key="run_select",
            label_visibility="collapsed",
        )
        selected_run_path = str(label_to_path[selected_run_label])
        st.session_state["selected_run_path"] = selected_run_path

        st.markdown(f"<small>{selected_run_path}</small>", unsafe_allow_html=True)

        # --- Export controls (left side) ---
        ctrl_col1, ctrl_col2 = st.columns(2)

        with ctrl_col1:
            if st.button("Export run .md", width="stretch"):
                with st.spinner("Exporting run summary..."):
                    md_path = export_run_markdown(
                        selected_run_path,
                        target_fitness=float(st.session_state.get("target_fitness", 0.9)),
                    )
                st.session_state["run_export_path"] = md_path
                st.session_state["run_export_content"] = Path(md_path).read_text(encoding="utf-8")
                st.toast(f"Run summary exported to {md_path}", icon="✅")

            if "run_export_content" in st.session_state:
                st.download_button(
                    "Download run .md",
                    data=st.session_state["run_export_content"],
                    file_name=Path(st.session_state["run_export_path"]).name,
                    mime="text/markdown",
                    width="stretch",
                )

        with ctrl_col2:
            if st.button("Export experiment .md", width="stretch"):
                partial_targets = st.session_state.get("partial_targets", {})
                targets_items = tuple(sorted((int(k), float(v)) for k, v in partial_targets.items()))
                with st.spinner("Exporting experiment summary..."):
                    md_path = export_experiment_markdown(
                        str(selected_experiment_dir),
                        partial_targets_items=targets_items,
                    )
                st.session_state["experiment_export_path"] = md_path
                st.session_state["experiment_export_content"] = Path(md_path).read_text(encoding="utf-8")
                st.toast(f"Experiment summary exported to {md_path}", icon="✅")

            if "experiment_export_content" in st.session_state:
                st.download_button(
                    "Download experiment .md",
                    data=st.session_state["experiment_export_content"],
                    file_name=Path(st.session_state["experiment_export_path"]).name,
                    mime="text/markdown",
                    width="stretch",
                )

    # ---------- MAIN PANEL ----------
    h1_col, h2_col = st.columns([3, 2])
    with h1_col:
        st.markdown("## neat-dnfs evolution overview dashboard")
    with h2_col:
        view = st.segmented_control(
            "View",
            _VIEWS,
            key="view",
            required=True,
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
        render_experiment_view(str(selected_experiment_dir))
    elif view == "Compare":
        render_cross_experiment_view(str(data_root))
