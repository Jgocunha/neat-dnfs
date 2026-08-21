from pathlib import Path
import streamlit as st

from . import theme
from .parsing import find_experiment_dirs, find_runs_with_overview, load_overview, run_picker_label
from .views import render_cross_experiment_view, render_experiment_view, render_fitness_view, render_mutations_view, render_species_view, render_topology_view
from .report import export_experiment_markdown, export_run_markdown

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]

# Single-run pages analyse one selected run; comparison pages aggregate across runs
# (Experiment) or across experiments (Compare). The sidebar's Experiment/Run pickers and
# export buttons are shown or hidden per page based on which scope it belongs to.
_SCOPE_RUN = {"Fitness", "Species", "Topology", "Mutations"}
_SCOPE_EXPERIMENT = {"Experiment"}

# icon + one-line tagline per page, shown as the page header so each of the 6 pages reads as
# its own place rather than an unlabeled continuation of the same screen.
_PAGE_TAGLINES = {
    "Fitness": (":material/trending_up:", "Best and average fitness against your target, generation by generation."),
    "Species": (":material/hive:", "Speciation and genetic diversity as the population evolves."),
    "Topology": (":material/account_tree:", "How genome structure -- fields, connections, kernels -- grows and settles."),
    "Mutations": (":material/shuffle:", "Which mutations fire most often, and which ones actually pay off."),
    "Experiment": (":material/science:", "Every run in this experiment, aggregated into one convergence picture."),
    "Compare": (":material/compare_arrows:", "Several experiments side by side -- what actually moved the needle."),
}


def _render_page_header(pg) -> None:
    icon, tagline = _PAGE_TAGLINES.get(pg.title, ("", ""))
    st.markdown(f"## {icon} {pg.title}".strip())
    if tagline:
        st.caption(tagline)


def _page_fitness():
    run_path = st.session_state["selected_run_path"]
    df = load_overview(run_path)
    render_fitness_view(df, tuple(df["generation"].tolist()), run_path)


def _page_species():
    run_path = st.session_state["selected_run_path"]
    df = load_overview(run_path)
    render_species_view(df, tuple(df["generation"].tolist()), run_path)


def _page_topology():
    run_path = st.session_state["selected_run_path"]
    df = load_overview(run_path)
    render_topology_view(df, run_path)


def _page_mutations():
    run_path = st.session_state["selected_run_path"]
    df = load_overview(run_path)
    render_mutations_view(df, tuple(df["generation"].tolist()), run_path)


def _page_experiment():
    render_experiment_view(st.session_state["selected_experiment_dir"])


def _page_compare():
    render_cross_experiment_view(st.session_state["data_root"])


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebarHeader"] {{
            height: auto !important;
            padding: {theme.SIDEBAR_LOGO_PAD_TOP_REM}rem {theme.SIDEBAR_LOGO_PAD_X_REM}rem
                     {theme.SIDEBAR_LOGO_PAD_BOTTOM_REM}rem !important;
        }}
        [data-testid="stSidebarLogo"] {{
            width: {theme.SIDEBAR_LOGO_WIDTH_PCT}% !important;
            height: auto !important;
            max-height: none !important;
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: {theme.MAX_CONTENT_WIDTH_PX}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    icon_candidate = _ANALYSIS_DIR.parent / "resources" / "icons" / "icon.ico"
    st.set_page_config(
        page_title="neat-dnfs evolution overview",
        page_icon=str(icon_candidate.resolve()) if icon_candidate.exists() else None,
        layout="wide",
    )

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
    _inject_css()

    pg = st.navigation(
        {
            "Single run": [
                st.Page(_page_fitness, title="Fitness", icon=":material/trending_up:", default=True),
                st.Page(_page_species, title="Species", icon=":material/hive:"),
                st.Page(_page_topology, title="Topology", icon=":material/account_tree:"),
                st.Page(_page_mutations, title="Mutations", icon=":material/shuffle:"),
            ],
            "Across runs": [
                st.Page(_page_experiment, title="Experiment", icon=":material/science:"),
                st.Page(_page_compare, title="Compare", icon=":material/compare_arrows:"),
            ],
        }
    )
    scope_run = pg.title in _SCOPE_RUN
    scope_experiment = pg.title in _SCOPE_EXPERIMENT

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

        selected_experiment_dir = None
        if scope_run or scope_experiment:
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
            st.session_state["selected_experiment_dir"] = str(selected_experiment_dir)

        if scope_run:
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

        # --- Export controls (stacked, scope-aware) ---
        if scope_run or scope_experiment:
            st.divider()

        if scope_run:
            if st.button(
                "Export run .md",
                width="stretch",
                help="Writes a Markdown summary of the selected run (fitness, species, topology "
                "and mutation statistics) into the run's own folder, then offers it below for "
                "download.",
            ):
                with st.spinner("Exporting run summary..."):
                    md_path = export_run_markdown(
                        selected_run_path,
                        target_fitness=float(st.session_state.get("target_fitness", 0.9)),
                    )
                st.session_state["run_export_path"] = md_path
                st.session_state["run_export_content"] = Path(md_path).read_text(encoding="utf-8")
                st.session_state["run_export_source"] = selected_run_path
                st.toast(f"Run summary exported to {md_path}", icon="✅")

            if st.session_state.get("run_export_source") == selected_run_path:
                st.download_button(
                    "Download run .md",
                    data=st.session_state["run_export_content"],
                    file_name=Path(st.session_state["run_export_path"]).name,
                    mime="text/markdown",
                    width="stretch",
                    help="Save the exported run summary to your machine.",
                )

        if scope_run or scope_experiment:
            if st.button(
                "Export experiment .md",
                width="stretch",
                help="Writes a Markdown summary aggregating every run in the selected experiment "
                "(convergence rates, success ratios, architecture stats) into the experiment's "
                "own folder, then offers it below for download.",
            ):
                partial_targets = st.session_state.get("partial_targets", {})
                targets_items = tuple(sorted((int(k), float(v)) for k, v in partial_targets.items()))
                with st.spinner("Exporting experiment summary..."):
                    md_path = export_experiment_markdown(
                        str(selected_experiment_dir),
                        partial_targets_items=targets_items,
                    )
                st.session_state["experiment_export_path"] = md_path
                st.session_state["experiment_export_content"] = Path(md_path).read_text(encoding="utf-8")
                st.session_state["experiment_export_source"] = str(selected_experiment_dir)
                st.toast(f"Experiment summary exported to {md_path}", icon="✅")

            if st.session_state.get("experiment_export_source") == str(selected_experiment_dir):
                st.download_button(
                    "Download experiment .md",
                    data=st.session_state["experiment_export_content"],
                    file_name=Path(st.session_state["experiment_export_path"]).name,
                    mime="text/markdown",
                    width="stretch",
                    help="Save the exported experiment summary to your machine.",
                )

    # ---------- MAIN PANEL ----------
    _render_page_header(pg)
    pg.run()
