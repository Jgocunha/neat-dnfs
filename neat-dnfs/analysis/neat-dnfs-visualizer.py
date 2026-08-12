#!/usr/bin/env python3
"""Entry point for the neat-dnfs evolution-overview Streamlit dashboard.

The implementation lives in viz/ (see that package for parsing, stats,
plotting, and view logic). This file stays a thin shim so
launch-visualizer.bat and `streamlit run` keep working unchanged, and so
tools that load this file directly (e.g. via importlib.util.spec_from_file_
location) still find export_run_markdown/export_experiment_markdown here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from viz.app import main
from viz.report import export_run_markdown, export_experiment_markdown  # noqa: F401  (re-exported)

if __name__ == "__main__":
    main()
