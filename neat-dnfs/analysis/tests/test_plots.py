import altair as alt
import pandas as pd

from viz.plots import chart_all_runs_overlay, chart_species_membership, chart_topology_trajectory, chart_total_fitness


def _has_field(chart_dict, field_name) -> bool:
    """Search a vega-lite spec dict (possibly nested under layer/hconcat/vconcat) for any
    encoding channel referencing `field_name`."""
    if "encoding" in chart_dict:
        for channel in chart_dict["encoding"].values():
            if isinstance(channel, dict) and channel.get("field") == field_name:
                return True
            if isinstance(channel, list):
                for c in channel:
                    if isinstance(c, dict) and c.get("field") == field_name:
                        return True
    for key in ("layer", "hconcat", "vconcat"):
        if key in chart_dict:
            for sub in chart_dict[key]:
                if _has_field(sub, field_name):
                    return True
    return False


def test_chart_total_fitness_returns_layered_chart_with_expected_fields():
    df = pd.DataFrame(
        {
            "generation": [0, 1, 2, 3],
            "avg_fitness": [0.1, 0.2, 0.3, 0.4],
            "best_fitness": [0.2, 0.3, 0.5, 0.6],
        }
    )
    chart = chart_total_fitness(df, target_fitness=0.5, success_generations=[4], success_label="met")
    assert isinstance(chart, (alt.LayerChart, alt.Chart))
    spec = chart.to_dict()
    assert _has_field(spec, "gen_display")
    assert _has_field(spec, "fitness")


def test_chart_total_fitness_without_success_generations_still_builds():
    df = pd.DataFrame({"generation": [0, 1], "avg_fitness": [0.1, 0.2], "best_fitness": [0.2, 0.3]})
    chart = chart_total_fitness(df, target_fitness=0.5)
    assert chart.to_dict() is not None


def test_chart_topology_trajectory_returns_expected_fields():
    traj_df = pd.DataFrame(
        {
            "generation": [0, 1, 2, 3],
            "hidden_fields": [0, 0, 1, 1],
            "enabled_connections": [3, 3, 3, 4],
        }
    )
    chart = chart_topology_trajectory(traj_df)
    spec = chart.to_dict()
    assert _has_field(spec, "gen_display")
    assert _has_field(spec, "count")


def test_chart_all_runs_overlay_handles_empty_input():
    chart = chart_all_runs_overlay([], target_fitness=0.9)
    assert chart.to_dict() is not None


def test_chart_all_runs_overlay_returns_expected_fields():
    all_run_metrics = [
        {"run_dir": "run1", "fitness_values": [0.1, 0.2, 0.3], "success": True},
        {"run_dir": "run2", "fitness_values": [0.05, 0.1], "success": False},
    ]
    chart = chart_all_runs_overlay(all_run_metrics, target_fitness=0.25)
    spec = chart.to_dict()
    assert _has_field(spec, "generation")
    assert _has_field(spec, "fitness")


def test_chart_species_membership_handles_empty_input():
    chart = chart_species_membership({})
    assert chart.to_dict() is not None


def test_chart_species_membership_returns_expected_fields():
    species_meta = {
        1: {"max_members": 10, "members_by_gen": {0: 5, 1: 6}},
        2: {"max_members": 3, "members_by_gen": {0: 2, 1: 2}},
    }
    chart = chart_species_membership(species_meta, top_n=1)
    spec = chart.to_dict()
    assert _has_field(spec, "generation")
    assert _has_field(spec, "members")
