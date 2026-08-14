import matplotlib
import matplotlib.colors as mcolors

from viz.theme import (
    CATEGORICAL_CYCLE,
    COLOR_AVG,
    COLOR_BEST,
    COLOR_FAILURE,
    COLOR_STRUCTURAL_CHANGE,
    COLOR_SUCCESS,
    COLOR_TARGET,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    apply_plot_style,
    theme_type,
)


def test_apply_plot_style_sets_expected_rcparams():
    apply_plot_style()
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["axes.grid"] is True
    assert matplotlib.rcParams["legend.frameon"] is False
    assert matplotlib.rcParams["figure.dpi"] == 150


def test_theme_type_defaults_to_light_outside_streamlit_context():
    assert theme_type() == "light"


def test_color_tokens_are_valid_matplotlib_colors():
    tokens = [COLOR_BEST, COLOR_AVG, COLOR_TARGET, COLOR_SUCCESS, COLOR_FAILURE, COLOR_STRUCTURAL_CHANGE]
    for c in tokens:
        assert mcolors.is_color_like(c), c
    for c in CATEGORICAL_CYCLE:
        assert mcolors.is_color_like(c), c


def test_colormap_names_are_registered_with_matplotlib():
    import matplotlib.pyplot as plt

    plt.get_cmap(SEQUENTIAL_CMAP)
    plt.get_cmap(DIVERGING_CMAP)


def test_register_fonts_falls_back_gracefully_when_font_dir_missing(monkeypatch, tmp_path):
    from viz import theme as theme_module

    monkeypatch.setattr(theme_module, "_FONTS_DIR", tmp_path / "does-not-exist")
    family = theme_module._register_fonts()
    assert family == theme_module._FALLBACK_FONT_FAMILY
