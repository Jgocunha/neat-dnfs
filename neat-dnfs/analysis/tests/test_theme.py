import altair as alt
import matplotlib
import matplotlib.colors as mcolors

from viz.theme import (
    CATEGORICAL_CYCLE,
    CHART_HEIGHT,
    CHART_HEIGHT_TALL,
    COLOR_AVG,
    COLOR_BEST,
    COLOR_FAILURE,
    COLOR_STRUCTURAL_CHANGE,
    COLOR_SUCCESS,
    COLOR_TARGET,
    DIVERGING_CMAP,
    MAX_CONTENT_WIDTH_PX,
    SEQUENTIAL_CMAP,
    SIDEBAR_LOGO_PAD_BOTTOM_REM,
    SIDEBAR_LOGO_PAD_TOP_REM,
    SIDEBAR_LOGO_PAD_X_REM,
    SIDEBAR_LOGO_WIDTH_PCT,
    apply_plot_style,
    register_altair_theme,
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


def test_sidebar_logo_and_layout_constants_are_sane():
    assert 0 < SIDEBAR_LOGO_WIDTH_PCT <= 100
    for pad in (SIDEBAR_LOGO_PAD_TOP_REM, SIDEBAR_LOGO_PAD_X_REM, SIDEBAR_LOGO_PAD_BOTTOM_REM):
        assert pad > 0
    assert MAX_CONTENT_WIDTH_PX > 0
    assert 0 < CHART_HEIGHT <= CHART_HEIGHT_TALL


def test_register_altair_theme_registers_and_enables():
    register_altair_theme()
    assert alt.theme.active == "neat_dnfs"


def test_register_altair_theme_config_uses_categorical_cycle():
    register_altair_theme()
    config = alt.theme.get()()
    assert config["config"]["range"]["category"] == CATEGORICAL_CYCLE
    assert config["config"]["view"]["height"] == CHART_HEIGHT
