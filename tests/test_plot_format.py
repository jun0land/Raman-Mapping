"""그래프 텍스트 서식(마크업 변환 + 요소별 굵기/기울기/색) 테스트."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import plot
from core.plot import (
    PlotConfig,
    apply_text_markup,
    _plotly_font_dict,
    _mpl_font_kw,
)


# ---------------------------------------------------------------------------
# Task 1: 마크업 변환
# ---------------------------------------------------------------------------
def test_superscript_plotly():
    assert apply_text_markup("cm^{-1}", "plotly") == "cm<sup>-1</sup>"


def test_superscript_mpl():
    assert apply_text_markup("cm^{-1}", "mpl") == "cm$^{-1}$"


def test_subscript_plotly():
    assert apply_text_markup("H_{2}O", "plotly") == "H<sub>2</sub>O"


def test_subscript_mpl():
    assert apply_text_markup("H_{2}O", "mpl") == "H$_{2}$O"


def test_no_markup_unchanged_both():
    # 마크업이 없으면 원문 그대로 (특히 mpl에 $ 미삽입)
    assert apply_text_markup("Intensity (a.u.)", "plotly") == "Intensity (a.u.)"
    assert apply_text_markup("Intensity (a.u.)", "mpl") == "Intensity (a.u.)"


def test_bare_caret_underscore_literal():
    assert apply_text_markup("a_b^c", "mpl") == "a_b^c"
    assert apply_text_markup("a_b^c", "plotly") == "a_b^c"


def test_multiple_tokens_mpl():
    assert apply_text_markup("a^{2}+b_{n}", "mpl") == "a$^{2}$+b$_{n}$"


def test_unterminated_markup_returns_original():
    assert apply_text_markup("cm^{-1", "plotly") == "cm^{-1"


def test_empty_text_safe():
    assert apply_text_markup("", "plotly") == ""


# ---------------------------------------------------------------------------
# Task 2: PlotConfig 필드 + 폰트 dict 헬퍼
# ---------------------------------------------------------------------------
def test_plotconfig_defaults_present():
    c = PlotConfig()
    assert c.font_bold_label is False and c.font_italic_title is False
    assert c.font_color_tick == "#000000"


def test_plotly_font_dict_bold_italic_color():
    c = PlotConfig(font_bold_title=True, font_italic_title=True,
                   font_color_title="#ff0000")
    d = _plotly_font_dict(c, "title")
    assert d["weight"] == "bold" and d["style"] == "italic"
    assert d["color"] == "#ff0000" and d["size"] == c.font_size_title


def test_mpl_font_kw_normal_defaults():
    c = PlotConfig()
    k = _mpl_font_kw(c, "tick")
    assert k["fontweight"] == "normal" and k["fontstyle"] == "normal"
    assert k["fontsize"] == c.font_size_tick and k["color"] == "#000000"


# ---------------------------------------------------------------------------
# Task 3: plotly 렌더러
# ---------------------------------------------------------------------------
def test_heatmap_applies_title_style_and_markup():
    c = PlotConfig(title="I^{2}", font_bold_title=True, font_color_title="#123456")
    fig = plot.make_heatmap(np.arange(9.0).reshape(3, 3), c)
    t = fig.layout.title
    assert t.text == "I<sup>2</sup>"
    assert t.font.weight == "bold" and t.font.color == "#123456"


def test_heatmap_tickfont_color():
    c = PlotConfig(font_color_tick="#00ff00", font_italic_tick=True)
    fig = plot.make_heatmap(np.arange(9.0).reshape(3, 3), c)
    assert fig.layout.xaxis.tickfont.color == "#00ff00"
    assert fig.layout.xaxis.tickfont.style == "italic"


def test_surface_axis_label_markup():
    c = PlotConfig(x_label="d_{x}")
    fig = plot.make_surface(np.arange(9.0).reshape(3, 3), c)
    assert fig.layout.scene.xaxis.title.text == "d<sub>x</sub>"


# ---------------------------------------------------------------------------
# Task 4: matplotlib 렌더러
# ---------------------------------------------------------------------------
def test_mpl_heatmap_builds_with_style():
    c = PlotConfig(title="I^{2}", x_label="d_{x}",
                   font_bold_label=True, font_color_title="#ff0000",
                   font_italic_tick=True)
    fig = plot.make_matplotlib_heatmap(np.arange(9.0).reshape(3, 3), c, dpi=72)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "d$_{x}$"
    assert ax.get_title() == "I$^{2}$"
    plt.close(fig)


def test_mpl_surface_builds_without_error():
    c = PlotConfig(title="T_{c}", font_bold_title=True)
    fig = plot.make_matplotlib_surface(np.arange(9.0).reshape(3, 3), c, dpi=72)
    assert fig.axes[0].get_title() == "T$_{c}$"
    plt.close(fig)
