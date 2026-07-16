# 그래프 텍스트 서식 심화 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 히트맵·3D 표면 서식에 요소별(라벨/눈금/제목) 굵기·기울기·색상과 `^{}`·`_{}` 위/아래첨자를 추가한다.

**Architecture:** `PlotConfig`에 요소별 9필드를 추가하고, plot.py에 폰트 dict를 만드는 헬퍼 2개(`_plotly_font_dict`, `_mpl_font_kw`)와 마크업 변환 순수 함수 `apply_text_markup` 1개를 도입한다. 렌더 함수 4개(plotly 2D/3D, matplotlib 2D/3D)의 인라인 font dict를 헬퍼 호출로 교체하고, 사용자 텍스트는 렌더 직전 마크업 변환한다. app.py는 세션 기본값·UI·config 빌더만 확장한다.

**Tech Stack:** Python, plotly 6.9, matplotlib 3.11, Streamlit, pytest.

## Global Constraints

- 인터프리터: `./.venv/Scripts/python.exe` (numpy 등은 여기에만 설치됨).
- 세션 서식 키는 반드시 `fmt_` 접두사 — `full_settings_dict`/`apply_settings_dict`가 `k.startswith("fmt_")`로 자동 저장/복원하므로 프리셋 코드는 건드리지 않는다.
- plotly 폰트: `weight`는 `"bold"`/`"normal"`, `style`는 `"italic"`/`"normal"` 문자열 사용(검증됨). matplotlib: `fontweight`/`fontstyle`/`color`.
- 마크업은 `^{`/`_{` 시퀀스만 특수 취급. 맨 `^`·`_`는 리터럴 유지.
- 마크업 변환은 렌더 직전 텍스트에만 적용. 세션에는 항상 원문 저장.
- 기존 전체 테스트 스위트(현재 42 passed) 무회귀.
- 커밋 전 브랜치 확인: 현재 `main`이면 작업 브랜치에서 진행(사용자 지시에 따름).

---

### Task 1: 마크업 변환 순수 함수

**Files:**
- Modify: `core/plot.py` (모듈 상단 헬퍼 영역에 함수 추가)
- Test: `tests/test_plot_format.py` (신규)

**Interfaces:**
- Produces: `apply_text_markup(text: str, target: str) -> str` — `target ∈ {"plotly","mpl"}`.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_plot_format.py`

```python
"""그래프 텍스트 서식(마크업 변환) 테스트."""
from core.plot import apply_text_markup


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

def test_none_text_safe():
    assert apply_text_markup("", "plotly") == ""
```

- [ ] **Step 2: 실패 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_plot_format.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_text_markup'`

- [ ] **Step 3: 구현** — `core/plot.py`, `import re`가 이미 있는지 확인 후 없으면 상단에 추가. `_plotly_font`/`_mpl_font` 등 저수준 헬퍼 근처에 삽입:

```python
def apply_text_markup(text: str, target: str) -> str:
    """'^{...}'/'_{...}' 마크업을 렌더러 문법으로 변환.

    target="plotly": '^{x}'->'<sup>x</sup>', '_{x}'->'<sub>x</sub>'
    target="mpl"   : '^{x}'->'$^{x}$',       '_{x}'->'$_{x}$'
    마크업이 없으면 원문을 그대로 반환한다(mpl에 불필요한 '$' 미삽입).
    미종료(예: 'cm^{-1')는 매칭되지 않아 원문 유지.
    """
    if not text:
        return text
    if target == "plotly":
        text = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", text)
        text = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", text)
    else:  # mpl
        text = re.sub(r"\^\{([^}]*)\}", r"$^{\1}$", text)
        text = re.sub(r"_\{([^}]*)\}", r"$_{\1}$", text)
    return text
```

- [ ] **Step 4: 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_plot_format.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: 커밋**

```bash
git add core/plot.py tests/test_plot_format.py
git commit -m "feat(plot): add ^{}/_{} superscript-subscript markup converter"
```

---

### Task 2: PlotConfig 요소별 서식 필드 + 폰트 dict 헬퍼

**Files:**
- Modify: `core/plot.py` (`PlotConfig` 데이터클래스 + 헬퍼 함수 추가)
- Test: `tests/test_plot_format.py`

**Interfaces:**
- Consumes: `apply_text_markup` (Task 1).
- Produces:
  - `PlotConfig` 새 필드: `font_bold_label/italic_label/color_label`,
    `font_bold_tick/italic_tick/color_tick`, `font_bold_title/italic_title/color_title`.
  - `_plotly_font_dict(cfg, elem) -> dict` — `elem ∈ {"label","tick","title"}`, 키: family,size,color,weight,style.
  - `_mpl_font_kw(cfg, elem) -> dict` — 키: fontsize,fontfamily,color,fontweight,fontstyle.

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_plot_format.py` 하단에

```python
from core.plot import PlotConfig, _plotly_font_dict, _mpl_font_kw

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
```

- [ ] **Step 2: 실패 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_plot_format.py -v`
Expected: FAIL — `ImportError`/`AttributeError` (필드·헬퍼 없음)

- [ ] **Step 3: 구현** — `PlotConfig`에 `font_size_title` 필드 바로 아래 추가:

```python
    font_bold_label: bool = False
    font_italic_label: bool = False
    font_color_label: str = "#000000"
    font_bold_tick: bool = False
    font_italic_tick: bool = False
    font_color_tick: str = "#000000"
    font_bold_title: bool = False
    font_italic_title: bool = False
    font_color_title: str = "#000000"
```

그리고 `_plotly_font`/`_mpl_font` 근처에 헬퍼 추가:

```python
_ELEM_SIZE_ATTR = {"label": "font_size_label", "tick": "font_size_tick",
                   "title": "font_size_title"}


def _plotly_font_dict(cfg: "PlotConfig", elem: str) -> dict:
    """요소(label/tick/title)별 plotly 폰트 dict."""
    return dict(
        family=_plotly_font(cfg.font_family),
        size=getattr(cfg, _ELEM_SIZE_ATTR[elem]),
        color=getattr(cfg, f"font_color_{elem}"),
        weight="bold" if getattr(cfg, f"font_bold_{elem}") else "normal",
        style="italic" if getattr(cfg, f"font_italic_{elem}") else "normal",
    )


def _mpl_font_kw(cfg: "PlotConfig", elem: str) -> dict:
    """요소별 matplotlib 텍스트 kwargs."""
    return dict(
        fontsize=getattr(cfg, _ELEM_SIZE_ATTR[elem]),
        fontfamily=_mpl_font(cfg.font_family),
        color=getattr(cfg, f"font_color_{elem}"),
        fontweight="bold" if getattr(cfg, f"font_bold_{elem}") else "normal",
        fontstyle="italic" if getattr(cfg, f"font_italic_{elem}") else "normal",
    )
```

- [ ] **Step 4: 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_plot_format.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: 커밋**

```bash
git add core/plot.py tests/test_plot_format.py
git commit -m "feat(plot): add per-element font style fields and font-dict helpers"
```

---

### Task 3: plotly 렌더러(2D/3D) 서식·마크업 반영

**Files:**
- Modify: `core/plot.py` — `make_heatmap`(≈328-431) 및 `make_surface`(≈593-653)
- Test: `tests/test_plot_format.py`

**Interfaces:**
- Consumes: `apply_text_markup`, `_plotly_font_dict` (Task 1·2).

- [ ] **Step 1: 실패 테스트 추가**

```python
import numpy as np
from core import plot

def test_heatmap_applies_title_style_and_markup():
    c = plot.PlotConfig(title="I^{2}", font_bold_title=True,
                        font_color_title="#123456")
    fig = plot.make_heatmap(np.arange(9.0).reshape(3, 3), c)
    t = fig.layout.title
    assert t.text == "I<sup>2</sup>"
    assert t.font.weight == "bold" and t.font.color == "#123456"

def test_heatmap_tickfont_color():
    c = plot.PlotConfig(font_color_tick="#00ff00", font_italic_tick=True)
    fig = plot.make_heatmap(np.arange(9.0).reshape(3, 3), c)
    assert fig.layout.xaxis.tickfont.color == "#00ff00"
    assert fig.layout.xaxis.tickfont.style == "italic"

def test_surface_axis_label_markup():
    c = plot.PlotConfig(x_label="d_{x}")
    fig = plot.make_surface(np.arange(9.0).reshape(3, 3), c)
    assert fig.layout.scene.xaxis.title.text == "d<sub>x</sub>"
```

- [ ] **Step 2: 실패 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_plot_format.py -k "heatmap or surface" -v`
Expected: FAIL (마크업·서식 미반영 → text/weight mismatch)

- [ ] **Step 3: 구현** — `make_heatmap`에서 교체:

colorbar dict(≈328-334):
```python
    colorbar = dict(
        title=dict(
            text=apply_text_markup(cfg.colorbar_label, "plotly"),
            font=_plotly_font_dict(cfg, "label"),
        ),
        tickfont=_plotly_font_dict(cfg, "tick"),
    )
```

xaxis/yaxis title·tickfont(≈390-406):
```python
    xaxis = dict(
        title=dict(text=apply_text_markup(cfg.x_label, "plotly"),
                   font=_plotly_font_dict(cfg, "label")),
        tickfont=_plotly_font_dict(cfg, "tick"),
        showticklabels=cfg.show_ticks,
        ticks="outside" if cfg.show_ticks else "",
        constrain="domain",
    )
    yaxis = dict(
        title=dict(text=apply_text_markup(cfg.y_label, "plotly"),
                   font=_plotly_font_dict(cfg, "label")),
        tickfont=_plotly_font_dict(cfg, "tick"),
        showticklabels=cfg.show_ticks,
        ticks="outside" if cfg.show_ticks else "",
        autorange="reversed",
    )
```

layout title(≈423-425):
```python
        title=dict(text=apply_text_markup(cfg.title, "plotly"),
                   font=_plotly_font_dict(cfg, "title")),
```

`make_surface`에서 교체 — surf_colorbar(≈601-605):
```python
    surf_colorbar = dict(
        title=dict(text=apply_text_markup(cfg.colorbar_label, "plotly"),
                   font=_plotly_font_dict(cfg, "label")),
        tickfont=_plotly_font_dict(cfg, "tick"),
    )
```

scene 축(≈621-631): `axis_font`/`tick_font` 지역변수를 제거하고 각 축에 헬퍼 사용 +
라벨 마크업:
```python
    scene = dict(
        xaxis=dict(title=dict(text=apply_text_markup(cfg.x_label, "plotly"),
                              font=_plotly_font_dict(cfg, "label")),
                   tickfont=_plotly_font_dict(cfg, "tick"),
                   showticklabels=cfg.show_ticks),
        yaxis=dict(title=dict(text=apply_text_markup(cfg.y_label, "plotly"),
                              font=_plotly_font_dict(cfg, "label")),
                   tickfont=_plotly_font_dict(cfg, "tick"),
                   showticklabels=cfg.show_ticks, autorange="reversed"),
        zaxis=dict(title=dict(text=apply_text_markup(cfg.colorbar_label, "plotly"),
                              font=_plotly_font_dict(cfg, "label")),
                   tickfont=_plotly_font_dict(cfg, "tick")),
        camera=dict(eye=camera_eye(cfg.cam_azim, cfg.cam_elev, cfg.cam_zoom)),
    )
```

layout title(≈648):
```python
        title=dict(text=apply_text_markup(cfg.title, "plotly"),
                   font=_plotly_font_dict(cfg, "title")),
```

- [ ] **Step 4: 통과 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_plot_format.py -v`
Expected: PASS (15 passed)

- [ ] **Step 5: 커밋**

```bash
git add core/plot.py tests/test_plot_format.py
git commit -m "feat(plot): apply per-element style and markup in plotly 2D/3D"
```

---

### Task 4: matplotlib 렌더러(2D/3D) 서식·마크업 반영

**Files:**
- Modify: `core/plot.py` — `make_matplotlib_heatmap`(≈514-539) 및 `make_matplotlib_surface`(≈703-717)
- Test: `tests/test_plot_format.py`

**Interfaces:**
- Consumes: `apply_text_markup`, `_mpl_font_kw` (Task 1·2).

- [ ] **Step 1: 실패 테스트 추가**

```python
import matplotlib
matplotlib.use("Agg")

def test_mpl_heatmap_builds_with_style():
    c = plot.PlotConfig(title="I^{2}", x_label="d_{x}",
                        font_bold_label=True, font_color_title="#ff0000",
                        font_italic_tick=True)
    fig = plot.make_matplotlib_heatmap(np.arange(9.0).reshape(3, 3), c, dpi=72)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "d$_{x}$"          # 마크업 변환
    assert ax.get_title() == "I$^{2}$"
    import matplotlib.pyplot as plt
    plt.close(fig)

def test_mpl_surface_builds_without_error():
    c = plot.PlotConfig(title="T_{c}", font_bold_title=True)
    fig = plot.make_matplotlib_surface(np.arange(9.0).reshape(3, 3), c, dpi=72)
    assert fig.axes[0].get_title() == "T$_{c}$"
    import matplotlib.pyplot as plt
    plt.close(fig)
```

- [ ] **Step 2: 실패 확인**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_plot_format.py -k mpl -v`
Expected: FAIL (라벨 텍스트가 원문 `d_{x}` 그대로 → mismatch)

- [ ] **Step 3: 구현** — `make_matplotlib_heatmap`에서 교체(≈514-539):

```python
    ax.set_xlabel(apply_text_markup(cfg.x_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    ax.set_ylabel(apply_text_markup(cfg.y_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    if cfg.title:
        ax.set_title(apply_text_markup(cfg.title, "mpl"), **_mpl_font_kw(cfg, "title"))

    tick_kw = _mpl_font_kw(cfg, "tick")
    ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(tick_kw["fontfamily"])
        lbl.set_fontweight(tick_kw["fontweight"])
        lbl.set_fontstyle(tick_kw["fontstyle"])
```

colorbar(≈537-539):
```python
    cbar.set_label(apply_text_markup(cfg.colorbar_label, "mpl"),
                   **_mpl_font_kw(cfg, "label"))
    cbar.ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
```

`make_matplotlib_surface`에서 교체(≈703-717):
```python
    ax.set_xlabel(apply_text_markup(cfg.x_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    ax.set_ylabel(apply_text_markup(cfg.y_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    ax.set_zlabel(apply_text_markup(cfg.colorbar_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    if cfg.title:
        ax.set_title(apply_text_markup(cfg.title, "mpl"), **_mpl_font_kw(cfg, "title"))
    tick_kw = _mpl_font_kw(cfg, "tick")
    ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
    ax.invert_yaxis()
    if not cfg.show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(apply_text_markup(cfg.colorbar_label, "mpl"),
                   **_mpl_font_kw(cfg, "label"))
    cbar.ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
```

- [ ] **Step 4: 통과 확인 + 전체 무회귀**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — 전체 그린(기존 42 + 신규 ≈17).

- [ ] **Step 5: 커밋**

```bash
git add core/plot.py tests/test_plot_format.py
git commit -m "feat(plot): apply per-element style and markup in matplotlib 2D/3D"
```

---

### Task 5: app.py — 기본값·UI·config 빌더

**Files:**
- Modify: `app.py` — `DEFAULTS`(≈106), `build_plot_config`(≈545-565), 서식 UI(≈1139-1143)

**Interfaces:**
- Consumes: `PlotConfig` 새 필드(Task 2).

- [ ] **Step 1: DEFAULTS 확장** — `app.py` `DEFAULTS`의 `"fmt_fs_title": 16,` 뒤에 추가:

```python
    "fmt_bold_label": False, "fmt_italic_label": False, "fmt_color_label": "#000000",
    "fmt_bold_tick": False, "fmt_italic_tick": False, "fmt_color_tick": "#000000",
    "fmt_bold_title": False, "fmt_italic_title": False, "fmt_color_title": "#000000",
```

- [ ] **Step 2: build_plot_config 전달** — `PlotConfig(...)` 호출의
`font_size_title=int(ss.fmt_fs_title),` 뒤에 추가:

```python
        font_bold_label=ss.fmt_bold_label, font_italic_label=ss.fmt_italic_label,
        font_color_label=ss.fmt_color_label,
        font_bold_tick=ss.fmt_bold_tick, font_italic_tick=ss.fmt_italic_tick,
        font_color_tick=ss.fmt_color_tick,
        font_bold_title=ss.fmt_bold_title, font_italic_title=ss.fmt_italic_title,
        font_color_title=ss.fmt_color_title,
```

- [ ] **Step 3: UI 교체** — 서식 블록(≈1140-1143)의 크기 3열(fs1/fs2/fs3)을 요소별 3행으로:

```python
            st.selectbox("폰트", FONTS, key="fmt_font")
            st.caption("위첨자 `^{ }`, 아래첨자 `_{ }`  예: `cm^{-1}`, `µm_{2}`")
            for label, sz, bd, it, col in [
                ("라벨", "fmt_fs_label", "fmt_bold_label", "fmt_italic_label", "fmt_color_label"),
                ("눈금", "fmt_fs_tick", "fmt_bold_tick", "fmt_italic_tick", "fmt_color_tick"),
                ("제목", "fmt_fs_title", "fmt_bold_title", "fmt_italic_title", "fmt_color_title"),
            ]:
                r = st.columns([1.1, 1.2, 0.7, 0.7, 1.1])
                r[0].markdown(f"**{label}**")
                r[1].number_input("크기", min_value=6, max_value=40, key=sz,
                                  label_visibility="collapsed")
                r[2].checkbox("B", key=bd)
                r[3].checkbox("I", key=it)
                r[4].color_picker("색", key=col, label_visibility="collapsed")
```

- [ ] **Step 4: 임포트·기본값 정합성 확인**

Run: `./.venv/Scripts/python.exe -c "import ast; ast.parse(open('app.py',encoding='utf-8').read()); print('app.py parse OK')"`
Expected: `app.py parse OK`

Run: `./.venv/Scripts/python.exe -c "import app; ks=[k for k in app.DEFAULTS if k.startswith('fmt_')]; assert all(k in app.DEFAULTS for k in ['fmt_bold_label','fmt_color_title','fmt_italic_tick']); print('DEFAULTS keys OK', len(ks))"`
Expected: `DEFAULTS keys OK ...` (예외 없이)

- [ ] **Step 5: 앱 실행 검증(verify 스킬)** — Streamlit 앱을 띄워 "⑤ 히트맵 서식"에서
라벨/눈금/제목 각 행의 B·I·색 위젯이 보이고, 제목에 `T^{2}` 입력 시 2D·3D·내보내기에서
위첨자로 렌더되는지 육안 확인. (프리셋 저장→불러오기 왕복 시 B/I/색 유지 확인.)

- [ ] **Step 6: 커밋**

```bash
git add app.py
git commit -m "feat(ui): per-element bold/italic/color font controls with markup hint"
```

---

## Self-Review

**Spec coverage:**
- 요소별 B/I/색 → Task 2(필드·헬퍼) + Task 3·4(렌더) + Task 5(UI). ✅
- 위/아래첨자 마크업 → Task 1 + 렌더 적용 Task 3·4, UI 힌트 Task 5. ✅
- 2D/3D/컬러바/mpl 폴백 전면 적용 → Task 3(plotly 2D/3D+컬러바), Task 4(mpl 2D/3D+컬러바). ✅
- 프리셋 자동 포함 → `fmt_` 접두사(Task 5) + 무변경, Task 5 Step 5 왕복 확인. ✅
- 무회귀 → Task 4 Step 4 전체 스위트. ✅

**Placeholder scan:** 모든 코드 스텝에 실제 코드 포함. TBD/TODO 없음. ✅

**Type consistency:** `apply_text_markup(text,target)`, `_plotly_font_dict(cfg,elem)`,
`_mpl_font_kw(cfg,elem)`, 필드명 `font_{bold,italic,color}_{label,tick,title}`가
모든 태스크에서 일관. UI 세션키 `fmt_{bold,italic,color}_{...}` ↔ build_plot_config ↔
DEFAULTS 동일. ✅
