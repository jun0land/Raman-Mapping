"""figure 생성 (순수 코어 모듈, Step 5 시각화 + 스펙트럼 뷰어).

Plotly(인터랙티브) + Matplotlib(논문용 고해상도 export) figure를 생성한다.
figure 객체를 **반환만** 하며, show/save는 하지 않는다(앱이 다운로드/kaleido
export 처리). Streamlit 등 UI 의존성 없음. 순수 함수. 입력 grid는 변형하지 않는다.

그리드 규약: grid는 np.ndarray shape (ny, nx), grid[y, x]. row 0 = 이미지 상단(TOP).
파수 cm⁻¹, 좌표 µm.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from math import cos, radians, sin
from typing import Optional

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # UI/디스플레이 백엔드 없이 figure만 생성
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# 번들 폰트 등록 (Pretendard, SIL OFL — 자유 재배포 가능)
# ---------------------------------------------------------------------------
# 클라이언트/배포 서버에 설치돼 있지 않아도 matplotlib export(PNG/SVG/PDF)에서
# 항상 동일하게 렌더되도록, static/fonts/ 의 정적 폰트 파일을 폰트매니저에 등록한다.
# (Myriad Pro 는 Adobe 상업 폰트라 재배포 라이선스가 없어 번들링하지 않는다 —
#  클라이언트에 실제로 설치돼 있을 때만 시스템 폰트로 렌더되고, 그 외엔
#  _mpl_font()의 폴백 체인으로 조용히 대체된다.)
_FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"
for _f in ("Pretendard-Regular.otf", "Pretendard-Bold.otf"):
    _p = _FONT_DIR / _f
    if _p.exists():
        try:
            matplotlib.font_manager.fontManager.addfont(str(_p))
        except Exception:
            pass  # 폰트 등록 실패는 치명적이지 않음 — 폴백 체인이 처리

__all__ = [
    "PlotConfig",
    "auto_zrange",
    "make_heatmap",
    "add_click_grid",
    "make_matplotlib_heatmap",
    "make_surface",
    "make_matplotlib_surface",
    "camera_eye",
    "make_spectrum_figure",
    "mpl_to_plotly_colorscale",
    "PLOTLY_CMAP",
    "MPL_CMAP",
]


# ---------------------------------------------------------------------------
# Colormap 이름 매핑 (한 곳에서 관리)
# 지원: jet, viridis, plasma, inferno, rainbow, gray, RdBu
# ---------------------------------------------------------------------------
# NOTE: 렌더링에는 더 이상 사용하지 않는다. Plotly colorscale은 matplotlib을
# 단일 진리 원천으로 삼아 mpl_to_plotly_colorscale()로 생성한다(색 불일치 버그 수정).
# 하위 호환(외부 import 가능성)을 위해 심볼만 유지한다.
PLOTLY_CMAP: dict[str, str] = {
    "jet": "Jet",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "rainbow": "Rainbow",
    "gray": "Greys",
    "grey": "Greys",
    "rdbu": "RdBu",
}

MPL_CMAP: dict[str, str] = {
    "jet": "jet",
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "rainbow": "rainbow",       # matplotlib 내장 'rainbow'
    "gray": "gray",
    "grey": "gray",
    "rdbu": "RdBu",
}


def _mpl_cmap(name: str) -> str:
    return MPL_CMAP.get(str(name).lower(), "jet")


@functools.lru_cache(maxsize=64)
def mpl_to_plotly_colorscale(cmap_name: str, n: int = 256) -> list[list]:
    """matplotlib colormap을 샘플링해 Plotly colorscale(list)로 변환한다.

    matplotlib을 색상의 **단일 진리 원천(single source of truth)**으로 삼아,
    Plotly figure와 Matplotlib figure가 동일한 colormap 이름에 대해 픽셀 단위로
    동일한 색을 쓰도록 보장한다.

    성능: (cmap_name, n) 은 hashable 하므로 functools.lru_cache 로 결과 list 를
    캐싱한다(256-샘플 스케일은 colormap 당 한 번만 계산 → 매 rerun 재사용).
    호출부(make_heatmap/make_surface)는 colorscale 을 **읽기만** 하고 변형하지
    않으므로 캐시된 동일 list 를 그대로 반환해도 안전하다.

    Args:
        cmap_name: 사용자 colormap 이름(jet|viridis|plasma|inferno|rainbow|gray|RdBu).
                   MPL_CMAP으로 해석하며, 미지원 이름은 "jet"으로 폴백.
        n: 샘플 개수(기본 256, 부드러운 스케일).

    Returns:
        Plotly colorscale: [[i/(n-1), "rgb(r,g,b)"], ...] (r,g,b는 0-255 정수).
    """
    cmap = matplotlib.colormaps[_mpl_cmap(cmap_name)]
    n = max(int(n), 2)
    scale: list[list] = []
    for i in range(n):
        t = i / (n - 1)
        r, g, b, _a = cmap(t)
        scale.append([
            t,
            f"rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})",
        ])
    return scale


def _plotly_colorscale(name: str) -> list[list]:
    """Plotly figure용 colorscale. matplotlib colormap을 샘플링한 list를 반환한다."""
    return mpl_to_plotly_colorscale(name)


# ---------------------------------------------------------------------------
# 폰트 helper — 웹 세이프가 아닌 폰트(Myriad Pro 등)에 폴백 체인을 부여
# ---------------------------------------------------------------------------
def _plotly_font(name: Optional[str]) -> str:
    """Plotly font.family 문자열(폴백 포함).

    Pretendard 는 static/fonts 에 번들되어 @font-face 로 항상 로드되므로(app.py),
    Myriad Pro 등 클라이언트에 없을 수 있는 폰트 다음 폴백으로 Arial 대신
    Pretendard 를 먼저 넣는다 — 어떤 환경에서도 시스템 기본 sans-serif 가 아닌
    일관된 모양으로 렌더된다.
    """
    n = str(name or "").strip()
    if not n:
        return "Pretendard, Arial, Helvetica, sans-serif"
    if "," in n:  # 이미 폴백 체인이면 그대로
        return n
    if n == "Pretendard":
        return "Pretendard, Arial, Helvetica, sans-serif"
    return f"{n}, Pretendard, Arial, Helvetica, sans-serif"


def _mpl_font(name: Optional[str]) -> list[str]:
    """Matplotlib fontfamily 폴백 리스트(설치 안 된 폰트도 예외 없이 폴백).

    Pretendard 는 폰트매니저에 번들 등록되어(core/plot.py 상단) 항상 사용 가능하므로,
    지정 폰트가 없거나 미설치일 때 시스템 sans-serif 대신 Pretendard 로 먼저 떨어진다.
    """
    n = str(name or "").strip()
    fam = [n] if n else []
    for fb in ("Pretendard", "Arial", "DejaVu Sans", "sans-serif"):
        if fb not in fam:
            fam.append(fb)
    return fam


def apply_text_markup(text: str, target: str) -> str:
    """텍스트 서식 마크업을 렌더러 문법으로 변환(부분 서식 지원).

    지원 마크업 — 각각 '{...}' 로 감싼 부분에만 적용:
      '^{x}' 위첨자 · '_{x}' 아래첨자 · '*{x}' 볼드 · '/{x}' 이탤릭

    target="plotly": <sup>/<sub>/<b>/<i> 태그로 변환(화면·kaleido export 완벽).
    target="mpl"   : mathtext($^{}$ / $_{}$ / $\\mathbf{}$ / $\\mathit{}$)로 변환.
                     (폴백 전용. 볼드/이탤릭은 수식 폰트로 렌더돼 약간 다를 수 있음.)
    마크업이 없으면 원문 그대로 반환. 미종료(예: 'cm^{-1')는 매칭 안 돼 원문 유지.
    맨 '^'·'_'·'*'·'/'는 리터럴. 마크업 중첩은 미지원.
    """
    if not text:
        return text
    if target == "plotly":
        text = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", text)
        text = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", text)
        text = re.sub(r"\*\{([^}]*)\}", r"<b>\1</b>", text)
        text = re.sub(r"/\{([^}]*)\}", r"<i>\1</i>", text)
    else:  # mpl (mathtext) — 볼드/이탤릭 내부 공백은 '\ '로 보존
        text = re.sub(r"\^\{([^}]*)\}", r"$^{\1}$", text)
        text = re.sub(r"_\{([^}]*)\}", r"$_{\1}$", text)
        text = re.sub(r"\*\{([^}]*)\}",
                      lambda m: "$\\mathbf{" + m.group(1).replace(" ", "\\ ") + "}$", text)
        text = re.sub(r"/\{([^}]*)\}",
                      lambda m: "$\\mathit{" + m.group(1).replace(" ", "\\ ") + "}$", text)
    return text


# 요소(label/tick/title)별 크기 속성 이름
_ELEM_SIZE_ATTR = {"label": "font_size_label", "tick": "font_size_tick",
                   "title": "font_size_title"}


def _plotly_font_dict(cfg: "PlotConfig", elem: str) -> dict:
    """요소(label/tick/title)별 plotly 폰트 dict (family,size,color,weight,style)."""
    return dict(
        family=_plotly_font(cfg.font_family),
        size=getattr(cfg, _ELEM_SIZE_ATTR[elem]),
        color=getattr(cfg, f"font_color_{elem}"),
        weight="bold" if getattr(cfg, f"font_bold_{elem}") else "normal",
        style="italic" if getattr(cfg, f"font_italic_{elem}") else "normal",
    )


# 제목 위치 → plotly title.x 값
_TITLE_X = {"left": 0.0, "center": 0.5, "right": 1.0}


def _z_label(cfg: "PlotConfig") -> str:
    """3D Z축 라벨. 지정이 없으면 colorbar_label 로 폴백(기존 동작)."""
    return cfg.z_label if str(cfg.z_label).strip() else cfg.colorbar_label


def _title_pos(cfg: "PlotConfig") -> str:
    """제목 위치 정규화. 알 수 없는 값은 center."""
    pos = str(getattr(cfg, "title_pos", "center")).lower()
    return pos if pos in _TITLE_X else "center"


def _mpl_font_kw(cfg: "PlotConfig", elem: str) -> dict:
    """요소별 matplotlib 텍스트 kwargs (fontsize,fontfamily,color,fontweight,fontstyle)."""
    return dict(
        fontsize=getattr(cfg, _ELEM_SIZE_ATTR[elem]),
        fontfamily=_mpl_font(cfg.font_family),
        color=getattr(cfg, f"font_color_{elem}"),
        fontweight="bold" if getattr(cfg, f"font_bold_{elem}") else "normal",
        fontstyle="italic" if getattr(cfg, f"font_italic_{elem}") else "normal",
    )


# ---------------------------------------------------------------------------
# PlotConfig
# ---------------------------------------------------------------------------
@dataclass
class PlotConfig:
    """히트맵 서식 설정 (모든 필드 기본값 제공).

    Colormap / z-range:
        colormap: jet|viridis|plasma|inferno|rainbow|gray|RdBu (기본 "jet").
        zmin, zmax: colorbar 범위. None이면 자동(데이터 min/max, Plotly 위임).

    축 / 물리 좌표:
        x_label, y_label: 축 이름 (기본 "X (μm)", "Y (μm)").
        title: 그래프 제목.
        step_x, step_y: 픽셀 간격(µm). 픽셀 인덱스를 물리 좌표로 변환.
        x0, y0: 좌표 원점(µm).
        show_ticks: 눈금 표시 여부.
        tick_spacing: 눈금 간격(µm). None이면 자동.

    Colorbar (Z축):
        colorbar_label: z 라벨 (예: "Intensity (a.u.)", "I_D/I_G").
        colorbar_ticks: colorbar 눈금 개수.

    폰트:
        font_family: "Arial", "Times New Roman" 등.
        font_size_label / font_size_tick / font_size_title: 각각 크기(pt).

    렌더링:
        interpolation: "none"(픽셀) | "bilinear"(부드럽게).
        lock_aspect: 1:1 종횡비 고정 여부.
        fill_mode: 2D 맵 채우기 방식. "pixel"(go.Heatmap 격자 히트맵, 기본) |
                   "contour"(go.Contour 등고선 채움, Origin 스타일 컬러 컨투어).
                   3D(make_surface)에는 영향을 주지 않는다. "contour"일 때
                   interpolation(zsmooth)은 무시된다.

    3D 표면 카메라 (make_surface 전용):
        cam_azim: 방위각(degrees). XY 평면에서의 회전 각도. 기본 -45.0.
        cam_elev: 고도(degrees). XY 평면 위로 올려다보는 각도(0=수평, 90=바로 위).
                  기본 25.0.
        cam_zoom: 카메라 거리(반지름 r). 클수록 더 멀리서(더 축소) 본다. 기본 2.2.
                  make_surface 는 이 구면 좌표로부터 Plotly scene.camera.eye 를 계산한다:
                  r=cam_zoom, az=radians(cam_azim), el=radians(cam_elev),
                  eye=(r·cos(el)·cos(az), r·cos(el)·sin(az), r·sin(el)).
                  마우스 드래그 회전은 클라이언트 전용이라 서버로 전달되지 않으므로,
                  export(kaleido)와 화면 각도를 일치시키려면 이 필드를 사용한다.
    """

    # colormap / z-range
    colormap: str = "jet"
    zmin: Optional[float] = None
    zmax: Optional[float] = None

    # 축 / 물리 좌표
    x_label: str = "X (μm)"
    y_label: str = "Y (μm)"
    # 3D 표면 Z축 라벨. 빈 문자열이면 colorbar_label 을 따른다(기존 동작 유지).
    z_label: str = ""
    title: str = ""
    step_x: float = 1.0
    step_y: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    show_ticks: bool = True
    tick_spacing: Optional[float] = None

    # colorbar
    colorbar_label: str = "Intensity (a.u.)"
    colorbar_ticks: int = 5

    # 폰트
    font_family: str = "Arial"
    font_size_label: int = 14
    font_size_tick: int = 12
    font_size_title: int = 16
    # 요소별 굵기/기울기/색 (Origin 스타일)
    font_bold_label: bool = False
    font_italic_label: bool = False
    font_color_label: str = "#000000"
    font_bold_tick: bool = False
    font_italic_tick: bool = False
    font_color_tick: str = "#000000"
    font_bold_title: bool = False
    font_italic_title: bool = False
    font_color_title: str = "#000000"
    # 제목 위치 (Origin 스타일): "left" | "center" | "right"
    title_pos: str = "center"

    # 렌더링
    interpolation: str = "none"
    lock_aspect: bool = True
    fill_mode: str = "pixel"  # "pixel"(go.Heatmap) | "contour"(go.Contour)
    show_contour_lines: bool = True  # contour 모드일 때 등고선 라인 오버레이 표시

    # 3D 표면 카메라 (구면 좌표 → Plotly scene.camera.eye)
    cam_azim: float = -45.0
    cam_elev: float = 25.0
    cam_zoom: float = 2.2


# ---------------------------------------------------------------------------
# z-range helper
# ---------------------------------------------------------------------------
def auto_zrange(grid: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    """grid의 percentile 기반 z-range (p_low, p_high) 반환.

    Args:
        grid: (ny, nx) 매핑 값 배열.
        low: 하위 percentile (기본 2).
        high: 상위 percentile (기본 98).

    Returns:
        (low_val, high_val) 튜플. NaN은 무시. 값이 모두 동일/NaN이면 방어.
    """
    arr = np.asarray(grid, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


# ---------------------------------------------------------------------------
# 물리 좌표축 helper
# ---------------------------------------------------------------------------
def _axis_coords(n: int, step: float, origin: float) -> np.ndarray:
    """픽셀 개수 n을 물리 좌표(µm) 중심 좌표 배열로 변환."""
    return origin + np.arange(n, dtype=float) * float(step)


def _pad_for_contour(
    arr: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
    step_x: float, step_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """등고선 라인 계산용으로 그리드 가장자리에 1픽셀 테두리를 복제 확장한다.

    go.Contour/ax.contour는 등고선을 그리드 정점(픽셀 중심) 사이에서만 보간하므로
    도메인이 [x_coords[0], x_coords[-1]]에서 끝난다. 반면 go.Heatmap·imshow 채움은
    셀 경계(중심 ± step/2)까지 반 픽셀씩 더 뻗어 나간다. 두 도메인이 어긋나면
    채움은 캔버스 테두리까지 닿는데 등고선 라인만 테두리 쪽 반 픽셀 폭만큼 비어
    보이는 문제가 생긴다. 가장자리 값을 그대로 복제해 셀 경계 좌표까지 확장하면
    등고선 계산 도메인이 채움 도메인과 일치해 테두리까지 라인이 그려진다.
    """
    padded_z = np.pad(arr, pad_width=1, mode="edge")
    padded_x = np.concatenate((
        [x_coords[0] - 0.5 * step_x], x_coords, [x_coords[-1] + 0.5 * step_x],
    ))
    padded_y = np.concatenate((
        [y_coords[0] - 0.5 * step_y], y_coords, [y_coords[-1] + 0.5 * step_y],
    ))
    return padded_z, padded_x, padded_y


def _tickvals(coords: np.ndarray, spacing: Optional[float]) -> Optional[list[float]]:
    """지정 간격(µm)에 맞춘 눈금 좌표 목록. spacing None이면 None(자동)."""
    if spacing is None or spacing <= 0 or coords.size == 0:
        return None
    lo, hi = float(coords.min()), float(coords.max())
    start = np.ceil(lo / spacing) * spacing
    ticks = np.arange(start, hi + spacing * 0.5, spacing)
    return [float(t) for t in ticks]


def _colorbar_ticks(zmin: Optional[float], zmax: Optional[float], n: int) -> dict:
    """Plotly colorbar 에 병합할 눈금 설정 dict 를 반환한다.

    Plotly 의 ``colorbar.nticks`` 는 최댓값 힌트일 뿐이고 "예쁜 숫자"로 반올림하기
    때문에 사용자가 요청한 정확한 개수가 반영되지 않는다. 유한한 (zmin, zmax)
    범위가 주어지면 ``np.linspace`` 로 정확히 n 개의 등간격 눈금(tickvals)을
    직접 지정해 요청 개수를 그대로 반영한다.

    Args:
        zmin, zmax: colorbar 값 범위. None/비유한/degenerate 면 폴백.
        n: 원하는 눈금 개수(>=2).

    Returns:
        colorbar config 에 병합할 dict. 유효 범위면
        ``dict(tickmode="array", tickvals=[...], nticks=n, tickformat=".3g")``,
        아니면 ``dict(nticks=n)``.
    """
    n = max(int(n), 2)
    finite = (
        zmin is not None and zmax is not None
        and np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin
    )
    if finite:
        tickvals = [float(v) for v in np.linspace(float(zmin), float(zmax), n)]
        return dict(tickmode="array", tickvals=tickvals, nticks=n, tickformat=".3g")
    return dict(nticks=n)


# ---------------------------------------------------------------------------
# Plotly heatmap
# ---------------------------------------------------------------------------
def make_heatmap(grid: np.ndarray, config: Optional[PlotConfig] = None) -> go.Figure:
    """go.Heatmap 기반 인터랙티브 히트맵 figure 반환.

    물리 좌표(step_x/y, x0/y0)를 축에 적용하고, colorbar 라벨/눈금 개수,
    폰트, zmin/zmax, zsmooth, 1:1 종횡비, row 0 = TOP(이미지 규약)을 반영한다.
    데이터는 그대로 두어 앱의 클릭 이벤트 연결이 가능하다.

    Args:
        grid: (ny, nx) 매핑 값 배열. grid[y, x], row 0 = 상단.
        config: PlotConfig. None이면 기본값.

    Returns:
        plotly.graph_objects.Figure.
    """
    cfg = config or PlotConfig()
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"grid는 (ny, nx) 2차원 배열이어야 합니다. 현재 차원: {arr.ndim}")
    ny, nx = arr.shape

    x_coords = _axis_coords(nx, cfg.step_x, cfg.x0)
    y_coords = _axis_coords(ny, cfg.step_y, cfg.y0)

    zsmooth = "best" if str(cfg.interpolation).lower() in ("bilinear", "best") else False

    colorscale = _plotly_colorscale(cfg.colormap)

    # 유효 z-range 결정(cfg 우선, 없으면 데이터 min/max). 이 범위로 colorbar 눈금
    # 개수를 정확히 지정한다(픽셀·컨투어 동일).
    zmin_eff = cfg.zmin if cfg.zmin is not None else float(np.nanmin(arr))
    zmax_eff = cfg.zmax if cfg.zmax is not None else float(np.nanmax(arr))
    if not (np.isfinite(zmin_eff) and np.isfinite(zmax_eff) and zmax_eff > zmin_eff):
        zmin_eff = zmax_eff = None

    colorbar = dict(
        title=dict(
            text=apply_text_markup(cfg.colorbar_label, "plotly"),
            font=_plotly_font_dict(cfg, "label"),
        ),
        tickfont=_plotly_font_dict(cfg, "tick"),
    )
    colorbar.update(_colorbar_ticks(zmin_eff, zmax_eff, int(cfg.colorbar_ticks)))

    if str(cfg.fill_mode).lower() == "contour":
        # Origin 스타일 컬러 컨투어 맵. 단일 go.Contour(contours_coloring="heatmap")
        # 는 colorbar 를 등고선 경계(밴드)로 그려 "계단식/분절"로 보이는 문제가 있다.
        # 픽셀 go.Heatmap·3D go.Surface 의 colorbar 와 동일하게 "연속"으로 만들기
        # 위해 두 개의 trace 로 구성한다:
        #   (1) go.Heatmap: 연속 색 채움(zsmooth="best")을 담당하고 연속 colorbar 를
        #       소유(showscale=True, 픽셀 브랜치와 동일한 colorbar 설정). 정확한 눈금
        #       개수의 매끄러운 그라데이션 colorbar 가 된다.
        #   (2) go.Contour: 등고선 "라인"만(contours_coloring="lines") 그린다. 채움도
        #       colorbar 도 없이(showscale=False) 반투명 어두운 라인을 위에 겹쳐
        #       컨투어 맵의 성격을 유지한다.
        heatmap_fill = go.Heatmap(
            z=arr,
            x=x_coords,
            y=y_coords,
            colorscale=colorscale,
            zmin=zmin_eff,
            zmax=zmax_eff,
            zsmooth="best",
            showscale=True,
            colorbar=colorbar,
            hovertemplate="X=%{x}<br>Y=%{y}<br>z=%{z}<extra></extra>",
        )
        padded_z, padded_x, padded_y = _pad_for_contour(
            arr, x_coords, y_coords, cfg.step_x, cfg.step_y)
        contour_lines = go.Contour(
            z=padded_z,
            x=padded_x,
            y=padded_y,
            zmin=zmin_eff,
            zmax=zmax_eff,
            contours_coloring="lines",
            ncontours=24,
            line=dict(color="rgba(0,0,0,0.35)", width=0.5),
            showscale=False,
            hoverinfo="skip",
        )
        # 라인이 채움 위에 오도록 heatmap 을 먼저, contour 를 나중에 추가한다.
        # show_contour_lines=False 면 라인 오버레이 없이 연속 채움만.
        trace = [heatmap_fill]
        if cfg.show_contour_lines:
            trace.append(contour_lines)
    else:
        trace = go.Heatmap(
            z=arr,
            x=x_coords,
            y=y_coords,
            colorscale=colorscale,
            zmin=cfg.zmin,
            zmax=cfg.zmax,
            zsmooth=zsmooth,
            colorbar=colorbar,
            hovertemplate="X=%{x}<br>Y=%{y}<br>z=%{z}<extra></extra>",
        )

    fig = go.Figure(data=trace)

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
        # row 0 이 상단에 오도록 y축 역방향 (이미지 규약)
        autorange="reversed",
    )

    # 지정 눈금 간격
    xt = _tickvals(x_coords, cfg.tick_spacing)
    if xt is not None:
        xaxis["tickmode"] = "array"
        xaxis["tickvals"] = xt
    yt = _tickvals(y_coords, cfg.tick_spacing)
    if yt is not None:
        yaxis["tickmode"] = "array"
        yaxis["tickvals"] = yt

    # 1:1 종횡비 고정: y축을 x축에 앵커
    if cfg.lock_aspect:
        yaxis["scaleanchor"] = "x"
        yaxis["scaleratio"] = 1

    fig.update_layout(
        title=dict(text=apply_text_markup(cfg.title, "plotly"),
                   font=_plotly_font_dict(cfg, "title"),
                   x=_TITLE_X[_title_pos(cfg)], xanchor=_title_pos(cfg)),
        xaxis=xaxis,
        yaxis=yaxis,
        font=dict(family=_plotly_font(cfg.font_family)),
        margin=dict(l=70, r=70, t=60 if cfg.title else 30, b=60),
        template="plotly_white",
    )
    return fig


def add_click_grid(fig: "go.Figure", ny: int, nx: int,
                   config: Optional[PlotConfig] = None) -> None:
    """2D 히트맵 위에 투명 클릭 타깃 산점 레이어를 추가한다 (in-place).

    plotly 의 Heatmap 트레이스는 클릭/박스 선택(selection) 이벤트를 지원하지 않아
    st.plotly_chart(on_select=...) 로는 픽셀 클릭이 잡히지 않는다. 픽셀 중심마다
    완전 투명한 마커를 깔아 두면 클릭이 산점 포인트로 잡혀 (x, y)가 이벤트에 실린다.
    hover 는 숨겨 히트맵 자체 hovertemplate 를 방해하지 않는다.
    """
    cfg = config or PlotConfig()
    xs = _axis_coords(nx, cfg.step_x, cfg.x0)
    ys = _axis_coords(ny, cfg.step_y, cfg.y0)
    X, Y = np.meshgrid(xs, ys)
    fig.add_trace(go.Scattergl(
        x=X.ravel(), y=Y.ravel(), mode="markers",
        marker=dict(size=14, opacity=0),
        hoverinfo="skip", showlegend=False, name="_click_targets",
    ))


# ---------------------------------------------------------------------------
# Matplotlib heatmap (publication quality)
# ---------------------------------------------------------------------------
def make_matplotlib_heatmap(
    grid: np.ndarray,
    config: Optional[PlotConfig] = None,
    dpi: int = 300,
) -> Figure:
    """논문용 고해상도 히트맵 (imshow). Figure 반환(앱이 PNG/SVG/PDF 저장).

    origin='upper'로 row 0을 상단에 두고, step_x/y로부터 extent를 계산해
    물리 좌표축을 만든다. colorbar 라벨, 폰트/크기, cmap 매핑 적용.

    Args:
        grid: (ny, nx) 매핑 값 배열. grid[y, x], row 0 = 상단.
        config: PlotConfig. None이면 기본값.
        dpi: 해상도. 기본 300.

    Returns:
        matplotlib.figure.Figure.
    """
    cfg = config or PlotConfig()
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"grid는 (ny, nx) 2차원 배열이어야 합니다. 현재 차원: {arr.ndim}")
    ny, nx = arr.shape

    # extent: (left, right, bottom, top). origin='upper'이므로 top이 y0.
    # 픽셀 중심 좌표 기준으로 반 픽셀씩 확장.
    left = cfg.x0 - 0.5 * cfg.step_x
    right = cfg.x0 + (nx - 0.5) * cfg.step_x
    top = cfg.y0 - 0.5 * cfg.step_y
    bottom = cfg.y0 + (ny - 0.5) * cfg.step_y
    extent = (left, right, bottom, top)

    interp = "bilinear" if str(cfg.interpolation).lower() in ("bilinear", "best") else "none"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    if str(cfg.fill_mode).lower() == "contour":
        # Origin 스타일 컬러 컨투어 맵. 화면(Plotly: 연속 Heatmap 채움 + 등고선 라인)과
        # 일치시키기 위해, 픽셀 경로와 동일한 imshow(origin='upper', bilinear)로 "연속"
        # 채움을 그리고(이 mappable 이 연속 colorbar 를 소유), 그 위에 얇은 등고선 라인을
        # 겹친다. imshow 는 extent/origin 규약(row 0 = 상단)을 그대로 따르므로 라인도
        # 동일 물리 좌표로 그리면 채움과 정렬된다.
        im = ax.imshow(
            arr,
            origin="upper",
            cmap=_mpl_cmap(cfg.colormap),
            vmin=cfg.zmin,
            vmax=cfg.zmax,
            extent=extent,
            interpolation="bilinear",
            aspect="equal" if cfg.lock_aspect else "auto",
        )
        if cfg.show_contour_lines:
            try:
                xc = _axis_coords(nx, cfg.step_x, cfg.x0)
                yc = _axis_coords(ny, cfg.step_y, cfg.y0)
                padded_arr, padded_xc, padded_yc = _pad_for_contour(
                    arr, xc, yc, cfg.step_x, cfg.step_y)
                X, Y = np.meshgrid(padded_xc, padded_yc)
                ax.contour(
                    X, Y, padded_arr,
                    levels=12,
                    colors="k",
                    linewidths=0.4,
                    alpha=0.35,
                )
            except Exception:
                pass
    else:
        im = ax.imshow(
            arr,
            origin="upper",
            cmap=_mpl_cmap(cfg.colormap),
            vmin=cfg.zmin,
            vmax=cfg.zmax,
            extent=extent,
            interpolation=interp,
            aspect="equal" if cfg.lock_aspect else "auto",
        )

    ax.set_xlabel(apply_text_markup(cfg.x_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    ax.set_ylabel(apply_text_markup(cfg.y_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    if cfg.title:
        ax.set_title(apply_text_markup(cfg.title, "mpl"), loc=_title_pos(cfg),
                     **_mpl_font_kw(cfg, "title"))

    tick_kw = _mpl_font_kw(cfg, "tick")
    ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(tick_kw["fontfamily"])
        lbl.set_fontweight(tick_kw["fontweight"])
        lbl.set_fontstyle(tick_kw["fontstyle"])

    if not cfg.show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    elif cfg.tick_spacing:
        xc = _axis_coords(nx, cfg.step_x, cfg.x0)
        yc = _axis_coords(ny, cfg.step_y, cfg.y0)
        xt = _tickvals(xc, cfg.tick_spacing)
        yt = _tickvals(yc, cfg.tick_spacing)
        if xt:
            ax.set_xticks(xt)
        if yt:
            ax.set_yticks(yt)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(apply_text_markup(cfg.colorbar_label, "mpl"),
                   **_mpl_font_kw(cfg, "label"))
    cbar.ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
    # colorbar 눈금 개수(정확한 등간격) — plotly 와 일치하도록 LinearLocator 사용.
    try:
        from matplotlib.ticker import LinearLocator
        cbar.locator = LinearLocator(numticks=int(cfg.colorbar_ticks))
        cbar.update_ticks()
    except Exception:
        pass

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D color-map surface (Origin 스타일) — Plotly (인터랙티브)
# ---------------------------------------------------------------------------
def camera_eye(azim: float, elev: float, zoom: float) -> dict:
    """구면 좌표(방위각·고도·거리)를 Plotly scene.camera.eye dict 로 변환한다.

    r=zoom, az=radians(azim), el=radians(elev),
    eye=(r·cos(el)·cos(az), r·cos(el)·sin(az), r·sin(el)).
    순수 함수(테스트 가능). make_surface 가 카메라 각도를 계산할 때 사용한다.
    """
    r = float(zoom)
    az = radians(float(azim))
    el = radians(float(elev))
    return dict(x=r * cos(el) * cos(az), y=r * cos(el) * sin(az), z=r * sin(el))


def make_surface(grid: np.ndarray, config: Optional[PlotConfig] = None) -> go.Figure:
    """go.Surface 기반 3D 컬러맵 표면(Origin 'color map surface') figure 반환.

    2D 히트맵과 동일한 grid/물리좌표를 사용해, X·Y는 물리 좌표(µm), Z(높이)는
    intensity, 표면 색상 또한 intensity로 매핑한 기울일 수 있는 3D 뷰를 만든다.
    colormap/zmin/zmax/colorbar/축 라벨/폰트는 히트맵과 동일 규약을 따른다.
    Y축은 히트맵처럼 역방향(autorange="reversed")이라 row 0 = 상단(TOP) 방향이
    일관되게 유지된다.

    Args:
        grid: (ny, nx) 매핑 값 배열. grid[y, x], row 0 = 상단(이미 정렬된 grid).
        config: PlotConfig. None이면 기본값.

    Returns:
        plotly.graph_objects.Figure (go.Surface).
    """
    cfg = config or PlotConfig()
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"grid는 (ny, nx) 2차원 배열이어야 합니다. 현재 차원: {arr.ndim}")
    ny, nx = arr.shape

    x_coords = _axis_coords(nx, cfg.step_x, cfg.x0)
    y_coords = _axis_coords(ny, cfg.step_y, cfg.y0)

    fam = _plotly_font(cfg.font_family)

    # 유효 z-range 결정(2D 히트맵과 동일 규약) → colorbar 눈금 개수 정확 지정.
    zmin_eff = cfg.zmin if cfg.zmin is not None else float(np.nanmin(arr))
    zmax_eff = cfg.zmax if cfg.zmax is not None else float(np.nanmax(arr))
    if not (np.isfinite(zmin_eff) and np.isfinite(zmax_eff) and zmax_eff > zmin_eff):
        zmin_eff = zmax_eff = None

    surf_colorbar = dict(
        title=dict(text=apply_text_markup(cfg.colorbar_label, "plotly"),
                   font=_plotly_font_dict(cfg, "label")),
        tickfont=_plotly_font_dict(cfg, "tick"),
    )
    surf_colorbar.update(_colorbar_ticks(zmin_eff, zmax_eff, int(cfg.colorbar_ticks)))

    surface = go.Surface(
        z=arr,
        x=x_coords,
        y=y_coords,
        colorscale=_plotly_colorscale(cfg.colormap),
        cmin=cfg.zmin,
        cmax=cfg.zmax,
        colorbar=surf_colorbar,
        hovertemplate="X=%{x}<br>Y=%{y}<br>z=%{z}<extra></extra>",
    )

    fig = go.Figure(data=surface)

    scene = dict(
        xaxis=dict(title=dict(text=apply_text_markup(cfg.x_label, "plotly"),
                              font=_plotly_font_dict(cfg, "label")),
                   tickfont=_plotly_font_dict(cfg, "tick"),
                   showticklabels=cfg.show_ticks),
        yaxis=dict(title=dict(text=apply_text_markup(cfg.y_label, "plotly"),
                              font=_plotly_font_dict(cfg, "label")),
                   tickfont=_plotly_font_dict(cfg, "tick"),
                   showticklabels=cfg.show_ticks, autorange="reversed"),
        zaxis=dict(title=dict(text=apply_text_markup(_z_label(cfg), "plotly"),
                              font=_plotly_font_dict(cfg, "label")),
                   tickfont=_plotly_font_dict(cfg, "tick")),
        # 마우스 드래그(자유 시점)는 서버로 전달되지 않으므로, 화면·export 각도를
        # 일치시키기 위해 카메라를 구면 좌표(cam_azim/cam_elev/cam_zoom)로 계산한다.
        camera=dict(eye=camera_eye(cfg.cam_azim, cfg.cam_elev, cfg.cam_zoom)),
    )

    # 종횡비: lock_aspect면 X:Y를 물리 비율로(z는 적당히 낮게), 아니면 cube.
    if cfg.lock_aspect:
        x_span = max(nx * float(cfg.step_x), 1e-9)
        y_span = max(ny * float(cfg.step_y), 1e-9)
        m = max(x_span, y_span)
        scene["aspectmode"] = "manual"
        scene["aspectratio"] = dict(x=x_span / m, y=y_span / m, z=0.55)
    else:
        scene["aspectmode"] = "cube"

    fig.update_layout(
        title=dict(text=apply_text_markup(cfg.title, "plotly"),
                   font=_plotly_font_dict(cfg, "title"),
                   x=_TITLE_X[_title_pos(cfg)], xanchor=_title_pos(cfg)),
        scene=scene,
        font=dict(family=fam),
        margin=dict(l=10, r=10, t=50 if cfg.title else 20, b=10),
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# 3D color-map surface — Matplotlib (고해상도 export)
# ---------------------------------------------------------------------------
def make_matplotlib_surface(
    grid: np.ndarray,
    config: Optional[PlotConfig] = None,
    dpi: int = 300,
) -> Figure:
    """논문용 고해상도 3D 컬러맵 표면 (plot_surface). Figure 반환.

    make_surface(Plotly)의 Matplotlib 대응. X·Y는 물리 좌표(µm), Z는 intensity,
    색상은 cmap(intensity). make_matplotlib_heatmap과 동일한 서식/폰트/cmap 규약.
    y축을 반전하여 row 0 = 상단(TOP) 방향을 히트맵과 일치시킨다.

    Args:
        grid: (ny, nx) 매핑 값 배열. grid[y, x], row 0 = 상단.
        config: PlotConfig. None이면 기본값.
        dpi: 해상도. 기본 300.

    Returns:
        matplotlib.figure.Figure.
    """
    # mpl_toolkits의 3d projection 등록 (import 자체가 등록 트리거)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    cfg = config or PlotConfig()
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"grid는 (ny, nx) 2차원 배열이어야 합니다. 현재 차원: {arr.ndim}")
    ny, nx = arr.shape

    x_coords = _axis_coords(nx, cfg.step_x, cfg.x0)
    y_coords = _axis_coords(ny, cfg.step_y, cfg.y0)
    X, Y = np.meshgrid(x_coords, y_coords)

    fam = _mpl_font(cfg.font_family)
    fig = plt.figure(figsize=(7, 5.5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, arr,
        cmap=_mpl_cmap(cfg.colormap),
        vmin=cfg.zmin, vmax=cfg.zmax,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True,
    )

    ax.set_xlabel(apply_text_markup(cfg.x_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    ax.set_ylabel(apply_text_markup(cfg.y_label, "mpl"), **_mpl_font_kw(cfg, "label"))
    ax.set_zlabel(apply_text_markup(_z_label(cfg), "mpl"), **_mpl_font_kw(cfg, "label"))
    if cfg.title:
        ax.set_title(apply_text_markup(cfg.title, "mpl"), loc=_title_pos(cfg),
                     **_mpl_font_kw(cfg, "title"))
    tick_kw = _mpl_font_kw(cfg, "tick")
    ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
    # 히트맵(row 0 = 상단)과 방향 일치: y축 반전
    ax.invert_yaxis()
    if not cfg.show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(apply_text_markup(cfg.colorbar_label, "mpl"),
                   **_mpl_font_kw(cfg, "label"))
    cbar.ax.tick_params(labelsize=tick_kw["fontsize"], labelcolor=tick_kw["color"])
    try:
        from matplotlib.ticker import LinearLocator
        cbar.locator = LinearLocator(numticks=int(cfg.colorbar_ticks))
        cbar.update_ticks()
    except Exception:
        pass

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectrum viewer (pixel click)
# ---------------------------------------------------------------------------
def make_spectrum_figure(
    waves: np.ndarray,
    spectrum: np.ndarray,
    title: Optional[str] = None,
    point_label: Optional[str] = None,
) -> go.Figure:
    """단일 포인트 스펙트럼 line plot (픽셀 클릭 뷰어용).

    Args:
        waves: (n_waves,) 파수 배열.
        spectrum: (n_waves,) intensity 배열.
        title: 그래프 제목(옵션).
        point_label: 범례에 표시할 포인트 라벨 (예: "(x=3, y=5)").

    Returns:
        plotly.graph_objects.Figure. x="Wavenumber (cm⁻¹)", y="Intensity (a.u.)".
    """
    w = np.asarray(waves, dtype=float).ravel()
    y = np.asarray(spectrum, dtype=float).ravel()
    if w.shape[0] != y.shape[0]:
        raise ValueError(
            f"waves 길이({w.shape[0]})와 spectrum 길이({y.shape[0]})가 맞지 않습니다"
        )

    trace = go.Scatter(
        x=w, y=y, mode="lines",
        name=point_label or "spectrum",
        line=dict(width=1.5),
        hovertemplate="%{x:.1f} cm⁻¹<br>I=%{y:.3g}<extra></extra>",
    )
    fig = go.Figure(data=trace)
    fig.update_layout(
        title=title or (point_label or ""),
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity (a.u.)",
        template="plotly_white",
        margin=dict(l=60, r=20, t=40 if (title or point_label) else 20, b=50),
    )
    return fig
