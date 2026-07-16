"""figure 생성 (순수 코어 모듈, Step 5 시각화 + 스펙트럼 뷰어).

Plotly(인터랙티브) + Matplotlib(논문용 고해상도 export) figure를 생성한다.
figure 객체를 **반환만** 하며, show/save는 하지 않는다(앱이 다운로드/kaleido
export 처리). Streamlit 등 UI 의존성 없음. 순수 함수. 입력 grid는 변형하지 않는다.

그리드 규약: grid는 np.ndarray shape (ny, nx), grid[y, x]. row 0 = 이미지 상단(TOP).
파수 cm⁻¹, 좌표 µm.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from math import cos, radians, sin
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # UI/디스플레이 백엔드 없이 figure만 생성
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "PlotConfig",
    "auto_zrange",
    "make_heatmap",
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
    """Plotly font.family 문자열(폴백 포함). Myriad Pro 등 비웹폰트도 우아히 폴백."""
    n = str(name or "").strip()
    if not n:
        return "Arial, Helvetica, sans-serif"
    if "," in n:  # 이미 폴백 체인이면 그대로
        return n
    return f"{n}, Arial, Helvetica, sans-serif"


def _mpl_font(name: Optional[str]) -> list[str]:
    """Matplotlib fontfamily 폴백 리스트(설치 안 된 폰트도 예외 없이 폴백)."""
    n = str(name or "").strip()
    fam = [n] if n else []
    for fb in ("Arial", "DejaVu Sans", "sans-serif"):
        if fb not in fam:
            fam.append(fb)
    return fam


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

    # 렌더링
    interpolation: str = "none"
    lock_aspect: bool = True

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


def _tickvals(coords: np.ndarray, spacing: Optional[float]) -> Optional[list[float]]:
    """지정 간격(µm)에 맞춘 눈금 좌표 목록. spacing None이면 None(자동)."""
    if spacing is None or spacing <= 0 or coords.size == 0:
        return None
    lo, hi = float(coords.min()), float(coords.max())
    start = np.ceil(lo / spacing) * spacing
    ticks = np.arange(start, hi + spacing * 0.5, spacing)
    return [float(t) for t in ticks]


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

    heatmap = go.Heatmap(
        z=arr,
        x=x_coords,
        y=y_coords,
        colorscale=_plotly_colorscale(cfg.colormap),
        zmin=cfg.zmin,
        zmax=cfg.zmax,
        zsmooth=zsmooth,
        colorbar=dict(
            title=dict(
                text=cfg.colorbar_label,
                font=dict(family=_plotly_font(cfg.font_family), size=cfg.font_size_label),
            ),
            nticks=int(cfg.colorbar_ticks),
            tickfont=dict(family=_plotly_font(cfg.font_family), size=cfg.font_size_tick),
        ),
        hovertemplate="X=%{x}<br>Y=%{y}<br>z=%{z}<extra></extra>",
    )

    fig = go.Figure(data=heatmap)

    xaxis = dict(
        title=dict(text=cfg.x_label,
                   font=dict(family=_plotly_font(cfg.font_family), size=cfg.font_size_label)),
        tickfont=dict(family=_plotly_font(cfg.font_family), size=cfg.font_size_tick),
        showticklabels=cfg.show_ticks,
        ticks="outside" if cfg.show_ticks else "",
        constrain="domain",
    )
    yaxis = dict(
        title=dict(text=cfg.y_label,
                   font=dict(family=_plotly_font(cfg.font_family), size=cfg.font_size_label)),
        tickfont=dict(family=_plotly_font(cfg.font_family), size=cfg.font_size_tick),
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
        title=dict(text=cfg.title,
                   font=dict(family=_plotly_font(cfg.font_family), size=cfg.font_size_title)),
        xaxis=xaxis,
        yaxis=yaxis,
        font=dict(family=_plotly_font(cfg.font_family)),
        margin=dict(l=70, r=70, t=60 if cfg.title else 30, b=60),
        template="plotly_white",
    )
    return fig


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

    ax.set_xlabel(cfg.x_label, fontsize=cfg.font_size_label, fontfamily=_mpl_font(cfg.font_family))
    ax.set_ylabel(cfg.y_label, fontsize=cfg.font_size_label, fontfamily=_mpl_font(cfg.font_family))
    if cfg.title:
        ax.set_title(cfg.title, fontsize=cfg.font_size_title, fontfamily=_mpl_font(cfg.font_family))

    ax.tick_params(labelsize=cfg.font_size_tick)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(_mpl_font(cfg.font_family))

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
    cbar.set_label(cfg.colorbar_label, fontsize=cfg.font_size_label,
                   fontfamily=_mpl_font(cfg.font_family))
    cbar.ax.tick_params(labelsize=cfg.font_size_tick)
    # colorbar 눈금 개수
    try:
        from matplotlib.ticker import MaxNLocator
        cbar.locator = MaxNLocator(nbins=int(cfg.colorbar_ticks))
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
    surface = go.Surface(
        z=arr,
        x=x_coords,
        y=y_coords,
        colorscale=_plotly_colorscale(cfg.colormap),
        cmin=cfg.zmin,
        cmax=cfg.zmax,
        colorbar=dict(
            title=dict(text=cfg.colorbar_label,
                       font=dict(family=fam, size=cfg.font_size_label)),
            nticks=int(cfg.colorbar_ticks),
            tickfont=dict(family=fam, size=cfg.font_size_tick),
        ),
        hovertemplate="X=%{x}<br>Y=%{y}<br>z=%{z}<extra></extra>",
    )

    fig = go.Figure(data=surface)

    axis_font = dict(family=fam, size=cfg.font_size_label)
    tick_font = dict(family=fam, size=cfg.font_size_tick)

    scene = dict(
        xaxis=dict(title=dict(text=cfg.x_label, font=axis_font),
                   tickfont=tick_font, showticklabels=cfg.show_ticks),
        yaxis=dict(title=dict(text=cfg.y_label, font=axis_font),
                   tickfont=tick_font, showticklabels=cfg.show_ticks,
                   autorange="reversed"),
        zaxis=dict(title=dict(text=cfg.colorbar_label, font=axis_font),
                   tickfont=tick_font),
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
        title=dict(text=cfg.title, font=dict(family=fam, size=cfg.font_size_title)),
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

    ax.set_xlabel(cfg.x_label, fontsize=cfg.font_size_label, fontfamily=fam)
    ax.set_ylabel(cfg.y_label, fontsize=cfg.font_size_label, fontfamily=fam)
    ax.set_zlabel(cfg.colorbar_label, fontsize=cfg.font_size_label, fontfamily=fam)
    if cfg.title:
        ax.set_title(cfg.title, fontsize=cfg.font_size_title, fontfamily=fam)
    ax.tick_params(labelsize=cfg.font_size_tick)
    # 히트맵(row 0 = 상단)과 방향 일치: y축 반전
    ax.invert_yaxis()
    if not cfg.show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(cfg.colorbar_label, fontsize=cfg.font_size_label, fontfamily=fam)
    cbar.ax.tick_params(labelsize=cfg.font_size_tick)
    try:
        from matplotlib.ticker import MaxNLocator
        cbar.locator = MaxNLocator(nbins=int(cfg.colorbar_ticks))
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
