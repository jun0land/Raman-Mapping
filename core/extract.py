"""매핑 값 추출 (Step 3, 순수 코어 모듈).

전처리(선택)까지 끝난 스펙트럼 배열에서, 각 측정 포인트마다 하나의 스칼라
매핑 값을 뽑아 길이 n_points의 1D 배열로 반환한다. 이 값을 grid.py가 2D로
reshape 한다.

모든 함수는 다음을 입력으로 받는다:
  spectra : np.ndarray (n_points, n_waves)  — intensity
  waves   : np.ndarray (n_waves,)           — 파수(cm⁻¹), 오름차순
반환:
  np.ndarray (n_points,)

순수 함수. Streamlit 등 UI 의존성 없음. 입력 배열을 변형하지 않는다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# NumPy 2.0에서 np.trapz → np.trapezoid로 이름 변경. 양쪽 버전 호환.
_trapz = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "single_intensity",
    "peak_max",
    "peak_area",
    "peak_position",
    "fwhm",
    "ratio",
    "extract_values",
]


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _as_2d(spectra: np.ndarray) -> np.ndarray:
    arr = np.asarray(spectra, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"spectra는 (n_points, n_waves) 2차원 배열이어야 합니다 (현재 {arr.ndim}차원)."
        )
    return arr


def _as_waves(waves: np.ndarray, n_waves: int) -> np.ndarray:
    w = np.asarray(waves, dtype=float)
    if w.ndim != 1:
        raise ValueError("waves는 1차원 배열이어야 합니다.")
    if w.shape[0] != n_waves:
        raise ValueError(
            f"waves 길이({w.shape[0]})가 spectra 파수축({n_waves})과 맞지 않습니다."
        )
    return w


def _window_mask(waves: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """[w1, w2] 구간(경계 포함) 파수 마스크. 구간 유효성 검증 포함.

    Raises:
        ValueError: w1 >= w2 이거나, 구간이 파수축 범위와 겹치지 않아 선택된
            파수가 하나도 없을 때 (친절한 한국어 메시지).
    """
    w1f, w2f = float(w1), float(w2)
    if not (w1f < w2f):
        raise ValueError(
            f"파수 구간이 올바르지 않습니다: 시작({w1f})은 끝({w2f})보다 작아야 합니다."
        )
    lo, hi = float(waves.min()), float(waves.max())
    mask = (waves >= w1f) & (waves <= w2f)
    if not mask.any():
        raise ValueError(
            f"파수 구간 [{w1f}, {w2f}]에 해당하는 데이터가 없습니다. "
            f"측정 파수 범위는 [{lo}, {hi}] cm⁻¹ 입니다."
        )
    return mask


# ---------------------------------------------------------------------------
# 추출 모드
# ---------------------------------------------------------------------------
def single_intensity(spectra: np.ndarray, waves: np.ndarray, wave: float) -> np.ndarray:
    """지정 파수에 가장 가까운 파수에서의 intensity.

    Args:
        spectra: (n_points, n_waves) intensity.
        waves: (n_waves,) 파수, 오름차순.
        wave: 대상 파수(cm⁻¹).

    Returns:
        (n_points,) 각 포인트의 intensity.
    """
    s = _as_2d(spectra)
    w = _as_waves(waves, s.shape[1])
    idx = int(np.argmin(np.abs(w - float(wave))))
    return s[:, idx].copy()


def peak_max(spectra: np.ndarray, waves: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """파수 구간 [w1, w2] 내 최대 intensity.

    Returns:
        (n_points,) 구간 내 최대값.
    """
    s = _as_2d(spectra)
    w = _as_waves(waves, s.shape[1])
    mask = _window_mask(w, w1, w2)
    return s[:, mask].max(axis=1)


def peak_area(spectra: np.ndarray, waves: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """파수 구간 [w1, w2]의 사다리꼴 적분값 (raw, baseline 미차감).

    Returns:
        (n_points,) 구간 적분 면적. 파수 오름차순 기준이므로 양의 부호.
    """
    s = _as_2d(spectra)
    w = _as_waves(waves, s.shape[1])
    mask = _window_mask(w, w1, w2)
    sub_w = w[mask]
    sub_s = s[:, mask]
    if sub_w.shape[0] < 2:
        # 단일 파수만 잡히면 면적 0.
        return np.zeros(s.shape[0], dtype=float)
    return _trapz(sub_s, sub_w, axis=1)


def peak_position(spectra: np.ndarray, waves: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """파수 구간 [w1, w2] 내 최대 intensity가 나타나는 파수(cm⁻¹).

    스트레인 매핑 등 피크 위치 이동 관찰용.

    Returns:
        (n_points,) 각 포인트에서 최대값 위치의 파수.
    """
    s = _as_2d(spectra)
    w = _as_waves(waves, s.shape[1])
    mask = _window_mask(w, w1, w2)
    sub_w = w[mask]
    sub_s = s[:, mask]
    arg = np.argmax(sub_s, axis=1)
    return sub_w[arg]


def fwhm(spectra: np.ndarray, waves: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """파수 구간 [w1, w2] 내 반치폭(FWHM, cm⁻¹).

    구간 내 최소값을 로컬 baseline으로 두고, half-max = baseline + (max-baseline)/2
    수준을 최대 위치 좌/우로 각각 선형 보간하여 교차점을 찾는다. 폭 = 우측 - 좌측.

    교차점을 찾지 못하는 경우(피크가 구간 경계에 걸쳐 한쪽이 반치까지 내려오지
    않는 등)에는 np.nan을 반환한다.

    Returns:
        (n_points,) FWHM (cm⁻¹). 미검출 포인트는 np.nan.
    """
    s = _as_2d(spectra)
    w = _as_waves(waves, s.shape[1])
    mask = _window_mask(w, w1, w2)
    sub_w = w[mask]
    sub_s = s[:, mask]
    n = sub_w.shape[0]
    out = np.full(s.shape[0], np.nan, dtype=float)
    if n < 2:
        return out
    for p in range(s.shape[0]):
        out[p] = _fwhm_row(sub_w, sub_s[p])
    return out


def _fwhm_row(x: np.ndarray, y: np.ndarray) -> float:
    """단일 스펙트럼 구간에서 FWHM 계산. 미검출 시 np.nan."""
    imax = int(np.argmax(y))
    ymax = float(y[imax])
    baseline = float(np.min(y))
    if ymax <= baseline:
        return np.nan
    half = baseline + (ymax - baseline) / 2.0

    # 좌측 교차: imax에서 왼쪽으로, y가 half 아래로 내려가는 첫 구간.
    x_left = np.nan
    for i in range(imax, 0, -1):
        y_hi, y_lo = float(y[i]), float(y[i - 1])
        if y_lo <= half <= y_hi and y_hi != y_lo:
            x_left = x[i - 1] + (half - y_lo) * (x[i] - x[i - 1]) / (y_hi - y_lo)
            break
    # 우측 교차: imax에서 오른쪽으로.
    x_right = np.nan
    for i in range(imax, len(y) - 1):
        y_hi, y_lo = float(y[i]), float(y[i + 1])
        if y_lo <= half <= y_hi and y_hi != y_lo:
            x_right = x[i] + (half - y_hi) * (x[i + 1] - x[i]) / (y_lo - y_hi)
            break
    if np.isnan(x_left) or np.isnan(x_right):
        return np.nan
    return float(x_right - x_left)


def ratio(
    spectra: np.ndarray,
    waves: np.ndarray,
    a1: float,
    a2: float,
    b1: float,
    b2: float,
    metric: str = "max",
) -> np.ndarray:
    """구간 A [a1,a2] 값 / 구간 B [b1,b2] 값 비율 (예: I_D/I_G).

    Args:
        metric: 'max' → 각 구간의 peak_max, 'area' → 각 구간의 peak_area.

    Returns:
        (n_points,) A/B. 분모(B)가 0인 포인트는 np.nan.
    """
    s = _as_2d(spectra)
    w = _as_waves(waves, s.shape[1])
    if metric == "max":
        a = peak_max(s, w, a1, a2)
        b = peak_max(s, w, b1, b2)
    elif metric == "area":
        a = peak_area(s, w, a1, a2)
        b = peak_area(s, w, b1, b2)
    else:
        raise ValueError(f"metric은 'max' 또는 'area'여야 합니다 (입력: {metric!r}).")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out = np.where(b == 0, np.nan, out)
    return out


# ---------------------------------------------------------------------------
# 디스패처
# ---------------------------------------------------------------------------
_MODES = ("single", "peak_max", "peak_area", "peak_position", "fwhm", "ratio")


def extract_values(spectra: np.ndarray, waves: np.ndarray, mode: str, **params: Any) -> np.ndarray:
    """모드 문자열에 따라 매핑 값을 추출하는 단일 진입점.

    Args:
        mode: 'single' | 'peak_max' | 'peak_area' | 'peak_position' | 'fwhm' | 'ratio'.
        **params: 모드별 파라미터.
            single       : wave
            peak_max     : w1, w2
            peak_area    : w1, w2
            peak_position: w1, w2
            fwhm         : w1, w2
            ratio        : a1, a2, b1, b2, [metric='max'|'area']

    Returns:
        (n_points,) 매핑 값 1D 배열.

    Raises:
        ValueError: 알 수 없는 mode, 필수 파라미터 누락, 잘못된 파수 구간일 때
            (친절한 한국어 메시지).
    """
    if mode not in _MODES:
        raise ValueError(
            f"알 수 없는 추출 모드입니다: {mode!r}. 사용 가능: {', '.join(_MODES)}."
        )

    def _need(*keys: str) -> None:
        missing = [k for k in keys if k not in params]
        if missing:
            raise ValueError(
                f"'{mode}' 모드에는 파라미터 {missing}가 필요합니다."
            )

    if mode == "single":
        _need("wave")
        return single_intensity(spectra, waves, params["wave"])
    if mode == "peak_max":
        _need("w1", "w2")
        return peak_max(spectra, waves, params["w1"], params["w2"])
    if mode == "peak_area":
        _need("w1", "w2")
        return peak_area(spectra, waves, params["w1"], params["w2"])
    if mode == "peak_position":
        _need("w1", "w2")
        return peak_position(spectra, waves, params["w1"], params["w2"])
    if mode == "fwhm":
        _need("w1", "w2")
        return fwhm(spectra, waves, params["w1"], params["w2"])
    # ratio
    _need("a1", "a2", "b1", "b2")
    return ratio(
        spectra, waves,
        params["a1"], params["a2"], params["b1"], params["b2"],
        metric=params.get("metric", "max"),
    )
