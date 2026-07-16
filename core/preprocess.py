"""라만 스펙트럼 전처리 (순수 코어 모듈, Step 2).

모든 함수는 `spectra: np.ndarray (n_points, n_waves)`를 입력받아
**새로운** 처리된 배열(같은 shape)을 반환한다. 원본은 절대 변형하지 않는다.
Streamlit 등 UI 의존성 없음. 순수 함수.

처리 단계:
  - baseline_als      : 비대칭 최소제곱(Eilers ALS) baseline 추정 후 차감
  - baseline_poly     : 다항식 fit baseline 차감
  - savgol_smooth     : Savitzky-Golay 평활
  - normalize         : max / 특정 피크 / off 정규화
  - remove_cosmic_rays: 파수축 rolling median + MAD 이상치(코스믹레이) 제거
  - apply_preprocessing: UI용 순서 파이프라인 진입점

파수는 cm⁻¹ 단위를 가정한다.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, medfilt

__all__ = [
    "baseline_als",
    "baseline_poly",
    "savgol_smooth",
    "normalize",
    "remove_cosmic_rays",
    "apply_preprocessing",
]


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------
def _as_2d_float(spectra: np.ndarray) -> np.ndarray:
    """입력을 (n_points, n_waves) float 배열 사본으로 정규화."""
    arr = np.asarray(spectra, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(
            f"spectra는 (n_points, n_waves) 2차원 배열이어야 합니다. 현재 차원: {arr.ndim}"
        )
    return arr.copy()


# ---------------------------------------------------------------------------
# Baseline — ALS (Eilers & Boelens, Asymmetric Least Squares)
# ---------------------------------------------------------------------------
def _als_baseline_1d(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    """단일 스펙트럼에 대한 ALS baseline 추정.

    최소화 대상: (y - z)에 대한 비대칭 가중 잔차 + lam * 2차 차분 평활 항.
    z_i > y_i 인 점(피크 위)에는 작은 가중치 p, 그 외에는 (1-p).
    """
    n = y.shape[0]
    if n < 3:
        # 2차 차분을 구성할 수 없는 짧은 신호 → baseline 0 취급.
        return np.zeros_like(y)
    # 2차 차분 행렬 D: (n-2, n)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    D = D.tocsc()
    DtD = lam * (D.T @ D)
    w = np.ones(n)
    z = y.copy()
    for _ in range(max(1, int(niter))):
        W = sparse.diags(w, 0, shape=(n, n))
        Z = (W + DtD).tocsc()
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return z


def baseline_als(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """비대칭 최소제곱(ALS) baseline을 추정하여 차감한 스펙트럼 반환.

    Eilers & Boelens (2005) 표준 ALS. 각 측정 포인트(행)에 개별 적용한다.
    sparse 행렬 + spsolve 사용.

    Args:
        spectra: (n_points, n_waves) intensity 배열.
        lam: 평활도(smoothness) 파라미터 λ. 클수록 baseline이 매끄러움. 기본 1e5.
        p: 비대칭 가중치 (0<p<1). 피크가 위로 솟은 신호는 작게(0.001~0.1). 기본 0.01.
        niter: 재가중 반복 횟수. 기본 10.

    Returns:
        baseline이 차감된 새 배열 (같은 shape).

    Raises:
        ValueError: 파라미터 범위가 잘못된 경우 (친절한 한국어 메시지).
    """
    if lam <= 0:
        raise ValueError(f"ALS baseline의 lam(λ)은 0보다 커야 합니다. 입력값: {lam}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"ALS baseline의 p는 0과 1 사이여야 합니다. 입력값: {p}")
    if niter < 1:
        raise ValueError(f"ALS baseline의 niter는 1 이상이어야 합니다. 입력값: {niter}")

    arr = _as_2d_float(spectra)
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        base = _als_baseline_1d(arr[i], lam=lam, p=p, niter=niter)
        out[i] = arr[i] - base
    return out


# ---------------------------------------------------------------------------
# Baseline — polynomial fit
# ---------------------------------------------------------------------------
def baseline_poly(
    spectra: np.ndarray,
    waves: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """다항식 fit baseline을 차감한 스펙트럼 반환. 각 행에 개별 적용.

    각 스펙트럼을 waves에 대한 `order`차 다항식으로 최소제곱 피팅한 뒤
    그 피팅 곡선을 baseline으로 차감한다.

    Args:
        spectra: (n_points, n_waves) intensity 배열.
        waves: (n_waves,) 파수 배열.
        order: 다항식 차수. 기본 3.

    Returns:
        baseline이 차감된 새 배열 (같은 shape).

    Raises:
        ValueError: order가 음수이거나 waves 길이가 맞지 않는 경우.
    """
    arr = _as_2d_float(spectra)
    w = np.asarray(waves, dtype=float)
    if w.ndim != 1 or w.shape[0] != arr.shape[1]:
        raise ValueError(
            f"waves 길이({w.shape})가 스펙트럼 파수 개수({arr.shape[1]})와 맞지 않습니다"
        )
    if order < 0:
        raise ValueError(f"다항식 차수(order)는 0 이상이어야 합니다. 입력값: {order}")
    if order >= w.shape[0]:
        raise ValueError(
            f"다항식 차수(order={order})가 파수 개수({w.shape[0]})보다 작아야 합니다"
        )

    # 수치 안정성을 위해 파수축을 [-1, 1]로 스케일
    w_min, w_max = float(w.min()), float(w.max())
    span = (w_max - w_min) or 1.0
    x = 2.0 * (w - w_min) / span - 1.0

    # Vandermonde 기반 일괄 최소제곱: 모든 행을 한 번에 피팅
    V = np.vander(x, N=order + 1, increasing=True)  # (n_waves, order+1)
    coeffs, *_ = np.linalg.lstsq(V, arr.T, rcond=None)  # (order+1, n_points)
    baseline = (V @ coeffs).T  # (n_points, n_waves)
    return arr - baseline


# ---------------------------------------------------------------------------
# Smoothing — Savitzky-Golay
# ---------------------------------------------------------------------------
def savgol_smooth(
    spectra: np.ndarray,
    window: int = 11,
    poly: int = 3,
) -> np.ndarray:
    """Savitzky-Golay 필터로 파수축을 따라 평활한 스펙트럼 반환.

    Args:
        spectra: (n_points, n_waves) intensity 배열.
        window: 필터 창 길이 (홀수, poly보다 커야 함). 기본 11.
        poly: 다항식 차수. 기본 3.

    Returns:
        평활된 새 배열 (같은 shape).

    Raises:
        ValueError: window가 짝수이거나 poly 이하인 경우 (친절한 한국어 메시지).
    """
    arr = _as_2d_float(spectra)
    n_waves = arr.shape[1]
    if window % 2 == 0:
        raise ValueError(f"Savitzky-Golay window는 홀수여야 합니다. 입력값: {window}")
    if window <= poly:
        raise ValueError(
            f"Savitzky-Golay window({window})는 다항식 차수 poly({poly})보다 커야 합니다"
        )
    if window > n_waves:
        raise ValueError(
            f"window({window})가 파수 개수({n_waves})보다 큽니다. 더 작은 window를 사용하세요"
        )
    if poly < 0:
        raise ValueError(f"poly는 0 이상이어야 합니다. 입력값: {poly}")

    return savgol_filter(arr, window_length=window, polyorder=poly, axis=1)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize(
    spectra: np.ndarray,
    mode: str = "max",
    waves: Optional[np.ndarray] = None,
    peak_wave: Optional[float] = None,
) -> np.ndarray:
    """스펙트럼 정규화.

    Args:
        spectra: (n_points, n_waves) intensity 배열.
        mode: {"off", "max", "peak"}.
            - "off" : 사본만 반환(변경 없음).
            - "max" : 각 스펙트럼을 자신의 최댓값으로 나눔.
            - "peak": `peak_wave`에 가장 가까운 파수 위치의 intensity로 나눔.
        waves: mode="peak"일 때 필요한 (n_waves,) 파수 배열.
        peak_wave: mode="peak"일 때 기준 파수(cm⁻¹).

    Returns:
        정규화된 새 배열 (같은 shape). 0으로 나누는 경우는 1로 대체해 방어.

    Raises:
        ValueError: mode가 유효하지 않거나 peak 모드 인자가 빠진 경우.
    """
    arr = _as_2d_float(spectra)
    if mode == "off":
        return arr

    if mode == "max":
        denom = np.nanmax(arr, axis=1, keepdims=True)
    elif mode == "peak":
        if waves is None or peak_wave is None:
            raise ValueError(
                "normalize(mode='peak')에는 waves와 peak_wave가 모두 필요합니다"
            )
        w = np.asarray(waves, dtype=float)
        if w.ndim != 1 or w.shape[0] != arr.shape[1]:
            raise ValueError(
                f"waves 길이({w.shape})가 스펙트럼 파수 개수({arr.shape[1]})와 맞지 않습니다"
            )
        idx = int(np.argmin(np.abs(w - float(peak_wave))))
        denom = arr[:, idx:idx + 1]
    else:
        raise ValueError(
            f"normalize mode는 'off'/'max'/'peak' 중 하나여야 합니다. 입력값: '{mode}'"
        )

    # 0(또는 NaN)으로 나누기 방어: 분모가 0이거나 유한하지 않으면 1로 대체
    denom = np.where(np.isfinite(denom) & (denom != 0.0), denom, 1.0)
    return arr / denom


# ---------------------------------------------------------------------------
# Cosmic ray removal
# ---------------------------------------------------------------------------
def remove_cosmic_rays(
    spectra: np.ndarray,
    threshold: float = 5.0,
    window: int = 5,
) -> np.ndarray:
    """코스믹레이(우주선) 스파이크 제거.

    방법(파수축 rolling median + robust MAD):
      각 스펙트럼(행)에 대해 파수축을 따라 이동 중앙값(rolling median)을 구하고
      잔차 = |원신호 - median| 을 계산한다. 잔차가
      threshold × (1.4826 × MAD) 를 초과하는 채널을 스파이크로 간주해
      해당 채널을 rolling median 값으로 대체한다. (1.4826은 MAD의 정규분포
      표준편차 환산 상수.) 코스믹레이는 1~수 채널 폭의 급격한 단발 스파이크
      이므로 좁은 창의 median으로 효과적으로 검출·대체된다.

    Args:
        spectra: (n_points, n_waves) intensity 배열.
        threshold: MAD 배수 임계값. 클수록 덜 공격적. 기본 5.0.
        window: rolling median 창 길이(홀수). 기본 5.

    Returns:
        스파이크가 제거된 새 배열 (같은 shape).

    Raises:
        ValueError: threshold<=0 또는 window가 짝수/1미만인 경우.
    """
    if threshold <= 0:
        raise ValueError(f"cosmic ray threshold는 0보다 커야 합니다. 입력값: {threshold}")
    if window < 3 or window % 2 == 0:
        raise ValueError(
            f"cosmic ray window는 3 이상의 홀수여야 합니다. 입력값: {window}"
        )

    arr = _as_2d_float(spectra)
    n_waves = arr.shape[1]
    if n_waves < window:
        # 신호가 창보다 짧으면 검출 불가 → 사본 그대로 반환.
        return arr

    out = arr.copy()
    for i in range(arr.shape[0]):
        row = arr[i]
        med = medfilt(row, kernel_size=window)
        resid = np.abs(row - med)
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if scale <= 0:
            # 잔차 분산이 없으면(평평) 스파이크 없음으로 간주.
            continue
        spikes = resid > threshold * scale
        if spikes.any():
            out[i, spikes] = med[spikes]
    return out


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def apply_preprocessing(
    spectra: np.ndarray,
    waves: np.ndarray,
    config: dict,
) -> np.ndarray:
    """UI용 순서 파이프라인 진입점. 원본은 변형하지 않는다.

    적용 순서(고정): cosmic → baseline → smooth → normalize.
    각 단계는 선택적이며 config에서 해당 키가 없거나 None이면 건너뛴다.

    Config 스키마 (모든 키 선택):
        {
          "cosmic": {"threshold": 5.0, "window": 5} | None,
          "baseline": {"method": "als", "lam": 1e5, "p": 0.01, "niter": 10} | None
                      | {"method": "poly", "order": 3} | None
                      | {"method": None},
          "smooth": {"window": 11, "poly": 3} | None,
          "normalize": {"mode": "max"} | {"mode": "peak", "peak_wave": 1580.0}
                      | {"mode": "off"} | None,
        }

    Args:
        spectra: (n_points, n_waves) 원본 intensity 배열.
        waves: (n_waves,) 파수 배열 (baseline poly / peak normalize에 사용).
        config: 위 스키마의 파이프라인 설정 dict.

    Returns:
        전처리된 새 배열 (같은 shape). 어떤 단계도 없으면 입력의 사본.

    Raises:
        ValueError: 하위 함수의 파라미터 검증 실패 시 (친절한 한국어 메시지).
    """
    cfg = config or {}
    out = _as_2d_float(spectra)  # 사본으로 시작 → 원본 불변

    # 1) Cosmic ray 제거
    cosmic = cfg.get("cosmic")
    if cosmic:
        out = remove_cosmic_rays(
            out,
            threshold=cosmic.get("threshold", 5.0),
            window=cosmic.get("window", 5),
        )

    # 2) Baseline 보정
    baseline = cfg.get("baseline")
    if baseline:
        method = baseline.get("method")
        if method == "als":
            out = baseline_als(
                out,
                lam=baseline.get("lam", 1e5),
                p=baseline.get("p", 0.01),
                niter=baseline.get("niter", 10),
            )
        elif method == "poly":
            out = baseline_poly(out, waves, order=baseline.get("order", 3))
        elif method in (None, "off"):
            pass
        else:
            raise ValueError(
                f"baseline method는 'als'/'poly'/None 중 하나여야 합니다. 입력값: '{method}'"
            )

    # 3) Smoothing
    smooth = cfg.get("smooth")
    if smooth:
        out = savgol_smooth(
            out,
            window=smooth.get("window", 11),
            poly=smooth.get("poly", 3),
        )

    # 4) Normalization
    norm = cfg.get("normalize")
    if norm:
        mode = norm.get("mode", "max")
        out = normalize(
            out,
            mode=mode,
            waves=waves,
            peak_wave=norm.get("peak_wave"),
        )

    return out
