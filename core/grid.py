"""Reshape & 방향 정렬 (Step 4, 순수 코어 모듈). ⭐ 가장 중요.

길이 n_points의 1D 매핑 값을 2D 그리드 (ny, nx)로 만들고, optic(현미경) 이미지와
방향을 맞추기 위한 flip/rotate/transpose를 적용한다.

핵심 규약 (반드시 숙지):
  - 반환 배열의 **row 0 = 이미지의 위쪽(TOP) 행**, 인덱싱은 grid[y, x].
  - 기본 raster는 row-major로 채운다: grid = values.reshape(ny, nx).
    즉 첫 측정점(values[0])이 top-left, fast axis(빠른 축)는 가로(행 내부).
  - scan="snake"(serpentine): 홀수 인덱스 행을 좌우 반전
    (grid[1::2] = grid[1::2, ::-1]). 이는 **원본 취득 순서(raw)** 기준으로 적용한다.
  - start(시작 코너)는 "첫 측정점이 실제로 놓인 코너"를 뜻한다.
    snake를 raw(첫 점 = top-left) 기준으로 먼저 적용한 뒤,
    아래 규칙으로 flip 하여 첫 점을 named 코너로 이동시킨다:
        top-left     : 변형 없음
        bottom-left  : flipud
        top-right    : fliplr
        bottom-right : flipud + fliplr (= 180° 회전)
    → 즉 **snake는 raw 순서로, start는 그 뒤 재배향**. 순서가 바뀌면 결과가 달라짐.

모든 함수는 새 배열을 반환하며 입력(values/grid)을 절대 변형하지 않는다.
Streamlit 등 UI 의존성 없음.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "reshape_to_grid",
    "flip_vertical",
    "flip_horizontal",
    "rotate90",
    "rotate_cw",
    "rotate_ccw",
    "transpose",
    "apply_transform",
]

_STARTS = ("top-left", "bottom-left", "top-right", "bottom-right")
_SCANS = ("raster", "snake")
_OPS = ("flip_v", "flip_h", "rotate_cw", "rotate_ccw", "transpose")


# ---------------------------------------------------------------------------
# reshape
# ---------------------------------------------------------------------------
def reshape_to_grid(
    values: np.ndarray,
    nx: int,
    ny: int,
    scan: str = "raster",
    start: str = "top-left",
) -> np.ndarray:
    """1D 매핑 값을 2D 그리드 (ny, nx)로 reshape + 스캔/시작코너 반영.

    Args:
        values: (n_points,) 매핑 값. n_points == nx*ny 이어야 한다.
        nx: 가로 셀 수 (열 개수).
        ny: 세로 셀 수 (행 개수).
        scan: 'raster' | 'snake'(serpentine). snake는 raw 취득 순서 기준으로
            홀수 행을 좌우 반전.
        start: 'top-left' | 'bottom-left' | 'top-right' | 'bottom-right'.
            첫 측정점(values[0])이 놓인 코너.

    Returns:
        (ny, nx) 새 배열. row 0 = TOP.

    Raises:
        ValueError: 길이 불일치, 알 수 없는 scan/start.
    """
    vals = np.asarray(values)
    if vals.ndim != 1:
        vals = vals.reshape(-1)
    if vals.size != nx * ny:
        raise ValueError(
            f"값 개수({vals.size})가 그리드({nx}×{ny}={nx * ny})와 맞지 않습니다."
        )
    if scan not in _SCANS:
        raise ValueError(f"scan은 {_SCANS} 중 하나여야 합니다 (입력: {scan!r}).")
    if start not in _STARTS:
        raise ValueError(f"start는 {_STARTS} 중 하나여야 합니다 (입력: {start!r}).")

    # raw: 첫 점 = top-left, fast axis = 가로
    grid = vals.reshape(ny, nx).copy()
    if scan == "snake":
        grid[1::2] = grid[1::2, ::-1]

    # start 코너로 재배향
    if start == "bottom-left":
        grid = np.flipud(grid)
    elif start == "top-right":
        grid = np.fliplr(grid)
    elif start == "bottom-right":
        grid = np.flipud(np.fliplr(grid))
    # top-left → 그대로

    return np.ascontiguousarray(grid)


# ---------------------------------------------------------------------------
# 개별 방향 조작 (모두 새 배열 반환)
# ---------------------------------------------------------------------------
def flip_vertical(grid: np.ndarray) -> np.ndarray:
    """상하 반전 (np.flipud). 위/아래 행 뒤집기."""
    return np.flipud(np.asarray(grid)).copy()


def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """좌우 반전 (np.fliplr). 좌/우 열 뒤집기."""
    return np.fliplr(np.asarray(grid)).copy()


def rotate_cw(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """시계방향(CW) 90°×k 회전.

    np.rot90은 반시계(CCW)가 양수 k이므로, CW는 np.rot90(grid, -k)로 구현.
    """
    return np.ascontiguousarray(np.rot90(np.asarray(grid), -k))


def rotate_ccw(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """반시계방향(CCW) 90°×k 회전 (np.rot90 기본 방향)."""
    return np.ascontiguousarray(np.rot90(np.asarray(grid), k))


def rotate90(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """양수 k에 대해 시계방향(CW) 90°×k 회전 (rotate_cw 별칭)."""
    return rotate_cw(grid, k)


def transpose(grid: np.ndarray) -> np.ndarray:
    """전치 (행↔열). 새 배열(복사본) 반환."""
    return np.asarray(grid).T.copy()


# ---------------------------------------------------------------------------
# 통합 진입점
# ---------------------------------------------------------------------------
_OP_FUNCS = {
    "flip_v": flip_vertical,
    "flip_h": flip_horizontal,
    "rotate_cw": rotate_cw,
    "rotate_ccw": rotate_ccw,
    "transpose": transpose,
}


def apply_transform(values: np.ndarray, nx: int, ny: int, config: dict) -> np.ndarray:
    """reshape(scan+start) 후, ops 리스트를 순서대로 적용하는 UI 단일 진입점.

    Config 스키마:
        {
            "scan":  "raster" | "snake",          # 기본 "raster"
            "start": "top-left" | "bottom-left"   # 기본 "top-left"
                     | "top-right" | "bottom-right",
            "ops":   ["flip_v", "flip_h",         # 순서대로 적용, 기본 []
                      "rotate_cw", "rotate_ccw", "transpose"]
        }
    ops 적용 순서 = 리스트 순서 (결정적). rotate_cw/ccw는 90° 1회.

    Args:
        values: (n_points,) 매핑 값.
        nx, ny: 그리드 크기.
        config: 위 스키마 딕셔너리.

    Returns:
        (변형 후) 2D np.ndarray 새 배열. 입력 values는 변형되지 않는다.

    Raises:
        ValueError: 알 수 없는 op 이름 등.
    """
    scan = config.get("scan", "raster")
    start = config.get("start", "top-left")
    ops = config.get("ops", []) or []

    grid = reshape_to_grid(values, nx, ny, scan=scan, start=start)
    for op in ops:
        func = _OP_FUNCS.get(op)
        if func is None:
            raise ValueError(
                f"알 수 없는 변환 op입니다: {op!r}. 사용 가능: {', '.join(_OPS)}."
            )
        grid = func(grid)
    return grid
