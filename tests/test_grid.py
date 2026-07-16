"""grid.py 단위 테스트 — 4×4(0..15) 손계산 기대값 + 비정방 + 불변성."""

import numpy as np
import pytest

from core import grid


@pytest.fixture
def base16():
    """0..15, nx=ny=4."""
    return np.arange(16)


def test_raster_top_left(base16):
    got = grid.reshape_to_grid(base16, 4, 4, scan="raster", start="top-left")
    exp = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])
    assert np.array_equal(got, exp)


def test_snake_top_left(base16):
    got = grid.reshape_to_grid(base16, 4, 4, scan="snake", start="top-left")
    exp = np.array([
        [0, 1, 2, 3],
        [7, 6, 5, 4],
        [8, 9, 10, 11],
        [15, 14, 13, 12],
    ])
    assert np.array_equal(got, exp)


def test_raster_bottom_left(base16):
    got = grid.reshape_to_grid(base16, 4, 4, scan="raster", start="bottom-left")
    exp = np.array([
        [12, 13, 14, 15],
        [8, 9, 10, 11],
        [4, 5, 6, 7],
        [0, 1, 2, 3],
    ])
    assert np.array_equal(got, exp)


def test_raster_top_right(base16):
    got = grid.reshape_to_grid(base16, 4, 4, scan="raster", start="top-right")
    exp = np.array([
        [3, 2, 1, 0],
        [7, 6, 5, 4],
        [11, 10, 9, 8],
        [15, 14, 13, 12],
    ])
    assert np.array_equal(got, exp)


def test_raster_bottom_right(base16):
    got = grid.reshape_to_grid(base16, 4, 4, scan="raster", start="bottom-right")
    exp = np.array([
        [15, 14, 13, 12],
        [11, 10, 9, 8],
        [7, 6, 5, 4],
        [3, 2, 1, 0],
    ])
    assert np.array_equal(got, exp)


def test_snake_bottom_left(base16):
    # snake는 raw(top-left) 기준 적용 후 flipud.
    got = grid.reshape_to_grid(base16, 4, 4, scan="snake", start="bottom-left")
    exp = np.array([
        [15, 14, 13, 12],
        [8, 9, 10, 11],
        [7, 6, 5, 4],
        [0, 1, 2, 3],
    ])
    assert np.array_equal(got, exp)


def test_flip_vertical(base16):
    g = grid.reshape_to_grid(base16, 4, 4)
    exp = np.array([
        [12, 13, 14, 15],
        [8, 9, 10, 11],
        [4, 5, 6, 7],
        [0, 1, 2, 3],
    ])
    assert np.array_equal(grid.flip_vertical(g), exp)


def test_flip_horizontal(base16):
    g = grid.reshape_to_grid(base16, 4, 4)
    exp = np.array([
        [3, 2, 1, 0],
        [7, 6, 5, 4],
        [11, 10, 9, 8],
        [15, 14, 13, 12],
    ])
    assert np.array_equal(grid.flip_horizontal(g), exp)


def test_rotate_cw(base16):
    g = grid.reshape_to_grid(base16, 4, 4)
    exp = np.array([
        [12, 8, 4, 0],
        [13, 9, 5, 1],
        [14, 10, 6, 2],
        [15, 11, 7, 3],
    ])
    assert np.array_equal(grid.rotate_cw(g), exp)
    # rotate90 별칭도 CW.
    assert np.array_equal(grid.rotate90(g), exp)


def test_rotate_ccw(base16):
    g = grid.reshape_to_grid(base16, 4, 4)
    exp = np.array([
        [3, 7, 11, 15],
        [2, 6, 10, 14],
        [1, 5, 9, 13],
        [0, 4, 8, 12],
    ])
    assert np.array_equal(grid.rotate_ccw(g), exp)


def test_transpose(base16):
    g = grid.reshape_to_grid(base16, 4, 4)
    exp = np.array([
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ])
    assert np.array_equal(grid.transpose(g), exp)


def test_apply_transform_composed(base16):
    cfg = {"scan": "raster", "start": "top-left", "ops": ["flip_v", "rotate_cw"]}
    got = grid.apply_transform(base16, 4, 4, cfg)
    exp = np.array([
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ])
    assert np.array_equal(got, exp)


def test_non_square_raster():
    # nx=4, ny=2 → (2,4). 축 버그 검출.
    vals = np.arange(8)
    got = grid.reshape_to_grid(vals, 4, 2, scan="raster", start="top-left")
    exp = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ])
    assert got.shape == (2, 4)
    assert np.array_equal(got, exp)


def test_non_square_snake():
    vals = np.arange(8)
    got = grid.reshape_to_grid(vals, 4, 2, scan="snake", start="top-left")
    exp = np.array([
        [0, 1, 2, 3],
        [7, 6, 5, 4],
    ])
    assert np.array_equal(got, exp)


def test_non_square_rotate_cw():
    vals = np.arange(8)
    g = grid.reshape_to_grid(vals, 4, 2)  # (2,4)
    got = grid.rotate_cw(g)
    exp = np.array([
        [4, 0],
        [5, 1],
        [6, 2],
        [7, 3],
    ])
    assert got.shape == (4, 2)
    assert np.array_equal(got, exp)


def test_inputs_not_mutated(base16):
    original = base16.copy()
    g = grid.reshape_to_grid(base16, 4, 4, scan="snake", start="bottom-right")
    assert np.array_equal(base16, original), "reshape_to_grid이 입력을 변형함"

    g_copy = g.copy()
    _ = grid.flip_vertical(g)
    _ = grid.rotate_cw(g)
    _ = grid.transpose(g)
    _ = grid.apply_transform(base16, 4, 4, {"ops": ["flip_h", "transpose"]})
    assert np.array_equal(g, g_copy), "조작 함수가 입력 grid를 변형함"
    assert np.array_equal(base16, original), "apply_transform이 values를 변형함"


def test_length_mismatch_raises(base16):
    with pytest.raises(ValueError):
        grid.reshape_to_grid(base16, 3, 4)  # 12 != 16


def test_bad_scan_start_raises(base16):
    with pytest.raises(ValueError):
        grid.reshape_to_grid(base16, 4, 4, scan="zigzag")
    with pytest.raises(ValueError):
        grid.reshape_to_grid(base16, 4, 4, start="center")


def test_bad_op_raises(base16):
    with pytest.raises(ValueError):
        grid.apply_transform(base16, 4, 4, {"ops": ["flip_v", "bogus"]})
