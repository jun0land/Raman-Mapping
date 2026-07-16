"""extract.py 단위 테스트 — 합성 스펙트럼으로 각 모드 검증 + 친절 에러."""

import numpy as np
import pytest

from core import extract


@pytest.fixture
def synth():
    """waves=[0..4], 2 포인트.

    row0 = [0,2,4,2,0] : 파수 2에서 최대 4 (대칭 삼각).
    row1 = [1,3,1,5,1] : 파수 3에서 최대 5.
    """
    waves = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spectra = np.array([
        [0.0, 2.0, 4.0, 2.0, 0.0],
        [1.0, 3.0, 1.0, 5.0, 1.0],
    ])
    return spectra, waves


def test_single_intensity_nearest(synth):
    s, w = synth
    # 2.2 → 가장 가까운 파수 2 (idx2).
    assert np.array_equal(extract.single_intensity(s, w, 2.2), np.array([4.0, 1.0]))
    # 0.4 → 파수 0.
    assert np.array_equal(extract.single_intensity(s, w, 0.4), np.array([0.0, 1.0]))


def test_peak_max(synth):
    s, w = synth
    got = extract.peak_max(s, w, 1.0, 3.0)  # 파수 1,2,3
    assert np.array_equal(got, np.array([4.0, 5.0]))


def test_peak_position(synth):
    s, w = synth
    got = extract.peak_position(s, w, 1.0, 3.0)
    assert np.array_equal(got, np.array([2.0, 3.0]))


def test_peak_area_full(synth):
    s, w = synth
    got = extract.peak_area(s, w, 0.0, 4.0)
    # 손계산: row0=8, row1=10.
    assert np.allclose(got, np.array([8.0, 10.0]))


def test_peak_area_window(synth):
    s, w = synth
    got = extract.peak_area(s, w, 1.0, 3.0)  # 파수 1,2,3
    # row0 [2,4,2] → 6 ; row1 [3,1,5] → 5.
    assert np.allclose(got, np.array([6.0, 5.0]))


def test_ratio_max(synth):
    s, w = synth
    # A=[1,3] → [4,5], B=[3,4] → row0 max([2,0])=2, row1 max([5,1])=5.
    got = extract.ratio(s, w, 1.0, 3.0, 3.0, 4.0, metric="max")
    assert np.allclose(got, np.array([2.0, 1.0]))


def test_ratio_area(synth):
    s, w = synth
    got = extract.ratio(s, w, 1.0, 3.0, 1.0, 3.0, metric="area")
    # 같은 구간 → 1.0.
    assert np.allclose(got, np.array([1.0, 1.0]))


def test_ratio_divide_by_zero():
    waves = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    spectra = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],  # B 구간 max=0 → nan
        [1.0, 1.0, 4.0, 1.0, 1.0],
    ])
    got = extract.ratio(spectra, waves, 1.0, 3.0, 3.0, 4.0, metric="max")
    assert np.isnan(got[0])
    assert np.isfinite(got[1])


def test_fwhm_symmetric():
    waves = np.arange(0.0, 11.0)  # 0..10
    row = np.array([0, 0, 0, 2, 4, 6, 4, 2, 0, 0, 0], dtype=float)
    spectra = np.vstack([row, row])
    got = extract.fwhm(spectra, waves, 0.0, 10.0)
    # baseline 0, max 6, half 3 → 좌 3.5, 우 6.5 → 3.0.
    assert np.allclose(got, np.array([3.0, 3.0]))


def test_fwhm_no_crossing_nan():
    # 단조 증가 → 왼쪽 교차 없음 → nan.
    waves = np.arange(0.0, 5.0)
    spectra = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    got = extract.fwhm(spectra, waves, 0.0, 4.0)
    assert np.isnan(got[0])


def test_extract_values_dispatch(synth):
    s, w = synth
    assert np.array_equal(
        extract.extract_values(s, w, "peak_max", w1=1.0, w2=3.0),
        np.array([4.0, 5.0]),
    )
    assert np.array_equal(
        extract.extract_values(s, w, "single", wave=2.0),
        np.array([4.0, 1.0]),
    )
    assert np.allclose(
        extract.extract_values(s, w, "ratio", a1=1.0, a2=3.0, b1=3.0, b2=4.0),
        np.array([2.0, 1.0]),
    )


def test_bad_range_raises(synth):
    s, w = synth
    with pytest.raises(ValueError):
        extract.peak_max(s, w, 3.0, 1.0)  # w1 >= w2
    with pytest.raises(ValueError):
        extract.extract_values(s, w, "peak_max", w1=3.0, w2=1.0)
    with pytest.raises(ValueError):
        extract.peak_max(s, w, 100.0, 200.0)  # 범위 밖 → 데이터 없음


def test_unknown_mode_and_missing_param(synth):
    s, w = synth
    with pytest.raises(ValueError):
        extract.extract_values(s, w, "bogus", w1=1.0, w2=2.0)
    with pytest.raises(ValueError):
        extract.extract_values(s, w, "peak_max")  # 파라미터 누락


def test_inputs_not_mutated(synth):
    s, w = synth
    s0, w0 = s.copy(), w.copy()
    extract.extract_values(s, w, "peak_area", w1=0.0, w2=4.0)
    extract.single_intensity(s, w, 2.0)
    assert np.array_equal(s, s0)
    assert np.array_equal(w, w0)
