"""loader.py 파일 포맷 로딩 테스트 — 실제 샘플 파일 기반.

새 'labeled equipment' 포맷(라벨 열 + 라벨 파수축 행) 지원 및
기존 equipment 포맷 무회귀(regression) 검증.
"""

from pathlib import Path

import numpy as np
import pytest

from core import loader

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
NEW_CSV = SAMPLE_DIR / "another format.csv"
NEW_TXT = SAMPLE_DIR / "another format.txt"
LEGACY_CSV = SAMPLE_DIR / "sample raman mapping raw data.csv"


def test_new_format_csv():
    data = loader.load_file(str(NEW_CSV))
    assert data.source_format == "equipment"
    assert data.spectra.shape == (400, 1024)
    assert data.n_waves == 1024
    assert data.map_width == 20
    assert data.map_height == 20
    assert data.step_x == 1.5
    assert data.step_y == 1.5
    assert data.coords is not None
    assert data.coords.shape == (400, 2)
    # 파수 오름차순
    assert np.all(np.diff(data.waves) > 0)
    # 원본 첫 데이터 셀(파수축이 이미 오름차순이라 열 순서 보존)
    assert data.spectra[0, 0] == 807.0


def test_new_format_txt():
    data = loader.load_file(str(NEW_TXT))
    assert data.source_format == "equipment"
    assert data.spectra.shape == (400, 1024)
    assert data.n_waves == 1024
    assert data.map_width == 20
    assert data.map_height == 20
    assert data.step_x == 1.5
    assert data.step_y == 1.5
    assert data.coords is not None
    assert data.coords.shape == (400, 2)
    assert np.all(np.diff(data.waves) > 0)
    assert data.spectra[0, 0] == 807.0


def test_new_format_csv_txt_equivalent():
    csv_data = loader.load_file(str(NEW_CSV))
    txt_data = loader.load_file(str(NEW_TXT))
    assert np.allclose(csv_data.waves, txt_data.waves)
    assert np.allclose(csv_data.spectra, txt_data.spectra)


def test_legacy_equipment_regression():
    data = loader.load_file(str(LEGACY_CSV))
    assert data.source_format == "equipment"
    assert data.spectra.shape == (400, 1024)


def test_validate_grid_exact_and_leftover():
    # 정확히 일치 → OK
    loader.validate_grid(400, 20, 20)
    # 그리드가 포인트보다 작음(초과 포인트는 호출부에서 잘라 씀) → OK
    loader.validate_grid(400, 10, 10)   # 100 <= 400
    loader.validate_grid(400, 19, 20)   # 380 <= 400


def test_validate_grid_rejects_oversized():
    # 그리드가 포인트 수보다 큼 → 값 부족이므로 에러
    with pytest.raises(ValueError):
        loader.validate_grid(100, 20, 20)   # 400 > 100
