"""라만 매핑 원본 데이터 로더 (순수 코어 모듈).

지원 파일: .csv / .txt (탭·콤마 구분) / .xlsx
지원 레이아웃(자동 감지):
  1. equipment : 장비 메타데이터 블록 + 빈 줄 + 파수축 행 + 스펙트럼 행들
                 (각 행 = 측정 포인트, 각 열 = 파수). 실제 샘플 포맷.
  2. wide      : 1열 = 파수, 2..N열 = 각 포인트 intensity (포인트 = 열).
  3. long      : X, Y, wavenumber, intensity 컬럼 구조.

내부 표준 자료구조로 변환:
  spectra : np.ndarray (n_points, n_waves)
  waves   : np.ndarray (n_waves,)  — 항상 오름차순 정렬, spectra 열도 함께 정렬

Streamlit 등 UI 의존성 없음. 순수 함수. 원본 데이터는 변형하지 않는다.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["RamanData", "load_file", "validate_grid"]


# ---------------------------------------------------------------------------
# 자료구조
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RamanData:
    """로드된 라만 매핑 데이터의 내부 표준 표현.

    Attributes:
        spectra: (n_points, n_waves) intensity 배열.
        waves: (n_waves,) 파수 배열, 오름차순.
        metadata: 헤더 블록에서 파싱한 key/value (키는 정규화됨).
        source_format: 'equipment' | 'wide' | 'long'.
        filename: 원본 파일명(있으면).
        map_width: 그리드 힌트(맵 가로), 파싱 가능 시 float.
        map_height: 그리드 힌트(맵 세로), 파싱 가능 시 float.
        step_x: X 스텝 크기(µm), 파싱 가능 시 float.
        step_y: Y 스텝 크기(µm), 파싱 가능 시 float.
        coords: long 포맷일 때 각 포인트의 (X, Y) 좌표 (n_points, 2), 그 외 None.
    """

    spectra: np.ndarray
    waves: np.ndarray
    metadata: dict = field(default_factory=dict)
    source_format: str = ""
    filename: Optional[str] = None
    map_width: Optional[float] = None
    map_height: Optional[float] = None
    step_x: Optional[float] = None
    step_y: Optional[float] = None
    coords: Optional[np.ndarray] = None

    @property
    def n_points(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_waves(self) -> int:
        return int(self.spectra.shape[1])


# ---------------------------------------------------------------------------
# 저수준 헬퍼
# ---------------------------------------------------------------------------
def _is_blank(cell: Any) -> bool:
    """빈 셀 여부 (None / 빈 문자열 / NaN)."""
    if cell is None:
        return True
    if isinstance(cell, float) and math.isnan(cell):
        return True
    return isinstance(cell, str) and cell.strip() == ""


def _is_number(cell: Any) -> bool:
    """숫자로 해석 가능한 셀인지."""
    if isinstance(cell, bool):
        return False
    if isinstance(cell, (int, float)):
        return not (isinstance(cell, float) and math.isnan(cell))
    if _is_blank(cell):
        return False
    try:
        float(str(cell).strip())
        return True
    except ValueError:
        return False


def _to_float(cell: Any) -> float:
    """셀을 float로. 빈 셀은 NaN."""
    if _is_blank(cell):
        return float("nan")
    if isinstance(cell, (int, float)):
        return float(cell)
    return float(str(cell).strip())


def _is_numeric_row(row: Sequence[Any], min_count: int = 2) -> bool:
    """빈 셀을 제외한 모든 셀이 숫자이고 숫자 개수가 min_count 이상인지."""
    nums = 0
    for cell in row:
        if _is_blank(cell):
            continue
        if not _is_number(cell):
            return False
        nums += 1
    return nums >= min_count


def _all_blank(row: Sequence[Any]) -> bool:
    return all(_is_blank(c) for c in row)


def _norm_key(text: str) -> str:
    """메타데이터 키 정규화: 소문자, 공백/슬래시 → 언더스코어."""
    key = str(text).strip().lower()
    key = re.sub(r"[\s/]+", "_", key)
    key = re.sub(r"[^\w㎛]+", "_", key)
    return key.strip("_")


def _extract_float(text: Any) -> Optional[float]:
    """'10', '1/1㎛', '1㎛', '0.50001 s' 등에서 첫 실수 추출."""
    if text is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", str(text))
    return float(m.group()) if m else None


def _extract_pair(text: Any) -> tuple[Optional[float], Optional[float]]:
    """'1/1㎛' 처럼 X/Y 두 값을 담은 문자열에서 (x, y) 추출."""
    nums = re.findall(r"[-+]?\d*\.?\d+", str(text))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        v = float(nums[0])
        return v, v
    return None, None


# ---------------------------------------------------------------------------
# 원본 → rows(List[List[object]]) 변환
# ---------------------------------------------------------------------------
def _sniff_delimiter(text: str) -> str:
    """탭 vs 콤마 구분자 추정 (앞부분 샘플 기준)."""
    sample = text[:8192]
    tabs = sample.count("\t")
    commas = sample.count(",")
    if tabs > commas:
        return "\t"
    # csv.Sniffer 보조 시도
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return ","


def _rows_from_text(text: str) -> list[list[str]]:
    """CSV/TXT 텍스트를 rows(문자열 셀)로 파싱."""
    delim = _sniff_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    return [list(r) for r in reader]


def _rows_from_xlsx(source: Any) -> list[list[Any]]:
    """xlsx 첫 시트를 rows(원시 객체 셀)로 파싱."""
    df = pd.read_excel(source, header=None, dtype=object)
    rows: list[list[Any]] = []
    for _, r in df.iterrows():
        rows.append([None if (isinstance(v, float) and math.isnan(v)) else v
                     for v in r.tolist()])
    return rows


def _read_rows(path_or_buffer: Any, ext: str) -> list[list[Any]]:
    """확장자에 따라 원본을 rows로 읽어온다."""
    if ext in (".xlsx", ".xlsm", ".xls"):
        return _rows_from_xlsx(path_or_buffer)

    # csv / txt : 텍스트로 디코드
    if hasattr(path_or_buffer, "read"):
        raw = path_or_buffer.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8-sig")
        else:
            text = raw
    else:
        with open(path_or_buffer, "r", encoding="utf-8-sig", newline="") as f:
            text = f.read()
    return _rows_from_text(text)


# ---------------------------------------------------------------------------
# 행렬 유틸
# ---------------------------------------------------------------------------
def _drop_trailing_blank_cols(block: list[list[Any]]) -> list[list[Any]]:
    """블록 전체에서 완전히 비어있는 (맨 끝) 열 제거 → trailing comma 처리."""
    if not block:
        return block
    ncols = max(len(r) for r in block)
    # 각 행을 동일 길이로 패딩
    padded = [list(r) + [None] * (ncols - len(r)) for r in block]
    keep = []
    for c in range(ncols):
        col_all_blank = all(_is_blank(padded[r][c]) for r in range(len(padded)))
        if not col_all_blank:
            keep.append(c)
    return [[row[c] for c in keep] for row in padded]


def _block_to_array(block: list[list[Any]]) -> np.ndarray:
    """정제된 숫자 블록을 float ndarray로."""
    block = _drop_trailing_blank_cols(block)
    ncols = max((len(r) for r in block), default=0)
    out = np.full((len(block), ncols), np.nan, dtype=float)
    for i, row in enumerate(block):
        for j, cell in enumerate(row):
            out[i, j] = _to_float(cell)
    return out


def _sort_waves(waves: np.ndarray, spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """waves 오름차순 정렬 + spectra 열 동기 정렬."""
    order = np.argsort(waves, kind="stable")
    if np.array_equal(order, np.arange(len(waves))):
        return waves, spectra
    return waves[order], spectra[:, order]


# ---------------------------------------------------------------------------
# 메타데이터 파싱
# ---------------------------------------------------------------------------
_GRID_KEYS = {
    "map_width": "map_width",
    "map_height": "map_height",
    "map_depth": "map_depth",
    "x_pixels": "x_pixels",
    "y_pixels": "y_pixels",
}


def _parse_metadata(leading: list[list[Any]]) -> dict:
    """헤더/메타 블록에서 key,value (및 key,value,key2,value2) 파싱."""
    meta: dict = {}
    for row in leading:
        cells = [c for c in row]
        # 완전 빈 줄 스킵
        if _all_blank(cells):
            continue
        # 뒤쪽 빈 셀 제거
        while cells and _is_blank(cells[-1]):
            cells.pop()
        if not cells:
            continue
        # 단일 토큰(섹션 헤더, 예: "Measuring Condition") → 마커로 저장
        if len(cells) == 1:
            meta.setdefault(_norm_key(cells[0]), "")
            continue
        # key,value 쌍을 2개씩 소비
        i = 0
        while i + 1 < len(cells) + 1 and i < len(cells):
            key = cells[i]
            if _is_blank(key):
                i += 1
                continue
            val = cells[i + 1] if i + 1 < len(cells) else ""
            meta[_norm_key(str(key))] = "" if _is_blank(val) else (
                val if not isinstance(val, str) else val.strip()
            )
            i += 2
    return meta


def _grid_hints(meta: dict) -> dict:
    """메타데이터에서 그리드/스텝 힌트 추출 (float)."""
    hints: dict = {"map_width": None, "map_height": None,
                   "step_x": None, "step_y": None}
    if "map_width" in meta:
        hints["map_width"] = _extract_float(meta["map_width"])
    if "map_height" in meta:
        hints["map_height"] = _extract_float(meta["map_height"])

    # Step X/Y 형태: 'step_x/y' → '1/1㎛'
    for k, v in meta.items():
        if k.startswith("step_x") and ("y" in k or k == "step_x/y" or k == "step_x_y"):
            sx, sy = _extract_pair(v)
            if sx is not None:
                hints["step_x"] = sx
            if sy is not None:
                hints["step_y"] = sy
    if hints["step_x"] is None and "step_x" in meta:
        hints["step_x"] = _extract_float(meta["step_x"])
    if hints["step_y"] is None and "step_y" in meta:
        hints["step_y"] = _extract_float(meta["step_y"])
    return hints


# ---------------------------------------------------------------------------
# 포맷별 파서
# ---------------------------------------------------------------------------
_WAVE_ALIASES = ("wavenumber", "wave", "raman", "shift", "cm-1", "cm_1", "cm")
_INT_ALIASES = ("intensity", "counts", "int", "value")


def _match_long_header(row: Sequence[Any]) -> Optional[dict]:
    """행이 long 포맷 헤더인지 판단하고 컬럼 인덱스 매핑 반환."""
    if _is_numeric_row(row):
        return None
    toks = [str(c).strip().lower() if not _is_blank(c) else "" for c in row]
    idx = {"x": None, "y": None, "w": None, "i": None}
    for j, t in enumerate(toks):
        if t == "x" and idx["x"] is None:
            idx["x"] = j
        elif t == "y" and idx["y"] is None:
            idx["y"] = j
        elif idx["w"] is None and any(a in t for a in _WAVE_ALIASES):
            idx["w"] = j
        elif idx["i"] is None and any(a in t for a in _INT_ALIASES):
            idx["i"] = j
    if all(v is not None for v in idx.values()):
        return idx
    return None


def _parse_long(rows: list[list[Any]], header_idx: int, cols: dict,
                filename: Optional[str]) -> RamanData:
    """long 포맷 → RamanData."""
    data = rows[header_idx + 1:]
    xs, ys, ws, iv = [], [], [], []
    for r in data:
        if _all_blank(r):
            continue
        try:
            xs.append(_to_float(r[cols["x"]]))
            ys.append(_to_float(r[cols["y"]]))
            ws.append(_to_float(r[cols["w"]]))
            iv.append(_to_float(r[cols["i"]]))
        except (IndexError, ValueError):
            continue
    df = pd.DataFrame({"x": xs, "y": ys, "w": ws, "i": iv})
    piv = df.pivot_table(index=["x", "y"], columns="w", values="i",
                         aggfunc="mean", sort=False)
    piv = piv.sort_index(axis=1)  # 파수 오름차순
    waves = piv.columns.to_numpy(dtype=float)
    spectra = piv.to_numpy(dtype=float)
    coords = np.array([[float(a), float(b)] for a, b in piv.index], dtype=float)
    return RamanData(spectra=spectra, waves=waves, metadata={},
                     source_format="long", filename=filename, coords=coords)


def _parse_equipment(leading: list[list[Any]], numeric_block: list[list[Any]],
                     filename: Optional[str]) -> RamanData:
    """equipment 포맷 → RamanData. 첫 숫자행 = 파수축, 이후 = 스펙트럼."""
    arr = _block_to_array(numeric_block)
    waves = arr[0]
    spectra = arr[1:]
    # 파수축이 NaN인 열(뒤쪽 등) 제거
    good = ~np.isnan(waves)
    if not good.all():
        waves = waves[good]
        spectra = spectra[:, good]
    waves, spectra = _sort_waves(waves, np.ascontiguousarray(spectra))
    meta = _parse_metadata(leading)
    hints = _grid_hints(meta)
    return RamanData(spectra=spectra, waves=waves, metadata=meta,
                     source_format="equipment", filename=filename, **hints)


def _parse_wide(leading: list[list[Any]], numeric_block: list[list[Any]],
                filename: Optional[str]) -> RamanData:
    """wide 포맷 → RamanData. 1열 = 파수, 나머지 열 = 포인트."""
    arr = _block_to_array(numeric_block)
    waves = arr[:, 0]
    spectra = arr[:, 1:].T  # (n_points, n_waves)
    good = ~np.isnan(waves)
    if not good.all():
        waves = waves[good]
        spectra = spectra[:, good]
    waves, spectra = _sort_waves(waves, np.ascontiguousarray(spectra))
    meta = _parse_metadata(leading)
    hints = _grid_hints(meta)
    return RamanData(spectra=spectra, waves=waves, metadata=meta,
                     source_format="wide", filename=filename, **hints)


def _is_labeled_numeric_row(row: Sequence[Any], min_count: int = 2) -> bool:
    """첫 셀이 텍스트 라벨이고 나머지 셀이 모두 숫자인 행인지 (labeled equipment).

    예) 'X Axis,546.57,549.28,…'  또는  '(0)(0)(0),807,791,…'
    첫 셀은 비어있지 않은 비숫자 라벨, row[1:] 은 (후행 빈 셀 제외) 모두 숫자이며
    숫자 개수가 min_count 이상이어야 한다.
    """
    cells = list(row)
    while cells and _is_blank(cells[-1]):
        cells.pop()
    if len(cells) < min_count + 1:
        return False
    if _is_blank(cells[0]) or _is_number(cells[0]):
        return False
    nums = 0
    for c in cells[1:]:
        if _is_blank(c):
            continue
        if not _is_number(c):
            return False
        nums += 1
    return nums >= min_count


def _is_coord_label(cell: Any) -> bool:
    """'(x)(y)' 형태의 좌표 라벨인지 (요약행 'Average' 등과 구분)."""
    return bool(re.search(r"\(-?\d+\)\s*\(-?\d+\)", str(cell)))


def _parse_point_labels(labels: Sequence[Any]) -> Optional[np.ndarray]:
    """'(x)(y)' / '(x)(y)(z)' 형태 라벨 리스트 → (n, 2) 좌표 배열. 실패 시 None."""
    coords: list[list[float]] = []
    for lab in labels:
        nums = re.findall(r"\((-?\d+)\)", str(lab))
        if len(nums) < 2:
            return None
        coords.append([float(nums[0]), float(nums[1])])
    if not coords:
        return None
    return np.array(coords, dtype=float)


def _parse_labeled_equipment(leading: list[list[Any]],
                             labeled_block: list[list[Any]],
                             filename: Optional[str]) -> RamanData:
    """labeled equipment 포맷 → RamanData.

    각 행 = [라벨, 숫자…]. 첫 라벨행 = 파수축('X Axis'), 이후 = 스펙트럼.
    데이터 행 라벨 '(x)(y)[(z)]' 에서 (x, y) 좌표를 파싱해 coords로 보관한다.
    내부 source_format 은 기존 파이프라인 호환을 위해 'equipment' 로 둔다.
    """
    labels = [row[0] for row in labeled_block]
    stripped = [list(row[1:]) for row in labeled_block]
    arr = _block_to_array(stripped)
    waves = arr[0]
    spectra = arr[1:]
    good = ~np.isnan(waves)
    if not good.all():
        waves = waves[good]
        spectra = spectra[:, good]
    waves, spectra = _sort_waves(waves, np.ascontiguousarray(spectra))
    # 데이터 행(파수축 제외) 라벨에서 좌표 파싱
    coords = _parse_point_labels(labels[1:])
    if coords is not None and coords.shape[0] != spectra.shape[0]:
        coords = None
    meta = _parse_metadata(leading)
    hints = _grid_hints(meta)
    return RamanData(spectra=spectra, waves=waves, metadata=meta,
                     source_format="equipment", filename=filename,
                     coords=coords, **hints)


# ---------------------------------------------------------------------------
# 포맷 감지 + 디스패치
# ---------------------------------------------------------------------------
def _detect_and_parse(rows: list[list[Any]], filename: Optional[str]) -> RamanData:
    # 완전 빈 후행 행 제거
    while rows and _all_blank(rows[-1]):
        rows.pop()
    if not rows:
        raise ValueError("빈 파일입니다: 데이터를 찾을 수 없습니다.")

    # 1) Long 헤더 탐색 (상단 몇 줄)
    for idx in range(min(len(rows), 10)):
        cols = _match_long_header(rows[idx])
        if cols is not None:
            return _parse_long(rows, idx, cols, filename)

    # 2) 첫 숫자행 위치 및 숫자 블록 확보
    first_num = None
    for i, r in enumerate(rows):
        if _is_numeric_row(r):
            first_num = i
            break
    if first_num is None:
        # 폴백: 라벨 있는 equipment (첫 셀=라벨, 나머지=숫자). 순수 숫자행이
        # 하나도 없을 때만 시도하므로 기존 포맷에는 영향이 없다.
        first_lab = None
        for i, r in enumerate(rows):
            if _is_labeled_numeric_row(r):
                first_lab = i
                break
        if first_lab is not None:
            # 첫 라벨행 = 파수축. 이후 좌표 라벨 '(x)(y)' 행만 데이터로 수집하고
            # 'Average' 등 요약행(비좌표 라벨)을 만나면 멈춘다.
            labeled_block = [rows[first_lab]]
            for r in rows[first_lab + 1:]:
                if _all_blank(r):
                    continue
                if _is_labeled_numeric_row(r) and _is_coord_label(r[0]):
                    labeled_block.append(r)
                else:
                    break
            return _parse_labeled_equipment(rows[:first_lab], labeled_block, filename)
        raise ValueError("숫자 데이터 행을 찾을 수 없습니다: 지원되지 않는 포맷입니다.")

    numeric_block = []
    for r in rows[first_num:]:
        if _is_numeric_row(r) or _all_blank(r):
            if _all_blank(r):
                continue
            numeric_block.append(r)
        else:
            break

    leading = rows[:first_num]
    leading_nonempty = [r for r in leading if not _all_blank(r)]
    blank_in_leading = any(_all_blank(r) for r in leading)

    # 3) equipment vs wide 판정
    #    - 메타 블록(비어있지 않은 선행행 2개 이상, 또는 선행부 빈 줄 존재) → equipment
    #    - 선행행 최대 1개(단일 헤더) → wide
    is_equipment = len(leading_nonempty) >= 2 or blank_in_leading
    if is_equipment:
        return _parse_equipment(leading, numeric_block, filename)
    return _parse_wide(leading, numeric_block, filename)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def load_file(path_or_buffer: Any, filename: Optional[str] = None) -> RamanData:
    """라만 매핑 파일을 로드하여 RamanData로 반환한다.

    Args:
        path_or_buffer: 파일 경로(str/os.PathLike) 또는 file-like 버퍼
            (예: Streamlit UploadedFile / BytesIO).
        filename: 버퍼 입력 시 확장자 판별에 사용할 파일명. 경로 입력 시 생략 가능.

    Returns:
        RamanData: spectra (n_points, n_waves), waves 오름차순 등.

    Raises:
        ValueError: 지원되지 않는 확장자이거나 데이터 파싱 실패 시.
    """
    # 확장자 판별
    name = filename
    if name is None and isinstance(path_or_buffer, (str, os.PathLike)):
        name = os.fspath(path_or_buffer)
    ext = os.path.splitext(name)[1].lower() if name else ""

    if ext == "" and isinstance(path_or_buffer, (str, os.PathLike)):
        ext = os.path.splitext(os.fspath(path_or_buffer))[1].lower()

    if ext not in (".csv", ".txt", ".tsv", ".xlsx", ".xlsm", ".xls", ""):
        raise ValueError(f"지원되지 않는 파일 형식입니다: '{ext}'")

    rows = _read_rows(path_or_buffer, ext if ext else ".csv")
    data = _detect_and_parse(rows, os.path.basename(name) if name else None)
    return data


def validate_grid(n_points: int, nx: int, ny: int) -> None:
    """포인트 수와 그리드 크기 정합성 검증.

    Args:
        n_points: 실제 측정 포인트 수.
        nx: 그리드 가로 셀 수.
        ny: 그리드 세로 셀 수.

    Raises:
        ValueError: nx*ny != n_points 일 때 친절한 한국어 메시지로 발생.
    """
    if nx * ny != n_points:
        raise ValueError(
            f"포인트 수({n_points})가 그리드({nx}×{ny}={nx * ny})와 맞지 않습니다"
        )
