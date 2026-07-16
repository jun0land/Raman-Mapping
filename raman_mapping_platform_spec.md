# 라만 매핑 데이터 가공·시각화 플랫폼 — 개발 사양서

> Claude Code 전달용 프로젝트 명세. 이 문서를 프로젝트 루트에 두고 작업 시작.

---

## 1. 프로젝트 개요

라만 분광 매핑 측정 결과(엑셀)를 업로드하면, 조건 입력만으로 2D 매핑 이미지를 생성하는 웹 기반 플랫폼.
기존에 Origin에서 수동으로 하던 "reshape → 방향 정렬 → matrix 입력 → 그래프 서식 조정" 과정을 자동화하는 것이 목표.

**기술 스택**

| 항목 | 선택 |
|---|---|
| 언어 | Python 3.11+ |
| UI | Streamlit |
| 데이터 처리 | pandas, numpy |
| 시각화 | Plotly (인터랙티브) + Matplotlib (논문용 고해상도 export) |
| 신호처리 | scipy (baseline, smoothing, peak) |
| 패키지 관리 | uv 또는 pip + requirements.txt |

---

## 2. 데이터 사양

### 입력
- 형식: `.xlsx`, `.csv`, `.txt` (탭 구분)
- 측정: N×N 그리드 매핑 (기본 20×20 = 400 포인트), 각 포인트마다 파수별 intensity 스펙트럼
- **예상 포맷 (둘 다 지원해야 함)**
  - **Wide**: 1열 = 파수(cm⁻¹), 2~401열 = 각 측정 포인트의 intensity
  - **Long**: `X, Y, wavenumber, intensity` 컬럼 구조
- ⚠️ 실제 샘플 파일 확인 후 파서 확정할 것. 헤더 행 위치, 단위 행 유무 등 장비별 편차 있음.

### 내부 표준 자료구조
```
spectra : np.ndarray, shape = (n_points, n_wavenumbers)
waves   : np.ndarray, shape = (n_wavenumbers,)
grid    : np.ndarray, shape = (ny, nx)   # 스칼라 매핑 값
```

---

## 3. 처리 파이프라인

### Step 1. 로드 & 파싱
- 포맷 자동 감지 (wide/long)
- 파수 축 오름차순 정렬
- 포인트 개수와 그리드 크기 정합성 검증 (`n_points == nx * ny`)

### Step 2. 전처리 (선택, 토글)
- **Baseline 보정**: asymmetric least squares(ALS) 또는 polynomial fit
- **Smoothing**: Savitzky-Golay 필터 (window, order 조절)
- **Normalization**: max normalize / 특정 피크 기준 / off
- **Cosmic ray 제거**: 인접 스펙트럼 median 대비 이상치 검출 (선택)

### Step 3. 매핑 값 추출 (핵심 "조건" 입력부)
드롭다운으로 모드 선택:

| 모드 | 파라미터 | 설명 |
|---|---|---|
| Single intensity | 파수 1개 | 해당 파수의 intensity |
| Peak max | 파수 구간 [w1, w2] | 구간 내 최대값 |
| Peak area | 파수 구간 [w1, w2] | 구간 적분값 (사다리꼴) |
| Peak position | 파수 구간 | 구간 내 최대값의 위치(cm⁻¹) — 스트레인 매핑용 |
| FWHM | 파수 구간 | 반치폭 |
| Ratio | 구간 A, 구간 B | A/B 비율 (예: I_D/I_G) |

→ 출력: 길이 400의 1D 배열

### Step 4. Reshape & 방향 정렬 ⭐ 가장 중요
optic 뷰(현미경 이미지)와 매핑 이미지의 방향을 일치시키는 단계.

```python
grid = values.reshape(ny, nx)          # 기본 raster
if scan_mode == "snake":
    grid[1::2] = grid[1::2, ::-1]      # 짝수 행 좌우 반전
```

UI에 아래 조작 버튼을 **실시간 미리보기와 함께** 제공:
- 스캔 방식: `raster` / `snake(serpentine)`
- 시작점: `top-left` / `bottom-left` / `top-right` / `bottom-right`
- `flip vertical` (np.flipud)
- `flip horizontal` (np.fliplr)
- `rotate 90° CW / CCW`
- `transpose`

> 장비(WITec / Horiba / Renishaw / Nanophoton)마다 스캔 순서가 다르므로 자동 판별은 불가.
> 사용자가 optic 이미지와 비교하며 맞추고, 그 설정을 **프리셋으로 저장/불러오기** 가능하게 할 것.

### Step 5. 시각화
Plotly `go.Heatmap` 기반. Origin matrix plot 수준의 서식 제어 제공:

- **Color scale**: min / max 수동 입력 + auto(percentile 2~98%) 버튼
- **Colormap**: jet, viridis, plasma, inferno, rainbow, gray, RdBu 등 선택
- **축**
  - X, Y 축 이름 (기본 "X (μm)", "Y (μm)")
  - 실제 물리 좌표 스케일 입력 (step size μm → 축 눈금 자동 변환)
  - tick 간격, 눈금 표시 여부
- **Colorbar (Z축)**: 라벨명 (예: "Intensity (a.u.)", "I_D/I_G"), tick 개수
- **폰트**: family(Arial, Times New Roman 등), size (축 라벨 / 눈금 / 타이틀 각각)
- **Interpolation**: none(픽셀) / bilinear smoothing 토글
- **Aspect ratio**: 1:1 고정 옵션

### Step 6. 부가 기능
- **스펙트럼 뷰어**: 히트맵의 픽셀 클릭 → 해당 포인트의 원본 스펙트럼 표시 (품질 검증용)
- **Export**
  - 이미지: PNG (DPI 지정), SVG, PDF
  - 데이터: 정렬 완료된 N×N matrix → CSV / XLSX (**Origin에 그대로 붙여넣기 가능하도록**)
  - 설정: 현재 조건 전체를 JSON으로 저장/불러오기
- **배치 처리**: 여러 파일 업로드 → 동일 조건 일괄 적용 → ZIP 다운로드

---

## 4. 프로젝트 구조

```
Raman Mapping/
├── app.py                  # Streamlit 엔트리포인트, UI 레이아웃
├── core/
│   ├── loader.py           # 엑셀/CSV 파싱, 포맷 감지
│   ├── preprocess.py       # baseline, smoothing, normalize
│   ├── extract.py          # 매핑 값 추출 (intensity/area/ratio/FWHM...)
│   ├── grid.py             # reshape, snake 보정, flip/rotate
│   └── plot.py             # Plotly / Matplotlib figure 생성
├── presets/                # 장비별 방향 설정 프리셋 JSON
├── tests/
│   ├── test_grid.py        # ⭐ reshape/snake/flip 로직 단위 테스트 필수
│   └── test_extract.py
├── sample_data/
└── requirements.txt
```

---

## 5. 개발 순서

1. **[선행] 실제 샘플 데이터 1개 확보** → `loader.py` 파서 확정
2. `core/` 모듈을 순수 함수로 먼저 구현 + `tests/` 단위 테스트
   - 특히 `grid.py`는 4×4 더미 배열로 snake/flip 동작을 반드시 검증
3. Jupyter나 CLI 스크립트로 파이프라인 end-to-end 검증
4. `app.py`로 Streamlit UI 래핑
5. Export / 프리셋 / 배치 기능 추가
6. 배포: 로컬 실행(`streamlit run app.py`) → 필요 시 랩 서버 또는 Streamlit Cloud

---

## 6. 개발 시 주의사항

- **UI에 무거운 연산 두지 말 것**: `@st.cache_data`로 파일 로드·전처리 결과 캐싱. 파라미터 바꿀 때마다 400개 스펙트럼 재계산하면 느려짐.
- **방향 정렬은 절대 추측하지 말 것**: 자동화 대신 사용자 조작 + 실시간 미리보기.
- **원본 데이터 불변**: 모든 처리는 복사본에서. 원본 배열은 세션 내내 보존.
- **단위 표기 일관성**: 파수는 cm⁻¹, 좌표는 μm.
- **에러 메시지 친절하게**: "포인트 수(384)가 그리드(20×20=400)와 맞지 않습니다" 수준으로 구체적으로.

---

## 7. Claude Code 시작 프롬프트 (복붙용)

```
이 프로젝트 사양서(raman_mapping_platform_spec.md)를 읽고 라만 매핑 시각화
Streamlit 앱을 개발해줘.

작업 순서:
1. 먼저 sample_data/ 의 엑셀 파일 구조를 파악해서 loader.py 파서를 만들어줘.
2. core/ 모듈들을 순수 함수로 구현하고, 특히 grid.py의 snake/flip 로직은
   단위 테스트를 함께 작성해줘.
3. 파이프라인이 검증되면 app.py로 Streamlit UI를 만들어줘.

각 단계마다 결과를 보여주고 진행 여부를 확인해줘.
```
