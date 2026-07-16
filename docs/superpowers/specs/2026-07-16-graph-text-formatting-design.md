# 그래프 텍스트 서식 심화 (Origin 스타일) — 설계

작성일: 2026-07-16

## 배경 / 목적

히트맵·3D 표면 서식에 이미 **폰트(family)** 와 **요소별 글씨 크기(라벨/눈금/제목)** 가
있다. Origin Lab의 그래프 편집 기본기에 맞춰, 여기에 **요소별 굵기(Bold)·기울기
(Italic)·색상(Color)** 과 **위/아래첨자**를 추가한다.

범위 밖(YAGNI): 축 범위·눈금선 서식, 격자선, 드래그+단축키 리치텍스트 에디터.
(첨자는 드래그+Ctrl+Shift+= UX 대신 **평문 마크업**으로 처리 — 근거는 아래.)

## 요구사항

1. 라벨(축 라벨 + 컬러바 라벨) / 눈금(tick 숫자) / 제목 **각각** Bold·Italic·Color 지정.
2. 위/아래첨자: `Raman Shift (cm^{-1})` → `cm⁻¹` 처럼 렌더. **화면 == 내보내기** 보장.
3. 2D 히트맵·3D 표면·컬러바·matplotlib 폴백 **모두** 일관 적용.
4. 프리셋 저장/불러오기에 자동 포함(기존 `fmt_` 접두사 규칙 재사용).
5. 기존 포맷/서식 무회귀.

## 결정 근거: 첨자 입력 = 마크업

Streamlit `st.text_input` 은 브라우저 **평문 입력칸**이라 드래그 선택 + Ctrl+Shift+=
같은 서식 단축키를 지원하지 않는다. 그 UX는 `contenteditable` 커스텀 컴포넌트를
새로 만들어야 하므로 범위에서 제외. 대신 렌더러 중립 **통일 마크업**을 채택:

- `^{...}` → 위첨자, `_{...}` → 아래첨자.
- 오직 `^{` / `_{` 시퀀스만 특수 취급 → 맨 `^`, `_` 는 리터럴 유지(파일명·`a.u.` 등 안전).
- 적용 대상 텍스트: X/Y 축 라벨, 제목, **컬러바 라벨**. (눈금 숫자는 대상 아님.)

## 컴포넌트 설계

### 1) `core/plot.py` — 순수 변환 함수 (신규)

```python
def apply_text_markup(text: str, target: str) -> str:
    """'^{...}'/'_{...}' 마크업을 렌더러 문법으로 변환.
    target='plotly': ^{x}->'<sup>x</sup>', _{x}->'<sub>x</sub>'
    target='mpl'  : ^{x}->'$^{x}$',      _{x}->'$_{x}$'
    마크업이 없으면 원문 그대로 반환(특히 mpl에서 불필요한 $...$ 미삽입).
    """
```

- 정규식: `r"\^\{([^}]*)\}"`, `r"_\{([^}]*)\}"` 두 번 치환.
- plotly: 태그로 치환. 나머지 텍스트는 기존과 동일하게 원문 유지(추가 이스케이프 없음 = 무회귀).
- mpl: 매칭 토큰만 `$...$`로 감싸 인라인 mathtext. 매칭이 없으면 원문 반환.
- 순수 함수 → 단위 테스트 용이.

### 2) `core/plot.py` — `PlotConfig` 필드 추가

기존 `font_size_label/tick/title`, `font_family` 옆에 요소별 9개 추가:

```python
font_bold_label:  bool = False
font_italic_label: bool = False
font_color_label: str = "#000000"
font_bold_tick:   bool = False
font_italic_tick: bool = False
font_color_tick:  str = "#000000"
font_bold_title:  bool = False
font_italic_title: bool = False
font_color_title: str = "#000000"
```

기본 색 `#000000`(검정). 현재는 색 미지정이라 렌더러 기본색(진회색)이 나오는데,
검정으로 명시되면서 미세하게 진해진다 — 의도된 기본값으로 수용.

### 3) `core/plot.py` — 렌더 적용

plotly 6.9 / matplotlib 3.11 모두 라벨·눈금·제목에 weight·style·color 지원(검증됨).

- **plotly** (`make_heatmap` 2D, `make_surface` 3D, 컬러바):
  - 축 title: `font=dict(family, size, color=color_label, weight='bold'|'normal',
    style='italic'|'normal')`, 텍스트는 `apply_text_markup(text,'plotly')`.
  - tickfont: `dict(family, size, color=color_tick, weight, style)`.
  - 제목: title.font 에 title 요소 서식 + `apply_text_markup(title,'plotly')`.
  - 컬러바 title: 라벨 요소 서식 + 마크업.
- **matplotlib** (`make_matplotlib_heatmap`, `make_matplotlib_surface`, 컬러바):
  - `set_xlabel/ylabel/zlabel(apply_text_markup(text,'mpl'), fontsize, fontfamily,
    fontweight='bold'|'normal', fontstyle='italic'|'normal', color=color_label)`.
  - `set_title(apply_text_markup(title,'mpl'), ... 제목 요소 서식)`.
  - 눈금: `ax.tick_params(labelsize, labelcolor=color_tick)` +
    `plt.setp(ax.get_xticklabels()+ax.get_yticklabels(), fontweight=, fontstyle=)`.
  - 컬러바 라벨: 라벨 요소 서식 + 마크업.

### 4) `app.py` — DEFAULTS + UI

- `DEFAULTS` 에 9개 키 추가(`fmt_bold_label` 등, 값은 위 기본값).
- "⑤ 히트맵 서식" 폰트 블록의 크기 3열(`fs1/fs2/fs3`)을 **요소별 3행**으로 교체:

```
라벨  [크기]  ☐B  ☐I  🎨
눈금  [크기]  ☐B  ☐I  🎨
제목  [크기]  ☐B  ☐I  🎨
```

각 행 = `st.columns([1.2, 1, 0.6, 0.6, 1])` 로 [이름][number_input 크기]
[checkbox B][checkbox I][color_picker]. 첨자 마크업 힌트는 텍스트 입력칸 아래
`st.caption("위첨자 ^{ }, 아래첨자 _{ }  예: cm^{-1}")`.

- `build_plot_config()` 가 새 세션 키를 `PlotConfig` 로 전달.

### 5) 프리셋

`full_settings_dict()`/`apply_settings_dict()` 는 `k.startswith("fmt_")` 로 format
키를 일괄 처리하므로 **코드 변경 없이** 새 키가 저장/복원된다. (회귀 확인만.)

## 데이터 흐름

세션(`fmt_*`) → `build_plot_config()` → `PlotConfig` → `make_heatmap/make_surface`
(화면) 및 `export_image_bytes`/`mpl_bytes`(내보내기). `apply_text_markup` 는 렌더 직전
텍스트에만 적용 → 세션에는 항상 원문(마크업) 저장 → 프리셋 왕복 안전.

## 에러 처리 / 엣지

- 마크업 미종료(`cm^{-1`): 매칭 실패 → 원문 그대로(깨지지 않음).
- 첨자 내용에 공백: mpl mathtext에서 문제될 수 있어 caption에 "첨자 안엔 공백 지양" 안내.
- 잘못된 색 문자열: color_picker 사용이라 항상 유효한 hex.

## 테스트 (TDD)

신규 `tests/test_plot_format.py`:
1. `apply_text_markup("cm^{-1}","plotly") == "cm<sup>-1</sup>"`
2. `apply_text_markup("cm^{-1}","mpl") == "cm$^{-1}$"`
3. `apply_text_markup("H_{2}O","plotly") == "H<sub>2</sub>O"` / mpl `"H$_{2}$O"`
4. 마크업 없음 → plotly·mpl 모두 원문 그대로(특히 mpl에 `$` 미삽입)
5. 다중 토큰 `"a^{2}+b_{n}"` 양쪽 변환
6. 미종료 마크업 → 원문 반환
7. (경량) `make_heatmap`/`make_matplotlib_heatmap` 가 굵기·색 설정으로 예외 없이 figure 생성

기존 전체 스위트 무회귀(현재 42 passed 유지).

## 변경 파일

- `core/plot.py` — `apply_text_markup`, `PlotConfig` 9필드, 렌더 4곳 + 컬러바 적용.
- `app.py` — `DEFAULTS` 9키, 서식 UI 3행, `build_plot_config` 전달.
- `tests/test_plot_format.py` — 신규.
