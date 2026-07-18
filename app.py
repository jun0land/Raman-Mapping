"""Raman Mapping Studio — Streamlit UI (Stage 3).

라만 매핑 측정 결과를 업로드 → 전처리 → 매핑 값 추출 → 방향 정렬(실시간
미리보기) → 히트맵 서식 → 스펙트럼 뷰어 → Export/프리셋/배치 처리까지
수행하는 웹 앱. 무거운 연산은 core/*.py 순수 함수에 위임하고, 파일 로드·전처리
결과는 @st.cache_data로 캐싱한다. 원본 데이터는 세션 내내 불변.

2D 매핑 전용 (z축/depth/3D 없음).
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core import loader, preprocess, extract, grid, plot


def section(label: str, description: str = "") -> None:
    """섹션 헤더 — st.subheader(divider="orange") + caption.

    (streamlit-extras의 deprecated colored_header 대체. divider 색 orange는
    테마 accent #ed542b 와 시각적으로 일치.)
    """
    st.subheader(label, divider="orange")
    if description:
        st.caption(description)


def hint_label(text: str, info: str = "", warn: str = "") -> str:
    """라벨 + 호버 툴팁 아이콘 HTML.

    라벨을 위젯 왼쪽에 두는 배치에서는 위젯 label 을 collapsed 로 숨기므로
    Streamlit 기본 help 아이콘을 쓸 수 없다. 그래서 브라우저 기본 title 툴팁을
    쓰는 아이콘(❓ 설명 · ⚠️ 주의)을 라벨 옆에 직접 붙인다.
    (라벨이 보이는 위젯은 Streamlit 의 help= 를 그대로 쓰는 편이 낫다.)
    """
    out = f"<span style='font-weight:600'>{escape(text)}</span>"
    if warn:
        out += (f" <span title='{escape(warn)}' style='cursor:help'>⚠️</span>")
    if info:
        out += (f" <span title='{escape(info)}' style='cursor:help'>❓</span>")
    return out


# ===========================================================================
# 0. 상수 / 기본값
# ===========================================================================
APP_DIR = Path(__file__).resolve().parent
PRESET_DIR = APP_DIR / "presets"
PRESET_DIR.mkdir(exist_ok=True)

# NOTE: 3D 표면의 마우스 드래그 회전을 슬라이더로 되돌려 받던 커스텀 컴포넌트
# (components/surface3d, Plotly.js 직접 렌더)는 렌더가 비어 나오는 문제로 폐기되었다.
# 3D 는 이제 st.plotly_chart(make_surface(...), key="surf3d_plot") 로 렌더하며,
# 카메라 각도는 방위각·고도·줌 슬라이더(cam_azim/cam_elev/cam_zoom)로만 제어한다.
# 마우스 드래그 회전은 브라우저 전용(자유 관찰용)이며 슬라이더/내보내기에 반영되지 않는다.

ACCENT = "#ed542b"
COLORMAPS = ["jet", "viridis", "plasma", "inferno", "rainbow", "gray", "RdBu"]
MODES = ["single", "peak_max", "peak_area", "peak_position", "fwhm", "ratio"]
MODE_LABELS = {
    "single": "Single intensity (단일 파수)",
    "peak_max": "Peak max (구간 최대)",
    "peak_area": "Peak area (구간 적분)",
    "peak_position": "Peak position (피크 위치)",
    "fwhm": "FWHM (반치폭)",
    "ratio": "Ratio (구간 A/B 비율)",
}
SCANS = ["raster", "snake"]
STARTS = ["top-left", "bottom-left", "top-right", "bottom-right"]
OPS = ["flip_v", "flip_h", "rotate_cw", "rotate_ccw", "transpose"]
OP_LABELS = {
    "flip_v": "상하 반전 (flip vertical)",
    "flip_h": "좌우 반전 (flip horizontal)",
    "rotate_cw": "시계방향 90° 회전",
    "rotate_ccw": "반시계방향 90° 회전",
    "transpose": "전치 (transpose)",
}
FONTS = ["Arial", "Myriad Pro", "Times New Roman", "Pretendard", "Calibri",
         "Helvetica", "Nanum Gothic"]

# 데이터 정보(① 포맷 metric) 표시용 친화 라벨.
# loader 의 내부 source_format 값("equipment"/"wide"/"long")은 그대로 유지한다
# (테스트가 "equipment" 에 의존). 화면에 보이는 텍스트만 매핑한다.
FORMAT_LABELS = {"equipment": "WEVE", "wide": "Wide", "long": "Long"}

# 세션 기본값 (모든 위젯 key를 여기서 초기화 → 프리셋/최상단 파이프라인이 안전하게 읽음)
DEFAULTS = {
    "nx": 20, "ny": 20,
    # 전처리
    "pp_cosmic": False, "pp_cosmic_thr": 5.0, "pp_cosmic_win": 5,
    "pp_baseline": "off", "pp_als_lam": 100000.0, "pp_als_p": 0.01,
    "pp_als_niter": 10, "pp_poly_order": 3,
    "pp_smooth": False, "pp_smooth_win": 11, "pp_smooth_poly": 3,
    "pp_norm": "off", "pp_norm_peak": 1580.0,
    # 추출
    "ex_mode": "single",
    "ex_wave": 1580.0,
    "ex_w1": 1300.0, "ex_w2": 1400.0,
    "ex_a1": 1300.0, "ex_a2": 1400.0, "ex_b1": 1550.0, "ex_b2": 1650.0,
    "ex_metric": "max",
    # 방향
    "or_scan": "raster", "or_start": "top-left",
    "op_flip_v": False, "op_flip_h": False,
    "op_rotate_cw": False, "op_rotate_ccw": False, "op_transpose": False,
    # 서식
    "fmt_cmap": "jet", "fmt_zmin": 0.0, "fmt_zmax": 1.0, "fmt_zauto": True,
    "fmt_xlabel": "X (μm)", "fmt_ylabel": "Y (μm)", "fmt_title": "",
    # 3D 표면 Z축 라벨. 빈 값이면 Colorbar 라벨을 따른다.
    "fmt_zlabel": "",
    "fmt_title_pos": "center",
    "fmt_stepx": 1.0, "fmt_stepy": 1.0,
    "fmt_showticks": True, "fmt_tickspacing": 0.0,
    "fmt_cbarlabel": "Intensity (a.u.)", "fmt_cbarticks": 5,
    "fmt_font": "Arial", "fmt_fs_label": 30, "fmt_fs_tick": 30, "fmt_fs_title": 30,
    "fmt_bold_label": False, "fmt_italic_label": False, "fmt_color_label": "#000000",
    "fmt_bold_tick": False, "fmt_italic_tick": False, "fmt_color_tick": "#000000",
    "fmt_bold_title": False, "fmt_italic_title": False, "fmt_color_title": "#000000",
    "fmt_interp": "none",
    "fmt_fill": "픽셀(격자)", "fmt_contour_lines": True,
    # 서식 툴바의 '대상' 선택 — UI 상태일 뿐이라 fmt_ 접두사를 쓰지 않는다
    # (프리셋 저장/복원은 fmt_ 키만 대상으로 하므로 프리셋이 오염되지 않음).
    "tb_target": "제목",
    # export
    "exp_dpi": 300,
    "exp_fmt": "PNG (투명)",
    # 스펙트럼 뷰어 선택 픽셀
    "sv_row": 0, "sv_col": 0,
    # 시각화 보기 방식 (2D 히트맵 / 3D 표면)
    "view_mode": "2D 히트맵",
    # 3D 표면 카메라 (구면 좌표 → export 각도 동기화)
    "cam_azim": -45.0, "cam_elev": 25.0, "cam_zoom": 2.2,
}


def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ===========================================================================
# 1. 페이지 설정 + Liquid Glass CSS
# ===========================================================================
st.set_page_config(page_title="Raman Mapping Studio", layout="wide",
                   page_icon="🔬")
init_state()


# 이미지는 static/ 정적 서빙(.streamlit/config.toml 의 enableStaticServing)으로 제공한다.
# base64 로 CSS 에 인라인하면 브라우저가 캐시할 수 없어 전체 rerun 마다 재전송되지만,
# 정적 URL 이면 최초 1회만 내려받고 이후 캐시된다.
# (배경 원본 4096² PNG 8.9MB → 인라인 11.8MB 였던 것을 1600² JPEG 117KB 로 줄인 뒤
#  정적 서빙으로 전환 — 인라인 페이로드 0.)
_STATIC = "app/static"          # Streamlit 정적 서빙 경로 (앱 루트 기준 상대 URL)
_BG_URL = f"{_STATIC}/liquid_bg.jpg"
_LOGO_URL = f"{_STATIC}/logo.png"

# 파일이 없으면 배경 없이 그라데이션으로 우아하게 폴백한다.
_has_bg = (APP_DIR / "static" / "liquid_bg.jpg").exists()
_has_logo = (APP_DIR / "static" / "logo.png").exists()

_bg_layer = (
    f"linear-gradient(rgba(255,255,255,0.72), rgba(255,255,255,0.82)), "
    f"url('{_BG_URL}')"
    if _has_bg else
    "linear-gradient(135deg, #fdf0ec 0%, #f7f7fb 100%)"
)

custom_css = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

html, body, [class*="css"], .stApp, button, input, textarea, select {{
    font-family: 'Myriad Pro', 'Pretendard', 'Nanum Gothic', -apple-system, sans-serif !important;
}}

.stApp {{
    background: {_bg_layer};
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}}

/* 컨테이너 투명화 */
[data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
[data-testid="stAppViewContainer"], .main .block-container {{ background: transparent !important; }}

/* Glassmorphism */
[data-testid="stForm"],
[data-testid="stExpander"],
[data-testid="stVerticalBlockBorderWrapper"],
.title-glass-container {{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(48px) saturate(150%);
    -webkit-backdrop-filter: blur(48px) saturate(150%);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.05);
    padding: 24px;
}}

/* 사이드바 글래스 */
[data-testid="stSidebar"] > div:first-child {{
    background: rgba(255,255,255,0.28);
    backdrop-filter: blur(40px) saturate(150%);
    -webkit-backdrop-filter: blur(40px) saturate(150%);
}}

/* 타이틀 헤더 */
.title-glass-container {{
    display: flex; align-items: center; gap: 18px;
    border-left: 6px solid {ACCENT};
    margin-bottom: 18px;
}}
.title-glass-container img {{ height: 52px; width: auto; }}
.title-glass-container h2 {{
    margin: 0; font-weight: 800; color: #1c1c1e; text-shadow: none;
    letter-spacing: -0.5px;
}}
.title-glass-container .subtitle {{ color: #6b6b70; font-size: 0.92rem; }}

/* 탭 — 배경 하이라이트 제거(모든 상태 투명), 선택 표시는 텍스트색+밑줄만 */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    gap: 6px;
}}
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:active,
.stTabs [data-baseweb="tab"]:focus,
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: transparent !important;
    background-color: transparent !important;
    border-radius: 12px 12px 0 0; padding: 8px 20px; font-weight: 600;
}}
/* 내부 하이라이트 요소까지 투명 처리 */
.stTabs [data-baseweb="tab"] > div,
.stTabs [data-baseweb="tab"] [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-highlight"] {{
    background: transparent !important;
    background-color: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 3px solid {ACCENT} !important;
}}

/* 매뉴얼 JS 주입용 iframe — 화면 공간을 차지하지 않도록 컨테이너를 접는다.
   (display:none 대신 height:0 으로 두어 iframe 이 확실히 로드·실행되게 한다) */
.st-key-manual_inject {{
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}

/* 콘텐츠 고정 최대폭 + 가운데 정렬 — 넓은 모니터에서 여백만 늘고 레이아웃은 동일하게
   유지(공간만 늘어나 성겨 보이는 문제 방지). 좁은 화면에서는 자연스럽게 reflow. */
.block-container {{
    max-width: 1560px;
    margin: 0 auto;
    padding-left: 3rem;
    padding-right: 3rem;
}}

/* 파일 업로더 */
[data-testid="stFileUploader"] section {{
    background: rgba(255,255,255,0.25);
    border: 2px dashed {ACCENT};
    border-radius: 16px;
}}

/* 입력/셀렉트 글래스 */
[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{
    background: rgba(255,255,255,0.35) !important;
    border-radius: 10px !important;
}}

/* 버튼 */
.stButton > button, .stDownloadButton > button {{
    background: rgba(255,255,255,0.35);
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 12px; font-weight: 600; color: #1c1c1e;
    transition: all 0.18s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    transform: translateY(-2px);
    border-color: {ACCENT}; color: {ACCENT};
    box-shadow: 0 6px 18px rgba(237,84,43,0.18);
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT}, #f68b21);
    color: white; border: none;
}}
.stButton > button[kind="primary"]:hover {{ color: white; opacity: 0.94; }}

/* metric 카드 */
[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.30);
    border-radius: 14px; padding: 12px 16px;
    border: 1px solid rgba(255,255,255,0.4);
}}
/* metric 값/라벨 크기 축소 (① 데이터 정보 등 st.metric 전역) */
[data-testid="stMetricValue"] {{ font-size: 1.5rem !important; }}
[data-testid="stMetricLabel"] {{ font-size: 0.8rem !important; }}

h1, h2, h3, h4, p, label, span {{ text-shadow: none; }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ===========================================================================
# 1.5 사용 설명서 슬라이드-인 패널 (self-contained JS 오버레이)
# ===========================================================================
# Streamlit 의 st.markdown 은 <script> 를 제거하므로, 전체 화면 오버레이(북마크 탭·
# 딤 배경·슬라이드 패널)는 st.iframe 안의 JS 가 window.parent.document
# 에 직접 주입해 만든다. localhost 는 same-origin 이라 parent 접근이 가능하지만 모든
# parent 접근은 try/catch 로 감싼다(호스팅 환경이 cross-origin 이면 조용히 무시됨).
# 슬라이드 애니메이션은 순수 CSS transition(transform)으로 처리(파이썬 왕복 없음).
# 열림 상태는 window.parent.__nbedlManualOpen 플래그로 rerun 사이에 유지한다.
_MANUAL_HTML = r"""
<script>
(function () {
  try {
    var pwin = window.parent;
    var pdoc = pwin.document;
    if (!pdoc || !pdoc.body) return;

    // --- rerun 마다 REMOVE 후 RE-CREATE. 지우기 전에 현재 열림 상태를 읽어 복원 ---
    var wasOpen = false;
    var prevRoot = pdoc.getElementById('nbedl-manual-root');
    if (prevRoot) wasOpen = prevRoot.classList.contains('open');
    if (typeof pwin.__nbedlManualOpen === 'boolean') wasOpen = pwin.__nbedlManualOpen;

    var oldStyle = pdoc.getElementById('nbedl-manual-style');
    if (oldStyle) oldStyle.remove();
    if (prevRoot) prevRoot.remove();

    // --- 스타일 ---
    var style = pdoc.createElement('style');
    style.id = 'nbedl-manual-style';
    style.textContent = `
      #nbedl-manual-root, #nbedl-manual-root * { box-sizing: border-box;
        font-family: 'Pretendard', -apple-system, 'Nanum Gothic', sans-serif; }
      #nbedl-manual-tab {
        position: fixed; top: 28%; right: 0; z-index: 2147483000;
        writing-mode: vertical-rl; text-orientation: mixed;
        background: linear-gradient(160deg, #ed542b, #f68b21);
        color: #fff; font-weight: 700; letter-spacing: 2px; font-size: 15px;
        padding: 20px 11px; border-radius: 14px 0 0 14px; cursor: pointer;
        box-shadow: -4px 4px 18px rgba(0,0,0,0.20); user-select: none;
        transition: padding-right .2s ease, box-shadow .2s ease; }
      #nbedl-manual-tab:hover {
        padding-right: 15px; box-shadow: -6px 6px 22px rgba(237,84,43,0.35); }
      #nbedl-manual-backdrop {
        position: fixed; inset: 0; z-index: 2147483100;
        background: rgba(20,20,28,0.34);
        backdrop-filter: blur(6px) saturate(120%);
        -webkit-backdrop-filter: blur(6px) saturate(120%);
        opacity: 0; pointer-events: none; transition: opacity .38s ease; }
      #nbedl-manual-panel {
        position: fixed; top: 0; right: 0; height: 100vh; z-index: 2147483200;
        width: min(460px, 92vw);
        background: rgba(255,255,255,0.94);
        backdrop-filter: blur(26px) saturate(160%);
        -webkit-backdrop-filter: blur(26px) saturate(160%);
        border-left: 6px solid #ed542b;
        box-shadow: -14px 0 44px rgba(0,0,0,0.24);
        transform: translateX(105%);
        transition: transform .38s cubic-bezier(.22,.61,.36,1);
        overflow-y: auto; padding: 24px 26px 64px; color: #23252a; }
      #nbedl-manual-root.open #nbedl-manual-panel { transform: translateX(0); }
      #nbedl-manual-root.open #nbedl-manual-backdrop { opacity: 1; pointer-events: auto; }
      #nbedl-manual-close {
        position: absolute; top: 16px; right: 18px; width: 34px; height: 34px;
        border: none; border-radius: 50%; cursor: pointer; font-size: 17px;
        background: rgba(237,84,43,0.12); color: #ed542b; line-height: 1;
        transition: background .18s ease; }
      #nbedl-manual-close:hover { background: rgba(237,84,43,0.24); }
      #nbedl-manual-panel h3 { color: #ed542b; margin: 4px 40px 4px 0;
        font-size: 1.18rem; font-weight: 800; }
      #nbedl-manual-panel h4 { color: #ed542b; margin: 18px 0 5px;
        font-size: 1.0rem; font-weight: 700; }
      #nbedl-manual-panel p { margin: 4px 0; font-size: .9rem; line-height: 1.55; }
      #nbedl-manual-panel .nbedl-sub { color: #6b6b70; font-size: .86rem;
        margin-bottom: 6px; }
      #nbedl-manual-panel b { color: #c8431f; font-weight: 700; }
      #nbedl-manual-panel .nbedl-note {
        background: rgba(255,244,235,0.85); border-left: 4px solid #f68b21;
        border-radius: 8px; padding: 12px 14px 4px; margin-top: 20px; }
      #nbedl-manual-panel .nbedl-note b { color: #ed542b; }
      #nbedl-manual-panel .nbedl-note ul { margin: 8px 0 8px; padding-left: 18px; }
      #nbedl-manual-panel .nbedl-note li { font-size: .87rem; line-height: 1.5;
        margin-bottom: 5px; }
    `;
    pdoc.head.appendChild(style);

    // --- 루트(탭 + 배경 + 패널) ---
    var root = pdoc.createElement('div');
    root.id = 'nbedl-manual-root';
    root.innerHTML = `
      <div id="nbedl-manual-tab">📖 사용 설명서</div>
      <div id="nbedl-manual-backdrop"></div>
      <div id="nbedl-manual-panel">
        <button id="nbedl-manual-close" aria-label="닫기">✕</button>
        <h3>📖 Raman Mapping Studio · 사용 설명서</h3>
        <p class="nbedl-sub">아래 순서대로 따라 하면 측정 파일에서 논문·보고서용 매핑 이미지까지 만들 수 있습니다.</p>

        <h4>STEP 1 · 파일 업로드</h4>
        <p>왼쪽 사이드바 <b>⚙️ 전역 설정</b>에서 <b>.xlsx / .csv / .txt(.tsv)</b> 파일을 올립니다. 포맷은 자동 감지됩니다.</p>

        <h4>STEP 2 · 그리드 nx · ny 설정</h4>
        <p><b>nx(열) × ny(행)</b>이 측정 포인트 수와 정확히 일치해야 합니다. 안 맞으면 상단에 친절한 안내 에러가 뜨니 값을 조정하세요. <b>① 데이터 정보</b>의 포인트 수·메타 힌트를 참고하세요.</p>

        <h4>STEP 3 · 전처리 (선택 · 탭 ②)</h4>
        <p>cosmic ray 제거 · baseline(off / ALS / poly) · Savitzky–Golay 평활 · normalize 토글. <b>ALS는 400 스펙트럼 기준 약 4초</b>로 가장 무겁습니다. 빠른 작업엔 off·poly 권장, 동일 설정은 캐시로 즉시 반영됩니다.</p>

        <h4>STEP 4 · 매핑 값 추출 (③)</h4>
        <p>single · peak_max · peak_area · peak_position · fwhm · ratio 중 선택하고 해당 <b>파수(cm⁻¹) 구간</b>을 입력합니다. ratio는 A(분자)·B(분모) 두 구간을 지정합니다.</p>

        <h4>STEP 5 · 방향 정렬 ⭐ (④)</h4>
        <p>scan(raster / snake) · 시작 코너 · flip / rotate / transpose를 <b>optic 이미지와 비교하며</b> 맞춥니다. 아래 최종 뷰에 실시간 미리보기로 반영되니 눈으로 확인하며 조정하세요. <b>추측하지 말고</b> 실제 이미지와 대조합니다.</p>

        <h4>STEP 6 · 히트맵 서식 (⑤)</h4>
        <p>colormap · z-range(자동 2–98%) · 축 라벨과 μm step · colorbar · 폰트(Myriad Pro 등) · 보간 · 1:1 종횡비. 서식 변경은 무거운 재계산 없이 즉시 반영됩니다.</p>

        <h4>STEP 7 · 2D / 3D 뷰 + 카메라</h4>
        <p>보기 방식에서 2D 히트맵 ↔ 3D 표면을 전환. 3D는 <b>방위각 · 고도 · 줌</b> 슬라이더로 각도를 잡습니다. 마우스 드래그는 자유 관찰용이며, <b>Export 각도는 슬라이더·프리셋 값</b>이 기준입니다.</p>

        <h4>STEP 8 · 스펙트럼 뷰어 (⑥)</h4>
        <p>히트맵 픽셀을 <b>클릭</b>하거나 X / Y를 입력하면 그 지점의 <b>원본 스펙트럼(전처리 전)</b>을 확인해 QC할 수 있습니다.</p>

        <h4>STEP 9 · Export</h4>
        <p>사이드바 💾 Export: <b>PNG(투명) / JPG(흰 배경) / SVG / PDF</b> 이미지, <b>CSV · XLSX 매트릭스</b>(Origin에 그대로 붙여넣기), <b>설정 JSON</b>. 화면의 Plotly 뷰를 그대로 저장합니다.</p>

        <h4>STEP 10 · 프리셋 저장 · 불러오기</h4>
        <p>장비별 방향·서식 조합을 이름 붙여 저장하고 재사용하세요(예: WITec_100x). Reset으로 전체 초기화할 수 있습니다.</p>

        <h4>STEP 11 · 배치 처리 (탭 📦)</h4>
        <p>여러 파일에 현재 설정을 일괄 적용 → 파일당 <b>2D PNG + 3D PNG + CSV</b>를 ZIP으로 받습니다. nx · ny가 맞는 파일만 처리됩니다.</p>

        <div class="nbedl-note">
          <b>⚠️ 주의사항 & 팁</b>
          <ul>
            <li><b>원본 데이터는 불변</b> — 모든 전처리·변환은 사본에만 적용됩니다.</li>
            <li><b>ALS는 무겁습니다</b> — 필요할 때만 사용하고 캐시를 활용하세요.</li>
            <li><b>방향은 자동 판별 불가</b> — 장비마다 스캔 방식이 달라 optic 이미지로 직접 맞춰야 합니다.</li>
            <li><b>nx · ny 정합성</b>이 안 맞으면 매핑이 생성되지 않습니다.</li>
            <li>서식 · 카메라 조정은 <b>무거운 재계산 없이 즉시</b> 반영됩니다.</li>
          </ul>
        </div>
      </div>`;
    pdoc.body.appendChild(root);

    // --- 열기/닫기 (CSS transition 이 슬라이드 담당) ---
    function setOpen(o) {
      pwin.__nbedlManualOpen = o;
      if (o) root.classList.add('open'); else root.classList.remove('open');
    }
    // rerun 직후 복원: class 를 즉시 부여하므로 열려 있던 상태가 그대로 유지됨
    setOpen(wasOpen);

    root.querySelector('#nbedl-manual-tab')
        .addEventListener('click', function () { setOpen(true); });
    root.querySelector('#nbedl-manual-close')
        .addEventListener('click', function () { setOpen(false); });
    root.querySelector('#nbedl-manual-backdrop')
        .addEventListener('click', function () { setOpen(false); });

    // Esc 닫기 — parent document 에 한 번만 부착 (rerun 중복 방지)
    if (!pwin.__nbedlManualEsc) {
      pwin.__nbedlManualEsc = true;
      pdoc.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && pwin.__nbedlManualOpen) {
          var r = pdoc.getElementById('nbedl-manual-root');
          if (r) { r.classList.remove('open'); pwin.__nbedlManualOpen = false; }
        }
      });
    }
  } catch (e) { /* cross-origin 등: 조용히 무시 */ }
})();

// --- st.form 안에서 Enter 제출 차단 (input/select 만, textarea 제외) ---
(function () {
  try {
    var pwin = window.parent;
    var pdoc = pwin.document;
    if (pwin.__nbedlEnterGuard) return;   // de-dup: 한 번만 부착
    pwin.__nbedlEnterGuard = true;
    pdoc.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter') return;
      var t = e.target;
      if (!t) return;
      var tag = (t.tagName || '').toLowerCase();
      if (tag === 'textarea') return;               // 멀티라인은 허용
      if (tag !== 'input' && tag !== 'select') return;
      var inForm = t.closest && t.closest('[data-testid="stForm"]');
      if (inForm) { e.preventDefault(); e.stopImmediatePropagation(); }
    }, true);   // capture 단계
  } catch (e) { /* 무시 */ }
})();
</script>
"""


def render_manual_panel() -> None:
    """사용 설명서 슬라이드-인 패널 + st.form Enter-가드를 parent DOM 에 주입.

    app.py 최상단에서 한 번만 호출 → 모든 화면·탭에 북마크 탭이 노출된다.
    height=0 iframe 이라 레이아웃을 차지하지 않는다.
    """
    # st.iframe 은 height=0 을 거부한다(양수/'stretch'/'content'만 허용). 이 iframe 은
    # JS 주입 용도라 화면 공간을 차지하면 안 되므로, key 로 식별되는 컨테이너에 담고
    # CSS(.st-key-manual_inject)로 높이를 접는다. display:none 대신 height:0+overflow
    # 를 쓰는 이유는 iframe 이 확실히 로드·실행되게 하기 위해서다.
    with st.container(key="manual_inject"):
        st.iframe(_MANUAL_HTML, height=1)


# 최상단에서 1회 호출 (모든 탭/화면에 북마크 탭 상시 노출)
render_manual_panel()


# ===========================================================================
# 2. 캐싱된 로드/전처리 (파일 bytes + config 문자열로 키)
# ===========================================================================
@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, filename: str):
    """파일 바이트로부터 RamanData 로드 (내용 기반 캐시)."""
    return loader.load_file(io.BytesIO(file_bytes), filename=filename)


@st.cache_data(show_spinner="전처리 중… (400 스펙트럼)")
def cached_preprocess(file_bytes: bytes, filename: str, config_json: str) -> np.ndarray:
    """전처리 결과 캐시. config_json 이 같으면 재계산하지 않음."""
    rd = cached_load(file_bytes, filename)
    cfg = json.loads(config_json)
    return preprocess.apply_preprocessing(rd.spectra, rd.waves, cfg)


# ===========================================================================
# 3. 세션 → config dict 빌더
# ===========================================================================
def build_preprocess_config() -> dict:
    ss = st.session_state
    cfg: dict = {}
    if ss.pp_cosmic:
        cfg["cosmic"] = {"threshold": float(ss.pp_cosmic_thr),
                         "window": int(ss.pp_cosmic_win)}
    if ss.pp_baseline == "als":
        cfg["baseline"] = {"method": "als", "lam": float(ss.pp_als_lam),
                           "p": float(ss.pp_als_p), "niter": int(ss.pp_als_niter)}
    elif ss.pp_baseline == "poly":
        cfg["baseline"] = {"method": "poly", "order": int(ss.pp_poly_order)}
    if ss.pp_smooth:
        cfg["smooth"] = {"window": int(ss.pp_smooth_win), "poly": int(ss.pp_smooth_poly)}
    if ss.pp_norm == "max":
        cfg["normalize"] = {"mode": "max"}
    elif ss.pp_norm == "peak":
        cfg["normalize"] = {"mode": "peak", "peak_wave": float(ss.pp_norm_peak)}
    return cfg


def build_extract_params() -> tuple[str, dict]:
    ss = st.session_state
    mode = ss.ex_mode
    if mode == "single":
        return mode, {"wave": float(ss.ex_wave)}
    if mode in ("peak_max", "peak_area", "peak_position", "fwhm"):
        return mode, {"w1": float(ss.ex_w1), "w2": float(ss.ex_w2)}
    return mode, {"a1": float(ss.ex_a1), "a2": float(ss.ex_a2),
                  "b1": float(ss.ex_b1), "b2": float(ss.ex_b2),
                  "metric": ss.ex_metric}


def build_grid_config() -> dict:
    ss = st.session_state
    ops = [op for op in OPS if ss.get(f"op_{op}", False)]
    return {"scan": ss.or_scan, "start": ss.or_start, "ops": ops}


def build_plot_config(grid_arr: np.ndarray | None = None) -> plot.PlotConfig:
    ss = st.session_state
    zmin, zmax = None, None
    if not ss.fmt_zauto:
        zmin, zmax = float(ss.fmt_zmin), float(ss.fmt_zmax)
    elif grid_arr is not None:
        zmin, zmax = plot.auto_zrange(grid_arr)
    tick = float(ss.fmt_tickspacing) if ss.fmt_tickspacing and ss.fmt_tickspacing > 0 else None
    return plot.PlotConfig(
        colormap=ss.fmt_cmap, zmin=zmin, zmax=zmax,
        x_label=ss.fmt_xlabel, y_label=ss.fmt_ylabel, title=ss.fmt_title,
        z_label=ss.fmt_zlabel, title_pos=ss.fmt_title_pos,
        step_x=float(ss.fmt_stepx), step_y=float(ss.fmt_stepy),
        show_ticks=ss.fmt_showticks, tick_spacing=tick,
        colorbar_label=ss.fmt_cbarlabel, colorbar_ticks=int(ss.fmt_cbarticks),
        font_family=ss.fmt_font, font_size_label=int(ss.fmt_fs_label),
        font_size_tick=int(ss.fmt_fs_tick), font_size_title=int(ss.fmt_fs_title),
        font_bold_label=ss.fmt_bold_label, font_italic_label=ss.fmt_italic_label,
        font_color_label=ss.fmt_color_label,
        font_bold_tick=ss.fmt_bold_tick, font_italic_tick=ss.fmt_italic_tick,
        font_color_tick=ss.fmt_color_tick,
        font_bold_title=ss.fmt_bold_title, font_italic_title=ss.fmt_italic_title,
        font_color_title=ss.fmt_color_title,
        # 매핑 그리드는 픽셀 1개가 항상 정사각(1:1)이어야 하므로 종횡비를 상시 고정한다
        # (가로/세로로 늘릴 일이 없어 사용자 옵션으로 두지 않는다).
        interpolation=ss.fmt_interp, lock_aspect=True,
        fill_mode=("contour" if ss.fmt_fill == "등고선(contour)" else "pixel"),
        show_contour_lines=ss.fmt_contour_lines,
        cam_azim=float(ss.cam_azim), cam_elev=float(ss.cam_elev),
        cam_zoom=float(ss.cam_zoom),
    )


def full_settings_dict() -> dict:
    """현재 전체 설정을 하나의 JSON 직렬화 가능 dict로."""
    mode, ex_params = build_extract_params()
    return {
        "grid": {"nx": int(st.session_state.nx), "ny": int(st.session_state.ny)},
        "preprocess": build_preprocess_config(),
        "extract": {"mode": mode, "params": ex_params},
        "orientation": build_grid_config(),
        "format": {k: st.session_state[k] for k in DEFAULTS if k.startswith("fmt_")},
        "camera": {k: st.session_state[k] for k in DEFAULTS if k.startswith("cam_")},
    }


def apply_settings_dict(cfg: dict):
    """설정 dict를 세션 위젯 값으로 복원 (프리셋/JSON 불러오기)."""
    g = cfg.get("grid", {})
    st.session_state.nx = int(g.get("nx", st.session_state.nx))
    st.session_state.ny = int(g.get("ny", st.session_state.ny))

    pp = cfg.get("preprocess", {})
    st.session_state.pp_cosmic = "cosmic" in pp
    if "cosmic" in pp:
        st.session_state.pp_cosmic_thr = float(pp["cosmic"].get("threshold", 5.0))
        st.session_state.pp_cosmic_win = int(pp["cosmic"].get("window", 5))
    bl = pp.get("baseline") or {}
    st.session_state.pp_baseline = bl.get("method") or "off"
    if bl.get("method") == "als":
        st.session_state.pp_als_lam = float(bl.get("lam", 1e5))
        st.session_state.pp_als_p = float(bl.get("p", 0.01))
        st.session_state.pp_als_niter = int(bl.get("niter", 10))
    elif bl.get("method") == "poly":
        st.session_state.pp_poly_order = int(bl.get("order", 3))
    sm = pp.get("smooth")
    st.session_state.pp_smooth = bool(sm)
    if sm:
        st.session_state.pp_smooth_win = int(sm.get("window", 11))
        st.session_state.pp_smooth_poly = int(sm.get("poly", 3))
    nm = pp.get("normalize") or {}
    st.session_state.pp_norm = nm.get("mode", "off")
    if nm.get("mode") == "peak":
        st.session_state.pp_norm_peak = float(nm.get("peak_wave", 1580.0))

    ex = cfg.get("extract", {})
    st.session_state.ex_mode = ex.get("mode", "peak_max")
    p = ex.get("params", {})
    for k in ("wave", "w1", "w2", "a1", "a2", "b1", "b2"):
        if k in p:
            st.session_state[f"ex_{k}"] = float(p[k])
    if "metric" in p:
        st.session_state.ex_metric = p["metric"]

    ori = cfg.get("orientation", {})
    st.session_state.or_scan = ori.get("scan", "raster")
    st.session_state.or_start = ori.get("start", "top-left")
    ops = set(ori.get("ops", []))
    for op in OPS:
        st.session_state[f"op_{op}"] = op in ops

    fm = cfg.get("format", {})
    for k, v in fm.items():
        if k in DEFAULTS:
            st.session_state[k] = v

    cam = cfg.get("camera", {})
    for k, v in cam.items():
        if k in DEFAULTS:
            st.session_state[k] = v


def clamp_wave_params(wmin: float, wmax: float):
    """추출 파수 파라미터를 로드된 데이터 파수 범위로 클램프 (위젯 생성 전)."""
    for k in ("ex_wave", "ex_w1", "ex_w2", "ex_a1", "ex_a2", "ex_b1",
              "ex_b2", "pp_norm_peak"):
        v = float(st.session_state.get(k, wmin))
        st.session_state[k] = float(min(max(v, wmin), wmax))


# ===========================================================================
# 4. Export 헬퍼
# ===========================================================================
# kaleido export가 실패해 matplotlib 폴백을 썼는지 표시하는 모듈 플래그.
# (export expander 렌더 시작 시 False로 리셋 → 버튼 렌더 후 True면 st.warning 1회)
_export_used_fallback = False


def mpl_bytes(gridarr, pcfg, fmt: str, dpi: int, surface: bool = False,
              transparent: bool = True) -> bytes:
    """[폴백 전용] matplotlib 이미지 bytes. kaleido 실패 시에만 사용.

    surface=True면 make_matplotlib_surface(3D)로, 아니면 make_matplotlib_heatmap(2D)로
    렌더링해 저장. png/svg/pdf는 transparent=True(투명 배경), jpg는 transparent=False(흰 배경).
    3D 표면은 plotly 카메라(eye≈1.6,-1.6,1.05)를 근사하도록 view_init을 조정한다.
    """
    import matplotlib.pyplot as plt
    if surface:
        fig = plot.make_matplotlib_surface(gridarr, pcfg, dpi=dpi)
        try:  # plotly 카메라 각도 근사(화면과 최대한 비슷하게)
            fig.axes[0].view_init(elev=22, azim=-60)
        except Exception:
            pass
    else:
        fig = plot.make_matplotlib_heatmap(gridarr, pcfg, dpi=dpi)
    buf = io.BytesIO()
    save_fmt = "jpeg" if str(fmt).lower() in ("jpg", "jpeg") else str(fmt).lower()
    fig.savefig(buf, format=save_fmt, dpi=dpi, bbox_inches="tight",
                transparent=transparent)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_image_bytes(gridarr, pcfg, fmt: str, dpi: int,
                       surface: bool = False) -> bytes:
    """화면의 Plotly figure를 그대로 kaleido로 export (화면 == export 보장).

    2D 히트맵/3D 표면 모두 화면과 동일한 figure(make_heatmap/make_surface)를 새로
    만들어 렌더링하므로 카메라 각도·종횡비·colormap·보간이 화면과 일치한다.

    fmt: "png"|"svg"|"pdf"|"jpg". "jpg"는 kaleido에서 "jpeg"로 매핑.
    배경: png/svg/pdf → 투명(paper/plot/scene 모두 rgba(0,0,0,0)).
          jpg/jpeg → 흰색(알파 미지원).
    scale: PNG DPI로부터 max(1, dpi/96). SVG/PDF는 벡터라도 scale 무해.
    kaleido가 런타임에 실패하면 matplotlib 폴백(mpl_bytes)으로 넘어가고
    _export_used_fallback 플래그를 세운다.
    """
    global _export_used_fallback
    f = str(fmt).lower()
    is_jpg = f in ("jpg", "jpeg")
    scale = max(1, dpi / 96)
    try:
        # 화면 figure 객체를 건드리지 않도록 export용으로 새로 생성
        fig = (plot.make_surface(gridarr, pcfg) if surface
               else plot.make_heatmap(gridarr, pcfg))
        if is_jpg:
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
            if surface:
                fig.update_scenes(xaxis_backgroundcolor="white",
                                  yaxis_backgroundcolor="white",
                                  zaxis_backgroundcolor="white")
        else:
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            if surface:
                fig.update_scenes(xaxis_backgroundcolor="rgba(0,0,0,0)",
                                  yaxis_backgroundcolor="rgba(0,0,0,0)",
                                  zaxis_backgroundcolor="rgba(0,0,0,0)")
        kfmt = "jpeg" if is_jpg else f
        return fig.to_image(format=kfmt, scale=scale)
    except Exception:
        _export_used_fallback = True
        return mpl_bytes(gridarr, pcfg, f, dpi, surface=surface,
                         transparent=not is_jpg)


def grid_csv_bytes(gridarr) -> bytes:
    return pd.DataFrame(gridarr).to_csv(index=False, header=False).encode("utf-8-sig")


def grid_xlsx_bytes(gridarr) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(gridarr).to_excel(xw, index=False, header=False)
    return buf.getvalue()


# ===========================================================================
# 5. 사이드바 — 파일 업로드 / 그리드 / Export / 프리셋 / Reset
# ===========================================================================
with st.sidebar:
    logo_html = (f"<img src='{_LOGO_URL}' "
                 f"style='height:40px;margin-bottom:8px;'/>" if _has_logo else "")
    st.markdown(logo_html, unsafe_allow_html=True)
    st.markdown("#### ⚙️ 전역 설정")

    up = st.file_uploader("라만 매핑 파일 (.xlsx / .csv / .txt)",
                          type=["xlsx", "xls", "csv", "txt", "tsv"],
                          key="uploader")

    # ---- 파일 로드 (nx/ny 위젯 생성보다 먼저 — 메타 자동 그리드 세팅을 위해) ----
    raman = None
    load_error = None
    if up is not None:
        try:
            raman = cached_load(up.getvalue(), up.name)
        except Exception as e:  # 친절한 에러
            load_error = str(e)
        # 메타데이터 Map Width/Height → nx/ny 자동 세팅.
        # XY 매핑만 다루므로 Map Depth 는 무시하고, z-stack 등으로 남는 포인트는
        # 파이프라인에서 앞에서부터 nx*ny 개만 사용한다. 파일당 1회만 적용하여
        # 이후 사용자가 입력칸에서 자유롭게 덮어쓸 수 있다.
        if raman is not None and raman.map_width and raman.map_height:
            _file_tag = f"{up.name}:{up.size}"
            if st.session_state.get("_autogrid_file") != _file_tag:
                _nx_m, _ny_m = int(raman.map_width), int(raman.map_height)
                if _nx_m >= 1 and _ny_m >= 1 and _nx_m * _ny_m <= raman.n_points:
                    st.session_state.nx = _nx_m
                    st.session_state.ny = _ny_m
                st.session_state["_autogrid_file"] = _file_tag

    st.markdown("**그리드 크기**")
    st.caption("메타데이터에 Map Width/Height 가 있으면 자동 설정됩니다. "
               "직접 수정도 가능합니다.")
    gc1, gc2 = st.columns(2)
    gc1.number_input("nx (열)", min_value=1, max_value=1000, step=1, key="nx")
    gc2.number_input("ny (행)", min_value=1, max_value=1000, step=1, key="ny")

# ===========================================================================
# 6. 타이틀 헤더
# ===========================================================================
logo_img = (f"<img src='{_LOGO_URL}'/>" if _has_logo else "")
st.markdown(
    f"""
    <div class="title-glass-container">
        {logo_img}
        <div>
            <h2>Raman Mapping Studio</h2>
            <div class="subtitle">라만 매핑 데이터 · 2D Contour Color Fill 및 3D Color Map Surface 지원</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"⚠️ 파일을 불러올 수 없습니다: {load_error}")

# ===========================================================================
# 7. 파이프라인 계산 (최상단 — 세션 config로부터, 사이드바 export/탭이 공유)
# ===========================================================================
pipe = {"ok": False, "grid": None, "pcfg": None, "values": None,
        "index_grid": None, "error": None}

if raman is not None:
    wmin, wmax = float(raman.waves.min()), float(raman.waves.max())
    clamp_wave_params(wmin, wmax)

    nx, ny = int(st.session_state.nx), int(st.session_state.ny)
    # 그리드 정합성
    try:
        loader.validate_grid(raman.n_points, nx, ny)
        grid_valid = True
        grid_msg = None
    except ValueError as e:
        grid_valid = False
        grid_msg = str(e)

    if grid_valid:
        try:
            pp_cfg = build_preprocess_config()
            spectra_pp = cached_preprocess(up.getvalue(), up.name,
                                           json.dumps(pp_cfg, sort_keys=True))
            mode, ex_params = build_extract_params()
            values = extract.extract_values(spectra_pp, raman.waves, mode, **ex_params)
            gcfg = build_grid_config()
            # XY 매핑: 그리드는 메타(Map Width/Height) 기준. z-stack 등으로 남는
            # 포인트는 취득 순서 앞에서부터 nx*ny 개만 사용한다.
            use_n = nx * ny
            gridarr = grid.apply_transform(values[:use_n], nx, ny, gcfg)
            # 클릭→원본 index 역추적용 index grid (같은 변환 적용)
            idx_grid = grid.apply_transform(
                np.arange(use_n, dtype=float), nx, ny, gcfg
            ).round().astype(int)
            pcfg = build_plot_config(gridarr)
            pipe.update(ok=True, grid=gridarr, pcfg=pcfg, values=values,
                        index_grid=idx_grid, spectra_pp=spectra_pp)
            # 사이드바 export(프래그먼트 밖, 이 아래 섹션 8)가 참조할 최신 값 seed.
            # 프래그먼트가 이후 매 rerun 마다 덮어써 카메라/서식 변경을 반영한다.
            st.session_state["_last_grid"] = gridarr
            st.session_state["_last_pcfg"] = pcfg
            st.session_state["_last_surface"] = (
                st.session_state.get("view_mode") == "3D 표면(Surface)")
        except Exception as e:
            pipe["error"] = str(e)
    else:
        pipe["error"] = grid_msg
        pipe["grid_mismatch"] = grid_msg


# ===========================================================================
# 8. 사이드바 (계속) — Export / 프리셋 / Reset  (파이프라인 결과 사용)
# ===========================================================================
def _preset_load_callback():
    name = st.session_state.get("preset_select")
    if not name or name == "—":
        return
    try:
        with open(PRESET_DIR / name, "r", encoding="utf-8") as f:
            apply_settings_dict(json.load(f))
        st.session_state["_preset_msg"] = f"✅ 프리셋 '{name}' 을 불러왔습니다."
    except Exception as e:
        st.session_state["_preset_msg"] = f"⚠️ 프리셋 로드 실패: {e}"


def _preset_delete_callback():
    """선택한 프리셋 파일을 서버 디스크에서 삭제 (콜백 시점에 위젯 key 안전 갱신)."""
    name = st.session_state.get("preset_select")
    if not name or name == "—":
        st.session_state["_preset_msg"] = "⚠️ 삭제할 프리셋을 선택하세요."
        return
    try:
        (PRESET_DIR / name).unlink()
        st.session_state["preset_select"] = "—"  # 삭제된 항목 선택 해제
        st.session_state["_preset_msg"] = f"🗑️ 프리셋 '{name}' 을 삭제했습니다."
    except Exception as e:
        st.session_state["_preset_msg"] = f"⚠️ 삭제 실패: {e}"


def _reset_callback():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def _apply_auto_z(gridarr):
    """'현재 데이터로 auto z 적용' 콜백. 위젯 인스턴스화 이전(콜백 시점)에
    fmt_zmin/fmt_zmax/fmt_zauto(위젯 key)를 안전하게 갱신한다."""
    lo, hi = plot.auto_zrange(gridarr)
    st.session_state.fmt_zmin = float(lo)
    st.session_state.fmt_zmax = float(hi)
    st.session_state.fmt_zauto = False


# 텍스트 서식 툴바의 대상 필드 (표시명 → 위젯 key)
_TB_TARGETS = {
    "제목": "fmt_title",
    "X축 라벨": "fmt_xlabel",
    "Y축 라벨": "fmt_ylabel",
    "Colorbar 라벨": "fmt_cbarlabel",
}


def _tb_insert(markup: str):
    """서식 툴바 콜백 — 선택된 대상 텍스트 필드 끝에 마크업을 삽입한다.

    Streamlit 의 text_input 은 평문 입력칸이라 커서/드래그 선택 위치를 파이썬이 알 수
    없다(오리진식 '선택 후 적용'은 커스텀 컴포넌트가 있어야 가능). 그래서 대상 필드를
    고르고 마크업 껍데기를 끝에 넣어주는 방식으로 구현한다. 위젯 인스턴스화 이전
    (콜백 시점)에 key 를 갱신하므로 입력칸이 새 값으로 다시 그려진다.
    """
    key = _TB_TARGETS[st.session_state.tb_target]
    st.session_state[key] = (st.session_state.get(key) or "") + markup


def _set_camera(azim: float, elev: float, zoom: float):
    """3D 표면 카메라 프리셋 콜백 (위젯 인스턴스화 이전에 key 갱신).

    슬라이더 key(cam_azim/cam_elev/cam_zoom)를 직접 세팅하므로 슬라이더가 새 값으로
    다시 그려지고, export 각도(같은 pcfg)도 자동으로 일치한다."""
    st.session_state.cam_azim = float(azim)
    st.session_state.cam_elev = float(elev)
    st.session_state.cam_zoom = float(zoom)


with st.sidebar:
    st.divider()
    with st.expander("🖼️ 내보내기 설정 (해상도)", expanded=False):
        # 2단계 "이미지 생성 → 다운로드" 흐름 (요청 시에만 kaleido 실행).
        # 예전엔 프래그먼트가 매 rerun 마다 4개 포맷을 EAGER 로 export_image_bytes 로
        # 만들어(포맷당 ~2s) 슬라이더/뷰 조정마다 수초 지연이 있었다. 이제 "이미지 생성"
        # 버튼 클릭(전체 rerun) 시점에만 1개 포맷을 생성한다. 버튼 클릭 rerun 에서
        # 사이드바는 직전 프래그먼트 렌더가 남긴 _last_grid/_last_pcfg/_last_surface
        # (= 현재 화면의 뷰/카메라/서식)를 읽으므로 생성 결과가 화면과 일치한다.
        # (설정 저장/공유는 아래 **프리셋**을 이용하세요.)
        st.number_input("PNG/JPG 해상도(DPI)", min_value=72, max_value=1200,
                        step=50, key="exp_dpi")

        _FMT_MAP = {
            "PNG (투명)": ("png", "image/png", ".png"),
            "JPG (흰 배경)": ("jpg", "image/jpeg", ".jpg"),
            "SVG (투명)": ("svg", "image/svg+xml", ".svg"),
            "PDF (투명)": ("pdf", "application/pdf", ".pdf"),
        }
        fmt_label = st.radio("이미지 형식",
                             ["PNG (투명)", "JPG (흰 배경)",
                              "SVG (투명)", "PDF (투명)"],
                             key="exp_fmt")
        fmt, fmt_mime, fmt_ext = _FMT_MAP[fmt_label]

        if st.button("🖼️ 이미지 생성", use_container_width=True):
            # (사이드바는 모듈 스코프라 _export_used_fallback 재대입에 global 선언
            #  불필요 — 오히려 SyntaxError. export_image_bytes 안에서 global 로 갱신됨.)
            gridarr = st.session_state.get("_last_grid")
            pcfg = st.session_state.get("_last_pcfg")
            is_3d = bool(st.session_state.get("_last_surface", False))
            dpi = int(st.session_state.get("exp_dpi", 300))
            if gridarr is None or pcfg is None:
                st.warning("먼저 파일을 로드하고 매핑을 생성하세요.")
            else:
                _export_used_fallback = False
                img = export_image_bytes(gridarr, pcfg, fmt, dpi, surface=is_3d)
                csv = grid_csv_bytes(gridarr)
                xlsx = grid_xlsx_bytes(gridarr)
                st.session_state["_gen_img"] = img
                st.session_state["_gen_img_name"] = (
                    f"raman_{'surface' if is_3d else 'map'}{fmt_ext}")
                st.session_state["_gen_img_mime"] = fmt_mime
                st.session_state["_gen_view_label"] = (
                    "3D 표면" if is_3d else "2D 히트맵")
                st.session_state["_gen_fmt_label"] = fmt_label
                st.session_state["_gen_dpi"] = dpi
                st.session_state["_gen_csv"] = csv
                st.session_state["_gen_xlsx"] = xlsx
                st.session_state["_gen_fallback"] = bool(_export_used_fallback)

        if st.session_state.get("_gen_img") is not None:
            st.caption(
                f"생성됨: **{st.session_state.get('_gen_view_label')} · "
                f"{st.session_state.get('_gen_fmt_label')}** "
                f"(DPI {st.session_state.get('_gen_dpi')})")
            st.download_button(
                "🖼️ 이미지 다운로드",
                data=st.session_state["_gen_img"],
                file_name=st.session_state.get("_gen_img_name", "raman_map.png"),
                mime=st.session_state.get("_gen_img_mime", "image/png"),
                use_container_width=True)
            st.download_button(
                "📊 CSV 매트릭스",
                data=st.session_state["_gen_csv"],
                file_name="raman_matrix.csv", mime="text/csv",
                use_container_width=True,
                help="Origin 붙여넣기용, 헤더/인덱스 없음")
            st.download_button(
                "📊 XLSX 매트릭스",
                data=st.session_state["_gen_xlsx"],
                file_name="raman_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet",
                use_container_width=True)
            if st.session_state.get("_gen_fallback"):
                st.warning("kaleido export 실패 → matplotlib 폴백으로 저장"
                           "(화면과 미세하게 다를 수 있음).")
            st.caption("뷰/각도/서식을 바꾼 뒤에는 다시 **이미지 생성**을 눌러야 "
                       "최신 상태로 저장됩니다. 3D는 마우스-드래그 회전은 아직 저장에 "
                       "반영 안 됨(슬라이더/프리셋 기준).")

    with st.expander("📌 프리셋 (방향/서식 저장·불러오기)", expanded=False):
        pname = st.text_input("프리셋 이름", key="preset_name",
                              placeholder="예: WITec_100x")
        if st.button("현재 설정 저장", use_container_width=True):
            safe = "".join(c for c in (pname or "").strip()
                           if c.isalnum() or c in ("_", "-", " ")).strip()
            if not safe:
                st.warning("유효한 프리셋 이름을 입력하세요.")
            else:
                try:
                    with open(PRESET_DIR / f"{safe}.json", "w", encoding="utf-8") as f:
                        json.dump(full_settings_dict(), f, ensure_ascii=False, indent=2)
                    st.success(f"저장됨: presets/{safe}.json")
                except Exception as e:
                    st.warning(f"저장 실패: {e}")

        existing = ["—"] + sorted(p.name for p in PRESET_DIR.glob("*.json"))
        st.selectbox("불러올 프리셋", existing, key="preset_select")
        pa, pd_ = st.columns(2)
        pa.button("적용", use_container_width=True,
                  on_click=_preset_load_callback)
        pd_.button("🗑️ 삭제", use_container_width=True,
                   on_click=_preset_delete_callback)

        # 선택 프리셋 개별 다운로드 (클라우드 배포 대비 백업·이식용)
        _sel = st.session_state.get("preset_select")
        if _sel and _sel != "—" and (PRESET_DIR / _sel).exists():
            st.download_button("📤 선택 프리셋 다운로드",
                               data=(PRESET_DIR / _sel).read_bytes(),
                               file_name=_sel, mime="application/json",
                               use_container_width=True)

        # 프리셋 JSON 업로드 → presets/ 에 저장 (재부팅으로 사라져도 복원 가능)
        up = st.file_uploader("📥 프리셋 JSON 업로드", type=["json"],
                              key="preset_upload")
        if up is not None:
            _sig = (up.name, up.size)
            if st.session_state.get("_preset_up_sig") != _sig:
                try:
                    raw = up.getvalue()
                    json.loads(raw.decode("utf-8"))  # JSON 유효성 검증
                    base = os.path.splitext(up.name)[0]
                    safe = "".join(c for c in base
                                   if c.isalnum() or c in ("_", "-", " ")).strip()
                    safe = safe or "uploaded_preset"
                    (PRESET_DIR / f"{safe}.json").write_bytes(raw)
                    st.session_state["_preset_up_sig"] = _sig
                    st.session_state["_preset_msg"] = (
                        f"📥 업로드 저장됨: presets/{safe}.json")
                except Exception as e:
                    st.session_state["_preset_up_sig"] = _sig
                    st.session_state["_preset_msg"] = (
                        f"⚠️ 업로드 실패(유효한 JSON이 아닙니다): {e}")

        if st.session_state.get("_preset_msg"):
            st.info(st.session_state.pop("_preset_msg"))
        st.caption("프리셋은 서버 디스크(presets/)에 저장됩니다. 클라우드 배포 시 "
                   "재부팅되면 UI로 만든 프리셋은 사라질 수 있으니, 중요한 설정은 "
                   "다운로드해 보관하고 필요할 때 업로드하세요.")

    st.divider()
    st.button("🔄 전체 초기화 (Reset)", use_container_width=True,
              on_click=_reset_callback)


# ===========================================================================
# 8.5 시각화 렌더 프래그먼트 (@st.fragment)
# ===========================================================================
@st.fragment
def render_visualization(raman, spectra_pp, nx, ny, wmin, wmax):
    """값싼 시각화 영역 전용 프래그먼트 (Streamlit @st.fragment).

    무거운 상단 파이프라인(파일 로드 · cached_preprocess)은 이 함수 '밖(위)'에서
    이미 계산되어 spectra_pp 로 전달된다. 이 프래그먼트는 값 추출(③)·방향(④)·
    서식(⑤)·최종 뷰·스펙트럼 뷰어(⑥)의 값싼 부분(extract_values + apply_transform
    + figure build + render)만 담당한다. 따라서 colormap·라벨·z-range·방향·카메라
    같은 잦은 조정은 전체 스크립트(≈900줄)를 재실행하지 않고 이 프래그먼트만
    재실행하여 즉시(instant) 반영된다.

    사이드바 Export(프래그먼트 밖, 전체 rerun 시 실행)를 위해 최신 grid/pcfg/
    surface 플래그를 st.session_state['_last_grid'/'_last_pcfg'/'_last_surface']에
    기록한다. Export 다운로드 버튼 클릭은 전체 rerun 을 유발하므로 최신 값을 읽는다.

    이미지 export UI 도 이 프래그먼트 안(최종 뷰 하단)에서 렌더링하므로 kaleido
    폴백 플래그(_export_used_fallback, 모듈 전역)를 리셋/판독하려면 global 선언이 필요하다.
    """
    global _export_used_fallback
    ss = st.session_state

    # 좌: 컨트롤(③④⑤) · 우: 최종 뷰. 왼쪽에서 값을 조정하면 오른쪽 맵이 즉시 반영되어
    # "조정하며 결과 보기"가 한 화면에서 된다. gap="large" 로 두 열 사이 여백을 확보.
    c_ctrl, c_view = st.columns([1, 1.35], gap="large")
    # ⑤ 서식은 컨트롤이 많아 좁은 열에 넣으면 답답하다. 풀폭 컨테이너를 미리 만들어
    # 배치는 두 열 '아래'에 두되, 실행은 아래 순서(⑤ → pcfg → 최종 뷰)를 유지한다.
    fmt_box = st.container()

    with c_ctrl:
        # ---- ③ 매핑 값 추출 ----
        section("③ 매핑 값 추출", "모드별 파수 조건 입력")
        st.selectbox("추출 모드", MODES, key="ex_mode",
                     format_func=lambda m: MODE_LABELS[m])
        ex_mode_sel = ss.ex_mode
        if ex_mode_sel == "single":
            st.number_input("파수 (cm⁻¹)", min_value=wmin, max_value=wmax,
                            step=1.0, key="ex_wave")
        elif ex_mode_sel in ("peak_max", "peak_area", "peak_position", "fwhm"):
            cc1, cc2 = st.columns(2)
            cc1.number_input("구간 시작 w1 (cm⁻¹)", min_value=wmin, max_value=wmax,
                             step=1.0, key="ex_w1")
            cc2.number_input("구간 끝 w2 (cm⁻¹)", min_value=wmin, max_value=wmax,
                             step=1.0, key="ex_w2")
        else:  # ratio
            st.markdown("**구간 A (분자)**")
            ca1, ca2 = st.columns(2)
            ca1.number_input("A 시작 a1", min_value=wmin, max_value=wmax,
                             step=1.0, key="ex_a1")
            ca2.number_input("A 끝 a2", min_value=wmin, max_value=wmax,
                             step=1.0, key="ex_a2")
            st.markdown("**구간 B (분모)**")
            cb1, cb2 = st.columns(2)
            cb1.number_input("B 시작 b1", min_value=wmin, max_value=wmax,
                             step=1.0, key="ex_b1")
            cb2.number_input("B 끝 b2", min_value=wmin, max_value=wmax,
                             step=1.0, key="ex_b2")
            st.radio("metric", ["max", "area"], key="ex_metric", horizontal=True)

        st.divider()

        # ---- ④ 방향 정렬 (별도 미리보기 히트맵 제거 → 오른쪽 최종 뷰가 미리보기 겸함) ----
        section("④ 방향 정렬",
                "optic 이미지와 방향을 맞추세요 · 결과는 오른쪽 최종 뷰에 실시간 반영")
        st.selectbox("스캔 방식", SCANS, key="or_scan")
        st.selectbox("시작 코너", STARTS, key="or_start")
        st.markdown("**추가 변환 (순서대로 적용)**")
        for op in OPS:
            st.checkbox(OP_LABELS[op], key=f"op_{op}")

        # ---- 값 추출 + 그리드 변환 (값싼 연산) ----
        grid_err = None
        gridarr = None
        idx_grid = None
        try:
            mode, ex_params = build_extract_params()
            values = extract.extract_values(spectra_pp, raman.waves, mode, **ex_params)
            gcfg = build_grid_config()
            # 파이프라인과 동일: 앞에서부터 nx*ny 포인트만 사용 (초과분 무시)
            use_n = nx * ny
            gridarr = grid.apply_transform(values[:use_n], nx, ny, gcfg)
            idx_grid = grid.apply_transform(
                np.arange(use_n, dtype=float), nx, ny, gcfg
            ).round().astype(int)
        except Exception as e:
            grid_err = str(e)

    with fmt_box:
        st.divider()
        # ---- ⑤ 히트맵 서식 ----
        section("⑤ 히트맵 서식", "colormap · z-range · 축 · colorbar · 폰트")
        with st.expander("서식 옵션 펼치기", expanded=True):
            # ---- 텍스트 서식 툴바 (오리진 스타일) ----
            # 대상 필드를 고르고 B/I/x²/x₂ 를 누르면 그 필드에 마크업이 삽입된다.
            _tb = st.columns([1.5, 0.45, 0.45, 0.45, 0.45, 2.7], gap="small")
            _tb[0].selectbox("서식 대상", list(_TB_TARGETS), key="tb_target",
                             label_visibility="collapsed")
            _tb[1].button("**B**", key="tb_bold", use_container_width=True,
                          on_click=_tb_insert, args=("*{}",),
                          help="볼드 — 대상 텍스트에 *{ } 삽입")
            _tb[2].button("*I*", key="tb_italic", use_container_width=True,
                          on_click=_tb_insert, args=("/{}",),
                          help="이탤릭 — 대상 텍스트에 /{ } 삽입")
            _tb[3].button("x²", key="tb_sup", use_container_width=True,
                          on_click=_tb_insert, args=("^{}",),
                          help="위첨자 — 대상 텍스트에 ^{ } 삽입")
            _tb[4].button("x₂", key="tb_sub", use_container_width=True,
                          on_click=_tb_insert, args=("_{}",),
                          help="아래첨자 — 대상 텍스트에 _{ } 삽입")
            _tb[5].caption("대상을 고르고 버튼을 누르면 그 텍스트 끝에 마크업이 "
                           "추가됩니다. 중괄호 `{ }` 안에 서식을 적용할 글자를 넣으세요. "
                           "직접 타이핑해도 됩니다.")
            st.divider()

            f1, f2, f3 = st.columns(3, gap="large")
            with f1:
                st.selectbox("Colormap", COLORMAPS, key="fmt_cmap")
                st.radio("2D 채우기", ["픽셀(격자)", "등고선(contour)"],
                         key="fmt_fill", horizontal=True,
                         help="픽셀=격자 히트맵, 등고선=Origin 스타일 컬러 컨투어 "
                              "(2D 뷰·내보내기에만 적용, 3D 표면은 무관)")
                if ss.fmt_fill == "등고선(contour)":
                    st.checkbox("등고선 라인 표시", key="fmt_contour_lines",
                                help="컨투어 채움 위에 겹쳐 그리는 등고선 라인만 켜고 "
                                     "끕니다 (채움·colorbar 는 그대로).")
                st.checkbox("z-range 자동 (2–98%)", key="fmt_zauto")
                if gridarr is not None:
                    st.button("현재 데이터로 auto z 적용",
                              on_click=_apply_auto_z, args=(gridarr,))
                if not ss.fmt_zauto:
                    st.number_input("z-min", key="fmt_zmin", format="%.4g")
                    st.number_input("z-max", key="fmt_zmax", format="%.4g")
            with f2:
                st.text_input("X 축 라벨", key="fmt_xlabel")
                st.text_input("Y 축 라벨", key="fmt_ylabel")
                st.text_input("Z 축 라벨 (3D)", key="fmt_zlabel",
                              placeholder="비우면 Colorbar 라벨을 따름",
                              help="3D 표면의 Z축 라벨. 비워두면 Colorbar 라벨과 "
                                   "같은 값을 사용합니다.")
                tt1, tt2 = st.columns([1.4, 1], vertical_alignment="bottom")
                tt1.text_input("제목", key="fmt_title")
                tt2.selectbox("위치", ["left", "center", "right"],
                              key="fmt_title_pos",
                              format_func=lambda p: {"left": "왼쪽", "center": "가운데",
                                                     "right": "오른쪽"}[p],
                              help="제목 위치 (프리셋에 저장됩니다).")
                st.number_input("Step X (μm)", min_value=0.0001, step=0.1,
                                key="fmt_stepx", format="%.4g")
                st.number_input("Step Y (μm)", min_value=0.0001, step=0.1,
                                key="fmt_stepy", format="%.4g")
            with f3:
                st.text_input("Colorbar 라벨", key="fmt_cbarlabel")
                st.number_input("Colorbar tick 개수", min_value=2, max_value=40,
                                step=1, key="fmt_cbarticks")
                st.caption("값을 높이면 colorbar·컨투어가 더 연속적으로 보입니다.")
                st.selectbox("폰트", FONTS, key="fmt_font")
                st.caption("아래는 요소 **전체**에 적용됩니다. 일부 글자만 바꾸려면 "
                           "위쪽 툴바(마크업)를 쓰세요.")
                for _lbl, _sz, _bd, _it, _col in [
                    ("라벨", "fmt_fs_label", "fmt_bold_label", "fmt_italic_label", "fmt_color_label"),
                    ("눈금", "fmt_fs_tick", "fmt_bold_tick", "fmt_italic_tick", "fmt_color_tick"),
                    ("제목", "fmt_fs_title", "fmt_bold_title", "fmt_italic_title", "fmt_color_title"),
                ]:
                    _r = st.columns([1.1, 1.2, 0.7, 0.7, 0.8])
                    _r[0].markdown(f"**{_lbl}**")
                    _r[1].number_input("크기", min_value=6, max_value=50, key=_sz,
                                       label_visibility="collapsed")
                    _r[2].checkbox("B", key=_bd)
                    _r[3].checkbox("I", key=_it)
                    _r[4].color_picker("색", key=_col, label_visibility="collapsed")
                st.checkbox("눈금 표시", key="fmt_showticks")
                st.number_input("눈금 간격 (μm, 0=자동)", min_value=0.0, step=1.0,
                                key="fmt_tickspacing")
                st.selectbox("Interpolation", ["none", "bilinear"], key="fmt_interp")

        if grid_err is not None:
            st.error(f"⚠️ {grid_err}")

    if grid_err is not None:
        return

    pcfg = build_plot_config(gridarr)

    # ---- 보기 방식 + 최종 뷰 (오른쪽 열 = 왼쪽 컨트롤 조정 결과를 즉시 확인) ----
    with c_view:
        st.radio("보기 방식", ["2D 히트맵", "3D 표면(Surface)"],
                 key="view_mode", horizontal=True)
        event = None
        if ss.view_mode == "3D 표면(Surface)":
            st.caption("🖱️ 마우스 드래그는 자유 관찰용이며 내보내기 각도에는 반영되지 "
                       "않습니다. 저장할 각도는 아래 슬라이더·프리셋으로 맞추세요.")
            surf_fig = plot.make_surface(gridarr, pcfg)
            st.plotly_chart(surf_fig, use_container_width=True, key="surf3d_plot")

            # 카메라 컨트롤 — 화면 == export 각도 동기화 (make_surface 가 eye 계산).
            # 맵 바로 아래에 두어 각도를 돌리며 결과를 같은 시야에서 확인한다.
            cca = st.columns(3)
            cca[0].slider("방위각 azimuth (°)", -180.0, 180.0, step=1.0, key="cam_azim")
            cca[1].slider("고도 elevation (°)", 0.0, 90.0, step=1.0, key="cam_elev")
            cca[2].slider("줌 거리 (클수록 축소)", 1.2, 4.0, step=0.1, key="cam_zoom")
            pcb = st.columns(3)
            pcb[0].button("정면 뷰", use_container_width=True,
                          on_click=_set_camera, args=(-90.0, 8.0, 2.4))
            pcb[1].button("등각 뷰", use_container_width=True,
                          on_click=_set_camera, args=(-45.0, 25.0, 2.2))
            pcb[2].button("위에서 뷰", use_container_width=True,
                          on_click=_set_camera, args=(-90.0, 89.0, 2.2))
        else:
            final_fig = plot.make_heatmap(gridarr, pcfg)
            if ss.fmt_fill == "등고선(contour)":
                # go.Contour 는 클릭 포인트 이벤트가 불안정하므로 on_select 를 걸지 않고,
                # 스펙트럼 QC 는 아래 X/Y 픽셀 입력(폴백)으로 처리한다.
                st.plotly_chart(final_fig, use_container_width=True,
                                key="final_contour")
                st.caption("최종 컨투어 맵 — 스펙트럼 QC 는 아래 X/Y 픽셀 입력을 "
                           "사용하세요.")
            else:
                event = st.plotly_chart(final_fig, use_container_width=True,
                                        key="final_heatmap", on_select="rerun",
                                        selection_mode=("points", "box"))
                st.caption("픽셀을 클릭하면 아래 ⑥ 스펙트럼 뷰어에 원본 스펙트럼이 "
                           "표시됩니다.")

    # (레거시) 예전 사이드바 Export 가 읽던 최신 값. 이제 이미지/매트릭스 export 는
    # 바로 아래 프래그먼트 내부에서 gridarr/pcfg/ss.view_mode 를 직접 사용하므로
    # 이 값들은 더 이상 export 에 쓰이지 않는다. 다른 코드가 참조하지 않아 무해하게 유지.
    # 사이드바 Export(프래그먼트 밖, 전체 rerun 시 실행)의 "이미지 생성" 버튼이
    # 읽는 최신 grid/pcfg/surface 값. 버튼 클릭은 전체 rerun 을 유발하므로,
    # 사이드바는 직전 프래그먼트 렌더가 기록한 아래 값(= 현재 화면 뷰/카메라/서식)을
    # 읽어 생성 결과가 화면과 일치한다.
    ss["_last_grid"] = gridarr
    ss["_last_pcfg"] = pcfg
    ss["_last_surface"] = (ss.view_mode == "3D 표면(Surface)")

    st.divider()
    # ---- ⑥ 스펙트럼 뷰어 ----
    section("⑥ 스펙트럼 뷰어", "히트맵 클릭 지점의 원본 스펙트럼 (QC)")
    ny_g, nx_g = idx_grid.shape

    # 클릭 이벤트 → row/col
    clicked = None
    try:
        pts = event["selection"]["points"]
    except Exception:
        pts = []
    if pts:
        px, py = pts[0].get("x"), pts[0].get("y")
        if px is not None and py is not None:
            col = int(round((float(px) - pcfg.x0) / pcfg.step_x))
            row = int(round((float(py) - pcfg.y0) / pcfg.step_y))
            if 0 <= row < ny_g and 0 <= col < nx_g:
                ss.sv_row = row
                ss.sv_col = col
                clicked = (row, col)

    # 폴백: 수동 픽셀 선택
    sc1, sc2, _ = st.columns([1, 1, 2])
    ss.sv_col = min(ss.sv_col, nx_g - 1)
    ss.sv_row = min(ss.sv_row, ny_g - 1)
    sc1.number_input("픽셀 X (열)", min_value=0, max_value=nx_g - 1,
                     step=1, key="sv_col")
    sc2.number_input("픽셀 Y (행)", min_value=0, max_value=ny_g - 1,
                     step=1, key="sv_row")

    row, col = int(ss.sv_row), int(ss.sv_col)
    orig_idx = int(idx_grid[row, col])
    label = f"(x={col}, y={row}) · 원본 index {orig_idx}"
    if clicked:
        st.caption(f"클릭 선택: {label}")
    spec_fig = plot.make_spectrum_figure(
        raman.waves, raman.spectra[orig_idx],
        title="원본 스펙트럼 (전처리 전)", point_label=label)
    st.plotly_chart(spec_fig, use_container_width=True, key="spectrum_fig")


# ===========================================================================
# 9. 메인 탭
# ===========================================================================
tab_map, tab_batch = st.tabs(["🗺️ 매핑 생성", "📦 배치 처리"])


# ---------------------------------------------------------------------------
# 탭 1 — 매핑 생성 (핵심 워크플로우)
# ---------------------------------------------------------------------------
with tab_map:
    if raman is None:
        st.info("👈 왼쪽 사이드바에서 라만 매핑 파일을 업로드하세요. "
                "(.xlsx / .csv / .txt 지원)")
    else:
        wmin, wmax = float(raman.waves.min()), float(raman.waves.max())

        # 상단 한 줄: ① 데이터 정보(더 중요 → 넓게) | ② 전처리(접힘 기본 → 좁게).
        # gap="large" 로 두 블록 사이 여백을 확보한다.
        c_info, c_pre = st.columns([1.6, 1], gap="large")

        # ---- 파일 정보 ----
        with c_info:
            section("① 데이터 정보", "감지된 포맷 · 메타데이터 · 포인트 수")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("포맷", FORMAT_LABELS.get(raman.source_format,
                                              raman.source_format))
            m2.metric("포인트 수", f"{raman.n_points}")
            m3.metric("파수 채널", f"{raman.n_waves}")
            m4.metric("파수 범위", f"{wmin:.0f}~{wmax:.0f}")
            if raman.metadata:
                with st.expander("메타데이터 전체 보기"):
                    st.json(raman.metadata)
            # 그리드 힌트 안내
            hint = []
            if raman.map_width:
                hint.append(f"Map Width={raman.map_width}")
            if raman.step_x:
                hint.append(f"Step X/Y={raman.step_x}/{raman.step_y}")
            if hint:
                st.caption("메타 힌트: " + ", ".join(hint) +
                           " · Map Width/Height 는 그리드(nx·ny)에 자동 반영됩니다")
            _use_n = int(st.session_state.nx) * int(st.session_state.ny)
            if _use_n < raman.n_points:
                st.caption(f"ℹ️ XY 매핑 기준 {raman.n_points}개 포인트 중 앞에서부터 "
                           f"{_use_n}개를 사용합니다 (깊이 스택 등 초과분 "
                           f"{raman.n_points - _use_n}개 제외).")

            if pipe.get("grid_mismatch"):
                st.error(f"⚠️ {pipe['grid_mismatch']} — 사이드바에서 nx/ny를 조정하세요.")

        # ---- 전처리 ----
        with c_pre:
            section("② 전처리 (선택)", "cosmic · baseline · smoothing · normalize")
            with st.expander("전처리 옵션 펼치기", expanded=False):
                # --- Baseline 보정: 라벨 왼쪽 · 선택 버튼 오른쪽(폭 축소) ---
                #     긴 ALS 경고는 캡션 대신 ⚠️ 호버 툴팁으로 접어 세로 공간을 아낀다.
                #     Normalization 과 같은 [2.6, 1] 비율 → 두 선택창 폭이 동일.
                bl1, bl2 = st.columns([2.6, 1], vertical_alignment="center")
                bl1.markdown(
                    hint_label(
                        "Baseline 보정",
                        info=("스펙트럼 아래 깔린 완만한 배경(형광 등)을 추정해 빼는 단계.\n"
                              "• als — Asymmetric Least Squares: 피크는 남기고 배경만 "
                              "반복적으로 추정해 제거. 배경이 복잡해도 잘 맞지만 느림.\n"
                              "• poly — 다항식 피팅으로 배경을 근사해 제거. 훨씬 빠르지만 "
                              "배경이 복잡하면 부정확할 수 있음."),
                        warn=("ALS는 400 스펙트럼에 약 4초 걸립니다(가장 무거운 단계). "
                              "빠른 작업은 off 또는 poly 를 권장합니다. "
                              "동일 설정은 캐시되어 재계산 없이 즉시 반영됩니다."),
                    ),
                    unsafe_allow_html=True,
                )
                bl2.selectbox("Baseline 보정", ["off", "als", "poly"],
                              key="pp_baseline", label_visibility="collapsed")
                if st.session_state.pp_baseline == "als":
                    ba1, ba2, ba3 = st.columns(3)
                    ba1.number_input("ALS λ (lam)", min_value=1.0,
                                     step=1000.0, key="pp_als_lam", format="%.0f")
                    ba2.number_input("ALS p", min_value=0.0001, max_value=0.5,
                                     step=0.001, key="pp_als_p", format="%.4f")
                    ba3.number_input("ALS niter", min_value=1, max_value=50,
                                     step=1, key="pp_als_niter")
                elif st.session_state.pp_baseline == "poly":
                    st.number_input("다항식 차수", min_value=0, max_value=15,
                                    step=1, key="pp_poly_order")

                # --- Normalization: Baseline 바로 아래(구분선 없이, 여백 최소화) ---
                #     Baseline 과 동일한 [2.6, 1] 비율로 선택창 폭을 맞춘다.
                nm1, nm2 = st.columns([2.6, 1], vertical_alignment="center")
                nm1.markdown(
                    hint_label(
                        "Normalization",
                        info=("스펙트럼 간 세기 차이를 없애 비교 가능하게 만드는 단계.\n"
                              "• max — 각 스펙트럼을 자신의 최댓값으로 나눠 0~1 로 정규화.\n"
                              "• peak — 지정한 기준 파수의 세기로 나눠, 그 피크 대비 "
                              "상대 세기로 비교."),
                    ),
                    unsafe_allow_html=True,
                )
                nm2.selectbox("Normalization", ["off", "max", "peak"],
                              key="pp_norm", label_visibility="collapsed")
                if st.session_state.pp_norm == "peak":
                    st.number_input("기준 파수 (cm⁻¹)", min_value=wmin,
                                    max_value=wmax, step=1.0, key="pp_norm_peak")

                st.divider()

                # --- Cosmic / Smoothing: 2열 ---
                c1, c2 = st.columns(2)
                with c1:
                    # 라벨이 보이는 위젯은 Streamlit 기본 help(❓ 아이콘)를 그대로 쓴다.
                    st.checkbox(
                        "Cosmic ray 제거", key="pp_cosmic",
                        help=("우주선(cosmic ray)이 검출기에 직접 튀어 생기는 1~2채널 폭의 "
                              "비정상적으로 뾰족한 스파이크를 제거합니다. 이웃 대비 "
                              "이상치(MAD 배수 기준)를 찾아 주변 값으로 대체합니다. "
                              "threshold 가 낮을수록 더 공격적으로 제거합니다."))
                    if st.session_state.pp_cosmic:
                        st.number_input("threshold (MAD 배수)", min_value=1.0,
                                        max_value=20.0, step=0.5, key="pp_cosmic_thr")
                        st.number_input("window (홀수)", min_value=3, max_value=51,
                                        step=2, key="pp_cosmic_win")
                with c2:
                    st.checkbox(
                        "Savitzky-Golay 평활", key="pp_smooth",
                        help=("이동 창 안에서 다항식을 피팅해 노이즈를 줄이는 필터. "
                              "단순 이동평균과 달리 피크의 높이·폭을 비교적 잘 보존합니다.\n"
                              "• window — 창 크기(홀수). 클수록 매끄럽지만 피크가 뭉개짐.\n"
                              "• polyorder — 피팅 다항식 차수. 낮을수록 더 강하게 평활."))
                    if st.session_state.pp_smooth:
                        st.number_input("window (홀수)", min_value=3, max_value=101,
                                        step=2, key="pp_smooth_win")
                        st.number_input("polyorder", min_value=0, max_value=10,
                                        step=1, key="pp_smooth_poly")

        st.divider()

        # ---- ③~⑥ 시각화·서식·렌더 (프래그먼트) ----
        # colormap·라벨·z-range·방향·카메라 등 값싼 조정은 이 프래그먼트만 재실행하여
        # 즉시 반영된다(전체 스크립트/전처리 재실행 없음). nx/ny·파일·전처리 변경만
        # 전체 rerun 을 유발하며 그것들은 프래그먼트 밖(사이드바/②)에 있다.
        if pipe["ok"]:
            render_visualization(
                raman, pipe["spectra_pp"],
                int(st.session_state.nx), int(st.session_state.ny), wmin, wmax)
        elif pipe["error"]:
            st.error(f"⚠️ {pipe['error']}")
        else:
            st.info("사이드바에서 그리드(nx×ny)를 데이터 포인트 수에 맞추면 "
                    "시각화가 표시됩니다.")


# ---------------------------------------------------------------------------
# 탭 2 — 배치 처리
# ---------------------------------------------------------------------------
with tab_batch:
    section("📦 배치 처리", "여러 파일에 현재 설정을 일괄 적용 → ZIP 다운로드")
    st.caption("현재 탭①의 전처리·추출·방향·서식 설정을 그대로 각 파일에 적용합니다. "
               "그리드 크기(nx×ny)와 포인트 수가 맞는 파일만 처리됩니다.")

    batch_files = st.file_uploader(
        "여러 라만 매핑 파일 업로드", type=["xlsx", "xls", "csv", "txt", "tsv"],
        accept_multiple_files=True, key="batch_uploader")

    if st.button("배치 실행", type="primary", disabled=not batch_files):
        nx, ny = int(st.session_state.nx), int(st.session_state.ny)
        pp_cfg = build_preprocess_config()
        mode, ex_params = build_extract_params()
        gcfg = build_grid_config()
        dpi = int(st.session_state.exp_dpi)

        zip_buf = io.BytesIO()
        results, errors = [], []
        prog = st.progress(0.0)
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, bf in enumerate(batch_files):
                try:
                    rd = cached_load(bf.getvalue(), bf.name)
                    loader.validate_grid(rd.n_points, nx, ny)
                    sp = preprocess.apply_preprocessing(rd.spectra, rd.waves, pp_cfg)
                    vals = extract.extract_values(sp, rd.waves, mode, **ex_params)
                    # 메인 파이프라인과 동일: 앞에서부터 nx*ny 포인트만 사용
                    garr = grid.apply_transform(vals[:nx * ny], nx, ny, gcfg)
                    pcfg = build_plot_config(garr)
                    stem = os.path.splitext(bf.name)[0]
                    zf.writestr(f"{stem}_2d.png",
                                export_image_bytes(garr, pcfg, "png", dpi,
                                                   surface=False))
                    zf.writestr(f"{stem}_3d.png",
                                export_image_bytes(garr, pcfg, "png", dpi,
                                                   surface=True))
                    zf.writestr(f"{stem}_matrix.csv", grid_csv_bytes(garr))
                    results.append(bf.name)
                except Exception as e:
                    errors.append(f"{bf.name}: {e}")
                prog.progress((i + 1) / len(batch_files))
            # 공통 설정도 포함
            zf.writestr("settings.json",
                        json.dumps(full_settings_dict(), ensure_ascii=False, indent=2))

        st.session_state["_batch_zip"] = zip_buf.getvalue()
        st.session_state["_batch_results"] = results
        st.session_state["_batch_errors"] = errors

    if st.session_state.get("_batch_zip"):
        res = st.session_state.get("_batch_results", [])
        errs = st.session_state.get("_batch_errors", [])
        st.success(f"✅ 처리 완료: {len(res)}개 파일 (2D PNG + 3D PNG + CSV)")
        if errs:
            with st.expander(f"⚠️ 실패 {len(errs)}건"):
                for e in errs:
                    st.write("• " + e)
        st.download_button("📥 결과 ZIP 다운로드", st.session_state["_batch_zip"],
                           "raman_batch.zip", "application/zip",
                           type="primary", use_container_width=True)
