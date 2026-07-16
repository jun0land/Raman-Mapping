"""Plotly와 Matplotlib figure의 색상 일치 회귀 테스트.

버그: 화면(Plotly)과 export(Matplotlib)가 같은 colormap 이름에 대해 서로 다른
색을 썼다(gray 반전, rainbow 완전 상이). 수정: Plotly colorscale을 matplotlib
colormap 샘플링으로 생성 -> 두 figure가 픽셀 단위로 동일한 색을 쓴다.
"""

from __future__ import annotations

import re

import matplotlib

from core.plot import MPL_CMAP, mpl_to_plotly_colorscale

# 지원 colormap 이름
SUPPORTED = ["jet", "viridis", "plasma", "inferno", "rainbow", "gray", "rdbu"]
SAMPLE_T = [0.0, 0.25, 0.5, 0.75, 1.0]
TOL = 2  # rgb 정수 반올림 허용 오차


def _mpl_rgb255(name: str, t: float) -> tuple[int, int, int]:
    """matplotlib(참조)의 t 지점 RGB를 0-255 정수로."""
    cmap = matplotlib.colormaps[MPL_CMAP[name]]
    r, g, b, _a = cmap(t)
    return (round(r * 255), round(g * 255), round(b * 255))


def _parse_rgb(s: str) -> tuple[int, int, int]:
    m = re.match(r"rgb\((\d+),(\d+),(\d+)\)", s.replace(" ", ""))
    assert m is not None, f"예상치 못한 색 형식: {s}"
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def _nearest_stop(scale: list[list], t: float) -> tuple[float, tuple[int, int, int]]:
    """생성된 Plotly colorscale에서 t에 가장 가까운 stop의 (fraction, RGB)."""
    best = min(scale, key=lambda pair: abs(pair[0] - t))
    return best[0], _parse_rgb(best[1])


def _nearest_stop_color(scale: list[list], t: float) -> tuple[int, int, int]:
    return _nearest_stop(scale, t)[1]


def test_scale_structure():
    """colorscale은 [[frac, "rgb(...)"], ...] 형태이며 0->1 단조 증가."""
    scale = mpl_to_plotly_colorscale("jet", n=256)
    assert len(scale) == 256
    assert scale[0][0] == 0.0
    assert scale[-1][0] == 1.0
    fracs = [p[0] for p in scale]
    assert fracs == sorted(fracs)
    for frac, color in scale:
        assert 0.0 <= frac <= 1.0
        _parse_rgb(color)  # 형식 검증


def test_plotly_matches_matplotlib_all_cmaps():
    """모든 지원 colormap에서 Plotly 생성 색 == matplotlib 참조 색 (±TOL)."""
    for name in SUPPORTED:
        scale = mpl_to_plotly_colorscale(name)
        for t in SAMPLE_T:
            # 매칭된 stop의 fraction에서 matplotlib을 평가 -> 순수 반올림 오차만 남음
            # (nearest-stop 양자화가 아니라 "스케일이 곧 matplotlib colormap"임을 검증)
            frac, got = _nearest_stop(scale, t)
            ref = _mpl_rgb255(name, frac)
            for c_ref, c_got, ch in zip(ref, got, "rgb"):
                assert abs(c_ref - c_got) <= TOL, (
                    f"{name} t={t} 채널 {ch}: matplotlib={ref} vs plotly={got}"
                )


def test_gray_not_inverted():
    """보고된 정확한 버그 가드: gray는 t=0 검정, t=1 흰색이어야 한다.

    (Plotly 'Greys' 내장은 반전되어 t=0이 흰색이었음.)
    """
    scale = mpl_to_plotly_colorscale("gray")
    r0, g0, b0 = _nearest_stop_color(scale, 0.0)
    r1, g1, b1 = _nearest_stop_color(scale, 1.0)
    # t=0 near-black
    assert r0 <= 5 and g0 <= 5 and b0 <= 5, f"gray t=0 이 검정이 아님: {(r0, g0, b0)}"
    # t=1 near-white
    assert r1 >= 250 and g1 >= 250 and b1 >= 250, f"gray t=1 이 흰색이 아님: {(r1, g1, b1)}"


def test_grey_alias_matches_gray():
    """'grey' 별칭도 'gray'와 동일 스케일."""
    assert mpl_to_plotly_colorscale("grey") == mpl_to_plotly_colorscale("gray")


def test_unknown_falls_back_to_jet():
    """미지원 이름은 jet으로 폴백."""
    assert mpl_to_plotly_colorscale("nope-not-real") == mpl_to_plotly_colorscale("jet")
