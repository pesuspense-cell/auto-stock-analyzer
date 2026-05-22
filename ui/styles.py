"""
ui/styles.py — Apple Finance 디자인 시스템 토큰

CSS 변수는 Python 상수로도 제공되어 인라인 스타일 생성에 활용됩니다.
실제 전역 CSS는 static/midnight_aurora.css에서 관리하고 inject_css()로 주입합니다.
"""
from __future__ import annotations
import os
import streamlit as st

# ─── 디자인 토큰 ──────────────────────────────────────────────────────────────
COLORS: dict[str, str] = {
    # 배경 레이어
    "bg":        "#f5f5f7",   # canvas-parchment
    "surface":   "#ffffff",   # pure white
    "elevated":  "#fafafc",   # pearl

    # 경계선
    "border":    "#e0e0e0",   # hairline
    "border_md": "#d2d2d7",   # hairline medium

    # 텍스트
    "text":      "#1d1d1f",   # near-black ink
    "text_2":    "#7a7a7a",   # ink muted 48
    "text_3":    "#b0b0b0",   # ink muted lighter

    # 매매 신호
    "gain":      "#34c759",
    "gain_dim":  "rgba(52,199,89,0.10)",
    "gain_rgb":  "52,199,89",
    "loss":      "#ff3b30",
    "loss_dim":  "rgba(255,59,48,0.10)",
    "loss_rgb":  "255,59,48",

    # 강조색 — Action Blue
    "accent":     "#0066cc",
    "accent_dim": "rgba(0,102,204,0.10)",
    "accent_rgb": "0,102,204",
    "blue":       "#2997ff",   # sky link blue (on dark)

    # 다크 서피스 (터미널 타일)
    "dark_tile":  "#272729",   # terminal dark 1
    "dark_tile2": "#2a2a2c",   # terminal dark 2
    "nav_black":  "#000000",   # global nav
}

RADIUS    = "12px"
RADIUS_LG = "16px"
SHADOW    = "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06)"
SHADOW_MD = "rgba(0,0,0,0.22) 3px 5px 30px 0"


def inject_css() -> None:
    """static/midnight_aurora.css를 Streamlit에 주입한다."""
    _css_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "static", "midnight_aurora.css"
    )
    try:
        with open(_css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


# ─── 인라인 스타일 헬퍼 ──────────────────────────────────────────────────────

def gain_or_loss_color(value: float) -> str:
    """양수면 gain 색, 음수/0이면 loss 색을 반환."""
    return COLORS["gain"] if value >= 0 else COLORS["loss"]


def gain_or_loss_rgb(value: float) -> str:
    return COLORS["gain_rgb"] if value >= 0 else COLORS["loss_rgb"]


def signed_color_style(value: float) -> str:
    """color: #... 인라인 스타일 문자열 반환."""
    return f"color:{gain_or_loss_color(value)}"


def card_style(
    border_color: str | None = None,
    bg: str | None = None,
    extra: str = "",
) -> str:
    """공통 카드 inline style 문자열 반환."""
    _bg     = bg or COLORS["surface"]
    _border = f"1px solid {border_color}" if border_color else f"1px solid {COLORS['border']}"
    return (
        f"background:{_bg};border:{_border};"
        f"border-radius:{RADIUS};padding:18px 20px;"
        f"box-shadow:{SHADOW};{extra}"
    )
