"""
ui/styles.py — 다크 핀테크 테마 CSS 시스템

CSS 변수는 Python 상수로도 제공되어 인라인 스타일 생성에 활용됩니다.
실제 전역 CSS는 static/midnight_aurora.css에서 관리하고 inject_css()로 주입합니다.
"""
from __future__ import annotations
import os
import streamlit as st

# ─── 디자인 토큰 ──────────────────────────────────────────────────────────────
COLORS: dict[str, str] = {
    # 배경 레이어
    "bg":        "#0E1117",
    "surface":   "#161B22",
    "elevated":  "#1C2128",
    # 경계선
    "border":    "#30363D",
    "border_md": "#21262D",
    # 텍스트
    "text":      "#E6EDF3",
    "text_2":    "#8B949E",
    "text_3":    "#484F58",
    # 매매 신호
    "gain":      "#26a69a",
    "gain_dim":  "rgba(38,166,154,0.15)",
    "gain_rgb":  "38,166,154",
    "loss":      "#ef5350",
    "loss_dim":  "rgba(239,83,80,0.15)",
    "loss_rgb":  "239,83,80",
    # 강조색
    "accent":    "#8B5CF6",
    "accent_dim":"rgba(139,92,246,0.15)",
    "accent_rgb":"139,92,246",
    "blue":      "#3B82F6",
}

RADIUS = "12px"
RADIUS_LG = "16px"
SHADOW = "0 4px 20px rgba(0,0,0,0.35)"


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
    _bg      = bg or COLORS["surface"]
    _border  = f"1px solid {border_color}" if border_color else f"1px solid {COLORS['border']}"
    return (
        f"background:{_bg};border:{_border};"
        f"border-radius:{RADIUS};padding:18px 20px;"
        f"box-shadow:{SHADOW};{extra}"
    )
