"""
ui/components.py — 재사용 가능한 HTML 컴포넌트 빌더

모든 함수는 순수 HTML 문자열을 반환합니다.
st.markdown(..., unsafe_allow_html=True) 로 렌더링하세요.
"""
from __future__ import annotations
from ui.styles import COLORS, RADIUS, SHADOW

# ─── SVG 아이콘 ───────────────────────────────────────────────────────────────
SVG_WALLET = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
    ' stroke-linejoin="round"><path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/>'
    '<path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/>'
    '<path d="M18 12a2 2 0 0 0 0 4h4v-4Z"/></svg>'
)
SVG_BAR_CHART = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
    ' stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/>'
    '<line x1="12" y1="20" x2="12" y2="4"/>'
    '<line x1="6" y1="20" x2="6" y2="14"/></svg>'
)
SVG_TREND = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"'
    ' stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>'
    '<polyline points="16 7 22 7 22 13"/></svg>'
)


# ─── 로딩 카드 ────────────────────────────────────────────────────────────────
def loading_card_html(
    icon: str,
    title: str,
    body: str,
    elapsed: int = 0,
    done: bool = False,
) -> str:
    """분석 진행 상태를 보여주는 로딩 카드 HTML을 반환합니다."""
    if done:
        border   = f"1px solid {COLORS['gain']}55"
        glow     = f"rgba({COLORS['gain_rgb']},0.15)"
        fill_bar = f'<div style="background:{COLORS["gain"]};width:100%;height:4px;border-radius:4px;"></div>'
        spin_cls = ""
    else:
        border   = f"1px solid {COLORS['border']}"
        glow     = "rgba(0,0,0,0.4)"
        fill_bar = '<div class="loading-bar-fill"></div>'
        spin_cls = 'class="loading-icon" '

    return f"""
<div style="background:{COLORS['surface']};border:{border};border-radius:{RADIUS};
            padding:28px 28px;text-align:center;margin:8px 0 16px;
            box-shadow:0 4px 24px {glow};">
  <div {spin_cls}style="font-size:36px;margin-bottom:12px;">{icon}</div>
  <div style="font-size:1.05rem;font-weight:700;color:{COLORS['text']};margin-bottom:8px;">{title}</div>
  <div style="color:{COLORS['text_2']};font-size:0.88rem;line-height:1.7;">{body}</div>
  <div class="loading-bar-track">{fill_bar}</div>
  <div style="color:{COLORS['text_2']};margin:10px 0 0;font-size:0.78rem;">
    ⏱ <b style="color:{COLORS['text']};">{elapsed}s</b> 경과
  </div>
</div>"""


# ─── 환율 카드 ────────────────────────────────────────────────────────────────
def rate_card_html(pair: str, rate: float, change: float) -> str:
    """환율 단일 행 카드 HTML."""
    arrow = "▲" if change > 0 else "▼"
    color = COLORS["gain"] if change > 0 else COLORS["loss"]
    return f"""
<div style="background:{COLORS['surface']};border:1px solid {COLORS['border']};border-radius:10px;
            padding:12px 16px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-size:0.72rem;color:{COLORS['text_2']};margin-bottom:3px;">{pair}</div>
    <div style="font-size:1.3rem;font-weight:700;color:{COLORS['text']};">{rate:,.2f}</div>
  </div>
  <div style="font-size:0.9rem;font-weight:700;color:{color};">{arrow} {abs(change):.3f}%</div>
</div>"""


# ─── 포트폴리오 헤더 카드 ─────────────────────────────────────────────────────
def header_metric_card_html(
    svg_icon: str,
    label: str,
    value: str,
    subtitle: str,
    icon_color: str,
    border_rgb: str,
    glow_rgb: str,
) -> str:
    """포트폴리오 요약 헤더 카드 HTML."""
    return f"""
<div class="ma-header-card" style="border:1px solid rgba({border_rgb},.25)">
  <div style="position:absolute;top:-15px;right:-15px;width:70px;height:70px;
              background:radial-gradient(circle,rgba({glow_rgb},.12) 0%,transparent 70%);border-radius:50%"></div>
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
    <span style="font-size:.68rem;color:{COLORS['text_2']};font-weight:500;
                 letter-spacing:.8px;text-transform:uppercase">{label}</span>
    <span style="color:{icon_color}">{svg_icon}</span>
  </div>
  <div class="key-metric">{value}</div>
  <div style="font-size:.72rem;color:{icon_color};font-weight:600;margin-top:10px">{subtitle}</div>
</div>"""


def placeholder_card_html() -> str:
    """데이터 없을 때 표시하는 빈 헤더 카드."""
    return (
        f'<div class="ma-header-card" style="border:1px solid {COLORS["border"]};text-align:center;">'
        f'<div style="font-size:1.6rem;font-weight:700;color:{COLORS["border_md"]};margin-bottom:8px;">—</div>'
        f'<div style="font-size:.72rem;color:{COLORS["text_3"]};">💼 포트폴리오 탭 접속 후 갱신</div></div>'
    )


# ─── 관심종목 아이템 ──────────────────────────────────────────────────────────
def watchlist_item_html(name: str, change_pct: float) -> str:
    """관심종목 이름 + 등락률 인라인 HTML."""
    arrow = "▲" if change_pct >= 0 else "▼"
    color = COLORS["gain"] if change_pct >= 0 else COLORS["loss"]
    return (
        f"**{name}**<br>"
        f"<span style='color:{color};font-size:0.85rem;font-weight:600;'>"
        f"{arrow} {change_pct:+.2f}%</span>"
    )


# ─── 감성 배지 ────────────────────────────────────────────────────────────────
def sentiment_badge_html(senti: str, score: float) -> str:
    """뉴스 감성 컬러 배지 HTML."""
    if senti == "긍정":
        fg, bg = "#a5d6a7", "#1b5e20"
    elif senti == "부정":
        fg, bg = "#ef9a9a", "#b71c1c"
    else:
        fg, bg = "#bdbdbd", "#212121"
    return (
        f'<span style="background:{bg};color:{fg};border-radius:6px;'
        f'padding:4px 14px;font-size:0.9rem;font-weight:bold;">'
        f'{senti} &nbsp; {score:+.2f}</span>'
    )


# ─── 거래정지 경고 배너 ───────────────────────────────────────────────────────
def halted_banner_html(reason: str, recent_vol: int, avg_vol: int, ratio: float) -> str:
    """거래정지 / 주의 종목 경고 카드 HTML."""
    c_loss = COLORS["loss"]
    return (
        f'<div style="background:rgba(239,83,80,0.08);border:1px solid {c_loss};border-radius:{RADIUS};'
        f'padding:14px 20px;margin-bottom:16px;">'
        f'<div style="font-size:1rem;font-weight:700;color:{c_loss};margin-bottom:8px;">⛔ 거래 정지 / 주의</div>'
        f'<div style="color:{COLORS["text"]};font-size:0.88rem;line-height:1.6;margin-bottom:6px;">{reason}</div>'
        f'<div style="color:{COLORS["text_2"]};font-size:0.78rem;">'
        f'최근 거래량: {recent_vol:,} &nbsp;·&nbsp; '
        f'20일 평균: {int(avg_vol):,} &nbsp;·&nbsp; '
        f'비율: {ratio * 100:.1f}%</div>'
        f'<div style="color:{COLORS["text_2"]};font-size:0.75rem;margin-top:4px;">'
        f'기술적 분석 점수 합산이 중단되었습니다. 거래 재개 후 분석을 다시 실행하세요.</div>'
        f'</div>'
    )


# ─── AI 종합 리포트 배너 ──────────────────────────────────────────────────────
def signal_report_html(
    signal: str,            # "BUY" / "SELL" / "WAIT"
    action: str,
    reasons: list[str],
    h_label: str,
    h_badge: str,
    h_score: float,
    news_score: float,
    fund_score: int,
    fund_label: str,
    cur_price: float,
    sl_price: str,
    tgt_price: str,
    is_krw: bool,
) -> str:
    """AI 종합 리포트 전체 너비 배너 HTML."""
    # 신호별 색상
    if signal == "BUY":
        border = "#22c55e"
        fc     = "#4ade80"
        glow   = "rgba(34,197,94,0.15)"
    elif signal == "SELL":
        border = "#ef4444"
        fc     = "#f87171"
        glow   = "rgba(239,68,68,0.15)"
    else:
        border = "#eab308"
        fc     = "#facc15"
        glow   = "rgba(234,179,8,0.12)"

    emoji   = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "🟡")
    tech_c  = "#10B981" if h_score >= 0 else "#ef4444"
    news_c  = "#10B981" if news_score >= 0 else "#ef4444"
    fund_c  = "#80cbc4" if fund_score >= 3 else ("#ffcc80" if fund_score <= -2 else "#94A3B8")

    fmt     = "{:,.0f}" if is_krw else "{:,.2f}"
    cur_str = (("₩" if is_krw else "$") + fmt.format(cur_price)) if cur_price > 0 else "—"

    reasons_html = "".join(
        f'<span style="display:inline-block;background:rgba(255,255,255,0.06);'
        f'border-radius:20px;padding:3px 12px;margin:3px 4px;font-size:0.78rem;color:#CBD5E1;">'
        f'{emoji} {r}</span>'
        for r in reasons[:3]
    )

    _bg     = COLORS["surface"]
    _border = COLORS["border"]
    _text2  = COLORS["text_2"]
    _text3  = COLORS["text_3"]
    _txt    = COLORS["text"]
    _bg_main = COLORS["bg"]

    return f"""
<div style="background:{_bg};border:1px solid {border}66;border-radius:{RADIUS};
            padding:22px 24px;margin-bottom:16px;
            box-shadow:0 0 32px {glow},0 4px 20px rgba(0,0,0,0.35);">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:240px;">
      <div style="font-size:0.65rem;color:{_text2};letter-spacing:3px;text-transform:uppercase;
                  margin-bottom:10px;display:flex;align-items:center;gap:6px;">
        <span style="width:4px;height:4px;border-radius:50%;background:{border};display:inline-block;"></span>
        AI 종합 리포트
      </div>
      <div style="font-size:2.4rem;font-weight:900;color:{fc};letter-spacing:2px;line-height:1.05;margin-bottom:10px;">
        {emoji} {signal}
      </div>
      <div style="font-size:0.88rem;color:{_txt};line-height:1.7;margin-bottom:10px;">{action}</div>
      <div>{reasons_html}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;min-width:270px;">
      <div style="background:{_bg_main};border:1px solid {_border};border-radius:10px;padding:10px 12px;text-align:center;">
        <div style="font-size:0.62rem;color:{_text3};letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">단타 신호</div>
        <div style="font-size:0.9rem;font-weight:700;color:{tech_c};">{h_badge} {h_label}</div>
        <div style="font-size:0.72rem;color:{tech_c};opacity:.75;margin-top:2px;">{h_score:+.1f}점</div>
      </div>
      <div style="background:{_bg_main};border:1px solid {_border};border-radius:10px;padding:10px 12px;text-align:center;">
        <div style="font-size:0.62rem;color:{_text3};letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">뉴스 감성</div>
        <div style="font-size:0.9rem;font-weight:700;color:{news_c};">{news_score:+.1f}점</div>
        <div style="font-size:0.72rem;color:{_text3};margin-top:2px;">±5 기준</div>
      </div>
      <div style="background:{_bg_main};border:1px solid {_border};border-radius:10px;padding:10px 12px;text-align:center;">
        <div style="font-size:0.62rem;color:{_text3};letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">장투 신호</div>
        <div style="font-size:0.9rem;font-weight:700;color:{fund_c};">{fund_label}</div>
        <div style="font-size:0.72rem;color:{fund_c};opacity:.75;margin-top:2px;">{fund_score:+.1f}점</div>
      </div>
      <div style="background:{_bg_main};border:1px solid {_border};border-radius:10px;padding:10px 12px;text-align:center;">
        <div style="font-size:0.62rem;color:{_text3};letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">현재가</div>
        <div style="font-size:0.9rem;font-weight:700;color:{_txt};">{cur_str}</div>
      </div>
      <div style="background:{_bg_main};border:1px solid {_border};border-radius:10px;padding:10px 12px;text-align:center;">
        <div style="font-size:0.62rem;color:{_text3};letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">1차 목표가</div>
        <div style="font-size:0.9rem;font-weight:700;color:{COLORS['gain']};">{tgt_price}</div>
      </div>
      <div style="background:{_bg_main};border:1px solid {_border};border-radius:10px;padding:10px 12px;text-align:center;">
        <div style="font-size:0.62rem;color:{_text3};letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">손절가</div>
        <div style="font-size:0.9rem;font-weight:700;color:{COLORS['loss']};">{sl_price}</div>
      </div>
    </div>
  </div>
</div>"""


# ─── 종목 배지 (차트 탭 상단) ─────────────────────────────────────────────────
def stock_badge_html(
    title_label: str,
    price_str: str,
    is_realtime: bool,
    chg_pct: float | None = None,
    rt_ts: str = "",
) -> str:
    """차트 탭 상단 종목명 + 현재가 + 등락률 배지 HTML."""
    rt_label = "● 실시간" if is_realtime else "○ 장마감"
    rt_color = COLORS["gain"] if is_realtime else COLORS["text_2"]

    if chg_pct is not None:
        _chg_c = "#10B981" if chg_pct >= 0 else "#ef4444"
        _chg_arrow = "▲" if chg_pct >= 0 else "▼"
        chg_html = (
            f'<span style="font-size:0.85rem;font-weight:700;color:{_chg_c};margin-left:8px;">'
            f'{_chg_arrow} {chg_pct:+.2f}%</span>'
        )
    else:
        chg_html = ""

    ts_html = (
        f'<span style="color:#64748B;margin-left:6px;">· {rt_ts}</span>'
        if rt_ts else ""
    )

    return f"""
<div class="ma-stock-badge">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div>
      <div class="gradient-text" style="font-size:1.15rem;font-weight:800">{title_label}</div>
      <div style="font-size:.72rem;color:#94A3B8;margin-top:3px">
        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
             stroke-linejoin="round" style="vertical-align:middle;margin-right:3px">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
        기술적 분석 · AI 매매 신호
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:1.3rem;font-weight:700;color:#E2E8F0;display:flex;align-items:center;">{price_str}{chg_html}</div>
      <div style="font-size:.68rem;color:{rt_color};margin-top:2px">{rt_label}{ts_html}</div>
    </div>
  </div>
</div>"""


# ─── 섹션 구분선 제목 ─────────────────────────────────────────────────────────
def section_heading_html(title: str, accent: str = COLORS["accent"]) -> str:
    """네온 포인트 좌측 바 + 제목 HTML."""
    return (
        f'<div style="display:flex;align-items:center;gap:10px;margin:20px 0 10px;">'
        f'<span style="width:3px;height:18px;background:{accent};border-radius:2px;'
        f'display:inline-block;box-shadow:0 0 8px {accent}88;"></span>'
        f'<span style="font-size:1rem;font-weight:700;color:{COLORS["text"]};">{title}</span>'
        f'</div>'
    )


# ─── 인포 칩 ──────────────────────────────────────────────────────────────────
def info_chip_html(text: str, color: str = COLORS["text_2"]) -> str:
    """소형 정보 칩 HTML."""
    return (
        f'<span style="display:inline-block;background:rgba(255,255,255,0.05);'
        f'border:1px solid {COLORS["border"]};border-radius:20px;padding:3px 10px;'
        f'font-size:0.76rem;color:{color};margin:2px 3px;">{text}</span>'
    )
