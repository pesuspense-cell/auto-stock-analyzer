"""
ui/components.py — 재사용 가능한 HTML 컴포넌트 빌더

모든 함수는 순수 HTML 문자열을 반환합니다.
st.markdown(..., unsafe_allow_html=True) 로 렌더링하세요.
"""
from __future__ import annotations
from ui.styles import COLORS, RADIUS, RADIUS_LG, SHADOW

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

_C = COLORS
_FONT_DISPLAY = "'SF Pro Display',system-ui,-apple-system,sans-serif"
_FONT_TEXT    = "'SF Pro Text',system-ui,-apple-system,sans-serif"


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
        border   = f"1px solid {_C['gain']}55"
        fill_bar = (
            f'<div style="background:{_C["gain"]};width:100%;height:3px;border-radius:4px;"></div>'
        )
        spin_cls = ""
    else:
        border   = f"1px solid {_C['border']}"
        fill_bar = '<div class="loading-bar-fill"></div>'
        spin_cls = 'class="loading-icon" '

    return f"""
<div style="background:{_C['surface']};border:{border};border-radius:{RADIUS};
            padding:32px 28px;text-align:center;margin:8px 0 16px;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);">
  <div {spin_cls}style="font-size:36px;margin-bottom:14px;">{icon}</div>
  <div style="font-size:1.05rem;font-weight:600;color:{_C['text']};margin-bottom:8px;
              font-family:{_FONT_DISPLAY};letter-spacing:-0.374px;">{title}</div>
  <div style="color:{_C['text_2']};font-size:0.875rem;line-height:1.7;
              font-family:{_FONT_TEXT};">{body}</div>
  <div class="loading-bar-track">{fill_bar}</div>
  <div style="color:{_C['text_2']};margin:10px 0 0;font-size:0.78rem;
              font-family:{_FONT_TEXT};">
    ⏱ <b style="color:{_C['text']};">{elapsed}s</b> 경과
  </div>
</div>"""


# ─── 환율 카드 ────────────────────────────────────────────────────────────────
def rate_card_html(pair: str, rate: float, change: float) -> str:
    """환율 단일 행 카드 HTML."""
    arrow = "▲" if change > 0 else "▼"
    color = _C["gain"] if change > 0 else _C["loss"]
    return f"""
<div style="background:{_C['surface']};border:1px solid {_C['border']};border-radius:10px;
            padding:12px 16px;margin:6px 0;display:flex;justify-content:space-between;
            align-items:center;box-shadow:0 1px 2px rgba(0,0,0,0.06);">
  <div>
    <div style="font-size:0.72rem;color:{_C['text_2']};margin-bottom:3px;
                font-family:{_FONT_TEXT};">{pair}</div>
    <div style="font-size:1.3rem;font-weight:600;color:{_C['text']};
                font-family:{_FONT_DISPLAY};letter-spacing:-0.28px;
                font-variant-numeric:tabular-nums;">{rate:,.2f}</div>
  </div>
  <div style="font-size:0.9rem;font-weight:600;color:{color};
              font-variant-numeric:tabular-nums;">{arrow} {abs(change):.3f}%</div>
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
<div class="ma-header-card" style="border:1px solid rgba({border_rgb},.18)">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
    <span style="font-size:.72rem;color:{_C['text_2']};font-weight:600;
                 letter-spacing:-0.12px;font-family:{_FONT_TEXT};">{label}</span>
    <span style="color:{icon_color}">{svg_icon}</span>
  </div>
  <div class="key-metric">{value}</div>
  <div style="font-size:.75rem;color:{icon_color};font-weight:600;margin-top:10px;
              font-family:{_FONT_TEXT};">{subtitle}</div>
</div>"""


def placeholder_card_html() -> str:
    """데이터 없을 때 표시하는 빈 헤더 카드."""
    return (
        f'<div class="ma-header-card" style="border:1px solid {_C["border"]};text-align:center;">'
        f'<div style="font-size:1.6rem;font-weight:600;color:{_C["border_md"]};margin-bottom:8px;">—</div>'
        f'<div style="font-size:.72rem;color:{_C["text_3"]};font-family:{_FONT_TEXT};">'
        f'포트폴리오 탭 접속 후 갱신</div></div>'
    )


# ─── 관심종목 아이템 ──────────────────────────────────────────────────────────
def watchlist_item_html(name: str, change_pct: float) -> str:
    """관심종목 이름 + 등락률 인라인 HTML."""
    arrow = "▲" if change_pct >= 0 else "▼"
    color = _C["gain"] if change_pct >= 0 else _C["loss"]
    return (
        f"**{name}**<br>"
        f"<span style='color:{color};font-size:0.85rem;font-weight:600;"
        f"font-variant-numeric:tabular-nums;'>"
        f"{arrow} {change_pct:+.2f}%</span>"
    )


# ─── 감성 배지 ────────────────────────────────────────────────────────────────
def sentiment_badge_html(senti: str, score: float) -> str:
    """뉴스 감성 컬러 배지 HTML."""
    if senti == "긍정":
        fg, bg = "#1a7a35", "rgba(52,199,89,0.12)"
        border = "rgba(52,199,89,0.3)"
    elif senti == "부정":
        fg, bg = "#cc2200", "rgba(255,59,48,0.10)"
        border = "rgba(255,59,48,0.3)"
    else:
        fg, bg = _C["text_2"], "rgba(0,0,0,0.04)"
        border = _C["border"]
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {border};'
        f'border-radius:9999px;padding:4px 14px;font-size:0.9rem;font-weight:600;'
        f'font-family:{_FONT_TEXT};">'
        f'{senti} &nbsp; {score:+.2f}</span>'
    )


# ─── 거래정지 경고 배너 ───────────────────────────────────────────────────────
def halted_banner_html(reason: str, recent_vol: int, avg_vol: int, ratio: float) -> str:
    """거래정지 / 주의 종목 경고 카드 HTML."""
    c_loss = _C["loss"]
    return (
        f'<div style="background:rgba(255,59,48,0.06);border:1px solid rgba(255,59,48,0.3);'
        f'border-radius:{RADIUS};padding:16px 20px;margin-bottom:16px;">'
        f'<div style="font-size:1rem;font-weight:600;color:{c_loss};margin-bottom:8px;'
        f'font-family:{_FONT_DISPLAY};letter-spacing:-0.374px;">거래 정지 / 주의</div>'
        f'<div style="color:{_C["text"]};font-size:0.875rem;line-height:1.6;margin-bottom:6px;">{reason}</div>'
        f'<div style="color:{_C["text_2"]};font-size:0.78rem;">'
        f'최근 거래량: {recent_vol:,} &nbsp;·&nbsp; '
        f'20일 평균: {int(avg_vol):,} &nbsp;·&nbsp; '
        f'비율: {ratio * 100:.1f}%</div>'
        f'<div style="color:{_C["text_2"]};font-size:0.75rem;margin-top:4px;">'
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
    if signal == "BUY":
        border = _C["gain"]
        fc     = _C["gain"]
        bg_tint = "rgba(52,199,89,0.06)"
    elif signal == "SELL":
        border = _C["loss"]
        fc     = _C["loss"]
        bg_tint = "rgba(255,59,48,0.06)"
    else:
        border = "#ff9500"
        fc     = "#e07000"
        bg_tint = "rgba(255,149,0,0.06)"

    emoji   = "●" if signal == "BUY" else ("●" if signal == "SELL" else "●")
    dot_c   = _C["gain"] if signal == "BUY" else (_C["loss"] if signal == "SELL" else "#ff9500")
    tech_c  = _C["gain"] if h_score >= 0 else _C["loss"]
    news_c  = _C["gain"] if news_score >= 0 else _C["loss"]
    fund_c  = _C["gain"] if fund_score >= 3 else (_C["loss"] if fund_score <= -2 else _C["text_2"])

    fmt     = "{:,.0f}" if is_krw else "{:,.2f}"
    cur_str = (("₩" if is_krw else "$") + fmt.format(cur_price)) if cur_price > 0 else "—"

    reasons_html = "".join(
        f'<span style="display:inline-block;background:rgba(0,0,0,0.04);'
        f'border:1px solid {_C["border"]};border-radius:9999px;'
        f'padding:3px 12px;margin:3px 4px;font-size:0.78rem;color:{_C["text_2"]};">'
        f'{r}</span>'
        for r in reasons[:3]
    )

    _bg     = _C["surface"]
    _border = _C["border"]
    _text2  = _C["text_2"]
    _text3  = _C["text_3"]
    _txt    = _C["text"]
    _parchment = _C["bg"]

    def _mini_card(top_label: str, val: str, sub: str, val_color: str) -> str:
        return (
            f'<div style="background:{_parchment};border:1px solid {_border};'
            f'border-radius:{RADIUS};padding:10px 12px;text-align:center;">'
            f'<div style="font-size:0.65rem;color:{_text3};letter-spacing:0;'
            f'font-family:{_FONT_TEXT};margin-bottom:4px;">{top_label}</div>'
            f'<div style="font-size:0.9rem;font-weight:600;color:{val_color};'
            f'font-variant-numeric:tabular-nums;font-family:{_FONT_DISPLAY};">{val}</div>'
            f'<div style="font-size:0.72rem;color:{val_color};opacity:.8;margin-top:2px;'
            f'font-family:{_FONT_TEXT};">{sub}</div>'
            f'</div>'
        )

    return f"""
<div style="background:{_bg};border:1px solid rgba({_border_to_rgb(border)},.3);
            border-radius:{RADIUS};padding:24px 24px;margin-bottom:16px;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);border-top:3px solid {border};">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:240px;">
      <div style="font-size:0.65rem;color:{_text2};letter-spacing:0;
                  font-family:{_FONT_TEXT};margin-bottom:10px;
                  display:flex;align-items:center;gap:6px;">
        <span style="width:6px;height:6px;border-radius:50%;background:{dot_c};
                     display:inline-block;flex-shrink:0;"></span>
        AI 종합 리포트
      </div>
      <div style="font-size:2rem;font-weight:600;color:{fc};letter-spacing:-0.28px;
                  line-height:1.07;margin-bottom:10px;
                  font-family:{_FONT_DISPLAY};font-variant-numeric:tabular-nums;">
        {signal}
      </div>
      <div style="font-size:0.875rem;color:{_txt};line-height:1.7;margin-bottom:10px;
                  font-family:{_FONT_TEXT};">{action}</div>
      <div>{reasons_html}</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;min-width:270px;">
      {_mini_card("단타 신호", f"{h_badge} {h_label}", f"{h_score:+.1f}점", tech_c)}
      {_mini_card("뉴스 감성", f"{news_score:+.1f}점", "±5 기준", news_c)}
      {_mini_card("장투 신호", fund_label, f"{fund_score:+.1f}점", fund_c)}
      {_mini_card("현재가", cur_str, "", _txt)}
      {_mini_card("1차 목표가", tgt_price, "", _C["gain"])}
      {_mini_card("손절가", sl_price, "", _C["loss"])}
    </div>
  </div>
</div>"""


def _border_to_rgb(hex_color: str) -> str:
    """#rrggbb → 'r,g,b' 변환 (간단 헬퍼)."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "0,0,0"


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
    rt_color = _C["gain"] if is_realtime else _C["text_2"]

    if chg_pct is not None:
        _chg_c = _C["gain"] if chg_pct >= 0 else _C["loss"]
        _chg_arrow = "▲" if chg_pct >= 0 else "▼"
        chg_html = (
            f'<span style="font-size:0.88rem;font-weight:600;color:{_chg_c};'
            f'margin-left:8px;font-variant-numeric:tabular-nums;">'
            f'{_chg_arrow} {chg_pct:+.2f}%</span>'
        )
    else:
        chg_html = ""

    ts_html = (
        f'<span style="color:{_C["text_2"]};margin-left:6px;">· {rt_ts}</span>'
        if rt_ts else ""
    )

    return f"""
<div class="ma-stock-badge">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div>
      <div style="font-size:1.15rem;font-weight:600;color:{_C['text']};
                  font-family:{_FONT_DISPLAY};letter-spacing:-0.374px;">{title_label}</div>
      <div style="font-size:.72rem;color:{_C['text_2']};margin-top:3px;
                  font-family:{_FONT_TEXT};">
        <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
             stroke-linejoin="round" style="vertical-align:middle;margin-right:3px">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
        기술적 분석 · AI 매매 신호
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:1.4rem;font-weight:600;color:{_C['text']};
                  display:flex;align-items:center;font-family:{_FONT_DISPLAY};
                  letter-spacing:-0.28px;font-variant-numeric:tabular-nums;">
        {price_str}{chg_html}
      </div>
      <div style="font-size:.7rem;color:{rt_color};margin-top:2px;
                  font-family:{_FONT_TEXT};">{rt_label}{ts_html}</div>
    </div>
  </div>
</div>"""


# ─── 섹션 구분선 제목 ─────────────────────────────────────────────────────────
def section_heading_html(title: str, accent: str = "") -> str:
    """좌측 Action Blue 바 + 제목 HTML."""
    color = accent or _C["accent"]
    return (
        f'<div class="section-heading">'
        f'<span class="section-heading-bar" style="background:{color};"></span>'
        f'<span class="section-heading-text">{title}</span>'
        f'</div>'
    )


# ─── 인포 칩 ──────────────────────────────────────────────────────────────────
def info_chip_html(text: str, color: str = "") -> str:
    """소형 정보 칩 HTML."""
    fg = color or _C["text_2"]
    return (
        f'<span style="display:inline-block;background:rgba(0,0,0,0.04);'
        f'border:1px solid {_C["border"]};border-radius:9999px;padding:3px 10px;'
        f'font-size:0.76rem;color:{fg};margin:2px 3px;'
        f'font-family:{_FONT_TEXT};">{text}</span>'
    )


# ─── SDT: 헤더 바 ─────────────────────────────────────────────────────────────
def stock_dashboard_header_html(
    title: str,
    price_str: str,
    chg_pct: float | None,
    is_realtime: bool,
    rt_ts: str,
) -> str:
    """Stock Dashboard Tile — header bar with stock name, live price, and change badge."""
    rt_label = "● 실시간" if is_realtime else "○ 장마감"
    rt_color  = _C["gain"] if is_realtime else _C["text_3"]

    if chg_pct is not None:
        _c     = _C["gain"] if chg_pct >= 0 else _C["loss"]
        _arrow = "▲" if chg_pct >= 0 else "▼"
        chg_html = (
            f'<span class="sdt-chg" style="color:{_c};">{_arrow}&nbsp;{chg_pct:+.2f}%</span>'
        )
    else:
        chg_html = f'<span class="sdt-chg" style="color:{_C["text_3"]};">— —</span>'

    return (
        f'<div class="sdt-header">'
        f'<span class="sdt-stock-name">{title}</span>'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;flex-shrink:0;gap:2px;">'
        f'<div class="sdt-price-row">'
        f'<span class="sdt-price">{price_str}</span>'
        f'{chg_html}'
        f'</div>'
        f'<span style="font-size:.68rem;color:{rt_color};font-family:{_FONT_TEXT};">'
        f'{rt_label}{(" · " + rt_ts) if rt_ts else ""}'
        f'</span>'
        f'</div>'
        f'</div>'
    )


# ─── SDT: AI Signal Card (우측 패널) ──────────────────────────────────────────
def signal_card_compact_html(
    signal: str,
    h_label: str,
    h_score: float,
    exp_return: float | None,
    risk_state: str,
    buy_price_str: str,
    sell_price_str: str,
) -> str:
    """Stock Dashboard Tile — compact AI signal card for the right split-pane."""
    if signal == "BUY":
        pill_bg, pill_fg, top_border = "var(--gain-dim)", _C["gain"],  _C["gain"]
        _br = "rgba(52,199,89,0.3)"
    elif signal == "SELL":
        pill_bg, pill_fg, top_border = "var(--loss-dim)", _C["loss"],  _C["loss"]
        _br = "rgba(255,59,48,0.3)"
    else:
        pill_bg, pill_fg, top_border = "rgba(255,149,0,0.10)", "#e07000", "#ff9500"
        _br = "rgba(255,149,0,0.3)"

    score_c = _C["gain"] if h_score >= 0 else _C["loss"]
    ret_c   = (
        (_C["gain"] if (exp_return or 0) >= 0 else _C["loss"])
        if exp_return is not None else _C["text_2"]
    )
    ret_str = f"{exp_return:+.1f}%" if exp_return is not None else "—"

    def _row(key: str, val: str, val_color: str | None = None) -> str:
        vc = val_color or _C["text"]
        return (
            f'<div class="sdt-signal-row">'
            f'<span class="sdt-signal-key">{key}</span>'
            f'<span class="sdt-signal-val" style="color:{vc};">{val}</span>'
            f'</div>'
        )

    return (
        f'<div class="sdt-signal-card" style="border-top:3px solid {top_border};">'
        f'<div class="sdt-signal-eyebrow">'
        f'<span style="width:6px;height:6px;border-radius:50%;background:{top_border};'
        f'display:inline-block;flex-shrink:0;"></span>'
        f'Enhanced Hybrid Signal'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
        f'<span class="sdt-signal-pill" style="background:{pill_bg};color:{pill_fg};'
        f'border:1px solid {_br};">{signal}</span>'
        f'<span style="font-size:0.82rem;font-weight:600;color:{score_c};'
        f'font-variant-numeric:tabular-nums;font-family:{_FONT_DISPLAY};">'
        f'{h_score:+.1f}점</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:{_C["text_2"]};margin-bottom:14px;'
        f'line-height:1.45;word-break:keep-all;font-family:{_FONT_TEXT};">{h_label}</div>'
        f'{_row("예상 수익률", ret_str, ret_c)}'
        f'{_row("리스크 상태", risk_state)}'
        f'{_row("추천 매수가", buy_price_str, _C["gain"])}'
        f'{_row("1차 목표가", sell_price_str, _C["gain"])}'
        f'</div>'
    )


# ─── SDT: 서브 데이터 그리드 (3칼럼) ───────────────────────────────────────────
def sub_data_grid_html(
    fund_score: float,
    fund_label: str,
    fund_weight: str,
    news_score: float,
    news_senti: str,
    pos_kw: int,
    neg_kw: int,
    vol_anomaly: bool,
    adv_rsi: float | None,
    adv_macd_cross: str,
    breakout_status: str,
) -> str:
    """Stock Dashboard Tile — 3-column sub-data grid (Fundamental · Sentiment · Technical)."""
    # ── Fundamental Score Card ────────────────────────────────────────────────
    if fund_score >= 3:    _fc, _fb = _C["gain"],   "rgba(52,199,89,0.06)"
    elif fund_score >= 1:  _fc, _fb = _C["accent"],  "rgba(0,102,204,0.06)"
    elif fund_score <= -2: _fc, _fb = _C["loss"],   "rgba(255,59,48,0.06)"
    else:                  _fc, _fb = _C["text_2"],  "rgba(0,0,0,0.02)"

    _fw_html = (
        f'&nbsp;<span style="font-size:0.72rem;color:{_C["text_2"]};font-weight:500;">'
        f'(Weight: {fund_weight})</span>'
    ) if fund_weight else ""

    card_fund = (
        f'<div class="sdt-sub-card" style="background:{_fb};">'
        f'<div class="sdt-sub-eyebrow">Fundamental Score</div>'
        f'<div class="sdt-sub-score" style="color:{_fc};">{fund_score:+.0f}'
        f'<span style="font-size:0.72rem;color:{_C["text_2"]};font-weight:500;"> / ±6</span>'
        f'</div>'
        f'<div class="sdt-sub-label" style="color:{_fc};">{fund_label}{_fw_html}</div>'
        f'<div class="sdt-sub-row">'
        f'<span class="sdt-sub-row-k">판정 기준</span>'
        f'<span class="sdt-sub-row-v" style="color:{_C["text_2"]};">Graham·Buffett·Lynch</span>'
        f'</div>'
        f'</div>'
    )

    # ── Sentiment Analysis Card ───────────────────────────────────────────────
    if news_senti == "긍정":   _sc, _sb = _C["gain"],  "rgba(52,199,89,0.06)"
    elif news_senti == "부정": _sc, _sb = _C["loss"],  "rgba(255,59,48,0.06)"
    else:                       _sc, _sb = _C["text_2"], "rgba(0,0,0,0.02)"

    _kw_pos_c = _C["gain"]   if pos_kw > 0 else _C["text_2"]
    _kw_neg_c = _C["loss"]   if neg_kw > 0 else _C["text_2"]

    card_senti = (
        f'<div class="sdt-sub-card" style="background:{_sb};">'
        f'<div class="sdt-sub-eyebrow">Sentiment Analysis</div>'
        f'<div class="sdt-sub-score" style="color:{_sc};">{news_score:+.1f}'
        f'<span style="font-size:0.72rem;color:{_C["text_2"]};font-weight:500;"> pts</span>'
        f'</div>'
        f'<div class="sdt-sub-label" style="color:{_sc};">{news_senti}</div>'
        f'<div class="sdt-sub-row">'
        f'<span class="sdt-sub-row-k">긍정 키워드</span>'
        f'<span class="sdt-sub-row-v" style="color:{_kw_pos_c};">+{pos_kw}</span>'
        f'</div>'
        f'<div class="sdt-sub-row">'
        f'<span class="sdt-sub-row-k">부정 키워드</span>'
        f'<span class="sdt-sub-row-v" style="color:{_kw_neg_c};">−{neg_kw}</span>'
        f'</div>'
        f'</div>'
    )

    # ── Technical Analysis Card ───────────────────────────────────────────────
    _va_lbl = "감지됨 ⚠️" if vol_anomaly else "정상"
    _va_c   = _C["loss"] if vol_anomaly else _C["gain"]

    _rsi_str = f"{adv_rsi:.1f}" if adv_rsi is not None else "—"
    _rsi_c   = (
        (_C["loss"] if adv_rsi > 70 else (_C["gain"] if adv_rsi < 30 else _C["text"]))
        if adv_rsi is not None else _C["text_2"]
    )

    _macd_c = (
        _C["gain"] if "골든" in adv_macd_cross else
        (_C["loss"] if "데드" in adv_macd_cross else _C["text_2"])
    ) if adv_macd_cross else _C["text_2"]

    if breakout_status == "breakout_both":
        _bk_lbl, _bk_c = "돌파 확인 🚀", _C["gain"]
    elif "breakout" in breakout_status:
        _bk_lbl, _bk_c = "부분 돌파",   _C["accent"]
    else:
        _bk_lbl, _bk_c = "관망",         _C["text_2"]

    card_tech = (
        f'<div class="sdt-sub-card">'
        f'<div class="sdt-sub-eyebrow">Technical Analysis</div>'
        f'<div class="sdt-sub-row">'
        f'<span class="sdt-sub-row-k">거래량 이상</span>'
        f'<span class="sdt-sub-row-v" style="color:{_va_c};">{_va_lbl}</span>'
        f'</div>'
        f'<div class="sdt-sub-row">'
        f'<span class="sdt-sub-row-k">RSI (14)</span>'
        f'<span class="sdt-sub-row-v" style="color:{_rsi_c};">{_rsi_str}</span>'
        f'</div>'
        f'<div class="sdt-sub-row">'
        f'<span class="sdt-sub-row-k">MACD 크로스</span>'
        f'<span class="sdt-sub-row-v" style="color:{_macd_c};">{adv_macd_cross or "—"}</span>'
        f'</div>'
        f'<div class="sdt-sub-row">'
        f'<span class="sdt-sub-row-k">돌파 신호</span>'
        f'<span class="sdt-sub-row-v" style="color:{_bk_c};">{_bk_lbl}</span>'
        f'</div>'
        f'</div>'
    )

    return (
        f'<div class="sdt-sub-grid sdt-fade-in">'
        f'{card_fund}{card_senti}{card_tech}'
        f'</div>'
    )
