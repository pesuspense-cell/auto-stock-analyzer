"""
ui/backtest_tab.py — 백테스트 탭 렌더링
"""
from __future__ import annotations

import contextlib
import io
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 마켓 선택지 정의
_MARKETS = ["KOSPI", "KOSDAQ", "S&P500", "NASDAQ"]

_BENCHMARK_OPTIONS = {
    "^KS11 (KOSPI)":   "^KS11",
    "^KQ11 (KOSDAQ)":  "^KQ11",
    "^IXIC (NASDAQ)":  "^IXIC",
    "^GSPC (S&P500)":  "^GSPC",
    "없음":            None,
}


def render_backtest_tab(tab) -> None:
    with tab:
        st.subheader("🔬 백테스트")
        st.caption(
            "전체 종목 유니버스에서 모멘텀·추세·거래량 기준으로 종목을 자동 선정하고, "
            "TradingStrategy(SMA 골든크로스 + ATR 익절·손절 + 분할매수)로 과거 수익률을 검증합니다."
        )

        # ── 설정 폼 ────────────────────────────────────────────────────────
        with st.form("backtest_form"):
            st.markdown("#### 📡 종목 자동 선정 (스크리닝)")
            col_mkt, col_screen = st.columns([1, 1], gap="large")

            with col_mkt:
                selected_markets = st.multiselect(
                    "대상 마켓",
                    options=_MARKETS,
                    default=["KOSPI", "KOSDAQ"],
                    help="선택한 마켓의 시가총액 상위 종목을 후보로 사용합니다.",
                )
                universe_n = st.slider(
                    "마켓별 후보 종목 수 (시가총액 상위)",
                    min_value=50, max_value=500, value=200, step=50,
                    help="값이 클수록 더 많은 종목을 스크리닝합니다 (시간 증가).",
                )

            with col_screen:
                top_n = st.slider(
                    "최종 선정 종목 수",
                    min_value=5, max_value=50, value=20, step=5,
                    help="스크리닝 통과 종목 중 복합 점수 상위 N개를 백테스트합니다.",
                )
                benchmark_label = st.selectbox(
                    "벤치마크 지수",
                    options=list(_BENCHMARK_OPTIONS.keys()),
                    index=0,
                )

            st.divider()
            st.markdown("#### 📅 백테스트 설정")
            col_l, col_r = st.columns([1, 1], gap="large")

            with col_l:
                c1, c2 = st.columns(2)
                start_date = c1.date_input("시작일", value=date(2020, 1, 1))
                end_date   = c2.date_input("종료일", value=date(2024, 12, 31))
                initial_capital = st.number_input(
                    "초기 자본금 (원)",
                    min_value=1_000_000, max_value=10_000_000_000,
                    value=10_000_000, step=1_000_000, format="%d",
                )

            with col_r:
                position_pct = st.slider(
                    "1회 매수 비중 (총자산 대비 %)",
                    min_value=5, max_value=50, value=20, step=5,
                )
                deposit_text = st.text_area(
                    "추가 입금 일정 (날짜,금액 — 한 줄에 하나씩)",
                    value="2021-01-04,5000000\n2022-01-03,5000000\n2023-01-02,5000000",
                    height=100,
                    help="형식: YYYY-MM-DD,금액  (공휴일은 다음 거래일로 자동 이월)",
                )

            submitted = st.form_submit_button(
                "▶ 스크리닝 후 백테스트 실행",
                use_container_width=True,
                type="primary",
            )

        # ── 실행 ──────────────────────────────────────────────────────────
        if submitted:
            if not selected_markets:
                st.error("마켓을 하나 이상 선택하세요.")
                return
            if start_date >= end_date:
                st.error("종료일이 시작일보다 이후여야 합니다.")
                return

            deposit_schedule: dict[str, float] = {}
            for line in deposit_text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d_str, amt_str = line.split(",", 1)
                    deposit_schedule[d_str.strip()] = float(amt_str.strip())
                except ValueError:
                    st.warning(f"입금 일정 파싱 오류 (무시됨): `{line}`")

            _run_and_display(
                markets          = selected_markets,
                universe_n       = universe_n,
                top_n            = top_n,
                initial_capital  = float(initial_capital),
                start_date       = str(start_date),
                end_date         = str(end_date),
                deposit_schedule = deposit_schedule,
                position_pct     = position_pct / 100,
                benchmark_ticker = _BENCHMARK_OPTIONS[benchmark_label],
                benchmark_label  = benchmark_label,
            )

        elif "bt_result" in st.session_state:
            _display_results(**st.session_state["bt_result"])


# ── 스크리닝 + 백테스트 실행 ────────────────────────────────────────────────────

def _run_and_display(
    markets, universe_n, top_n,
    initial_capital, start_date, end_date,
    deposit_schedule, position_pct,
    benchmark_ticker, benchmark_label,
):
    from backtest import BacktestEngine, StockScreener

    # ── Phase 1: 스크리닝 ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📡 Phase 1 — 종목 스크리닝")
    screen_ph   = st.empty()
    screen_prog = st.progress(0.0)

    def _screen_cb(pct: float, msg: str) -> None:
        screen_ph.caption(msg)
        screen_prog.progress(min(pct / 100, 1.0))

    screener = StockScreener(universe_per_market=universe_n)
    selected = screener.screen(
        markets     = markets,
        top_n       = top_n,
        progress_cb = _screen_cb,
    )

    screen_prog.empty()

    if not selected:
        screen_ph.error("스크리닝 통과 종목이 없습니다. 마켓 또는 필터 설정을 조정하세요.")
        return

    screen_ph.success(
        f"✅ 스크리닝 완료 — 상위 {len(selected)}개 종목 선정 "
        f"({', '.join(m for m in markets)} 유니버스 {universe_n}개/마켓 기준)"
    )

    # 선정 종목 테이블
    df_sel = pd.DataFrame(selected)[["ticker", "name", "momentum", "volatility", "avg_volume"]]
    df_sel.index = range(1, len(df_sel) + 1)
    df_sel.columns = ["티커", "종목명", "3개월 수익률(%)", "변동성(%)", "일평균 거래량"]
    df_sel["3개월 수익률(%)"] = df_sel["3개월 수익률(%)"].apply(lambda x: f"{x:+.1f}%")
    df_sel["변동성(%)"]       = df_sel["변동성(%)"].apply(lambda x: f"{x:.2f}%")
    df_sel["일평균 거래량"]   = df_sel["일평균 거래량"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(df_sel, use_container_width=True)

    tickers = [s["ticker"] for s in selected]

    # ── Phase 2: 백테스트 ─────────────────────────────────────────────────
    st.markdown("#### 📊 Phase 2 — 백테스트 실행")
    bt_ph = st.empty()
    bt_ph.caption(f"{len(tickers)}개 종목 · {start_date} ~ {end_date}")

    log_buf = io.StringIO()
    engine  = BacktestEngine(
        tickers             = tickers,
        initial_capital     = initial_capital,
        start_date          = start_date,
        end_date            = end_date,
        deposit_schedule    = deposit_schedule,
        position_sizing_pct = position_pct,
    )

    with st.spinner("백테스트 실행 중... (다운로드 포함, 최대 수 분 소요)"):
        with contextlib.redirect_stdout(log_buf):
            engine.run()

    bt_ph.success("✅ 백테스트 완료")

    # 벤치마크 수집
    bm_df = _fetch_benchmark(benchmark_ticker, engine.equity_curve)

    result = dict(
        equity_curve     = engine.equity_curve,
        trade_log        = engine.trade_log,
        log_text         = log_buf.getvalue(),
        bm_df            = bm_df,
        bm_label         = benchmark_label,
        selected_stocks  = selected,
        deposit_schedule = deposit_schedule,
    )
    st.session_state["bt_result"] = result
    _display_results(**result)


def _fetch_benchmark(ticker: str | None, equity_curve: list[dict]) -> pd.DataFrame | None:
    if not ticker or not equity_curve:
        return None
    try:
        raw = yf.download(
            ticker,
            start=equity_curve[0]["date"],
            end=equity_curve[-1]["date"],
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            return None
        close    = raw["Close"].dropna()
        first_p  = float(close.iloc[0])
        return pd.DataFrame({
            "date":       [d.strftime("%Y-%m-%d") for d in close.index],
            "return_pct": [(float(p) / first_p - 1) * 100 for p in close],
        })
    except Exception:
        return None


# ── 결과 렌더링 ─────────────────────────────────────────────────────────────────

def _display_results(
    equity_curve, trade_log, log_text,
    bm_df, bm_label, selected_stocks, deposit_schedule, **_,
):
    if not equity_curve:
        st.warning("결과 데이터가 없습니다.")
        return

    ec, last = equity_curve, equity_curve[-1]
    first_invested = ec[0]["invested"]
    final_asset    = last["total_asset"]
    final_return   = last["return_pct"]
    invested       = last["invested"]
    cash           = last["cash"]

    # MDD
    assets = [r["total_asset"] for r in ec]
    peak, mdd = assets[0], 0.0
    for a in assets:
        peak = max(peak, a)
        mdd  = max(mdd, (peak - a) / peak * 100)

    # CAGR
    n_years = (pd.Timestamp(ec[-1]["date"]) - pd.Timestamp(ec[0]["date"])).days / 365.25
    cagr = ((final_asset / first_invested) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0

    # 승률
    sell_tp  = [t for t in trade_log if t["action"] == "SELL_TP"]
    sell_sl  = [t for t in trade_log if t["action"] == "SELL_SL"]
    buys     = [t for t in trade_log if "BUY" in t["action"]]
    win_rate = len(sell_tp) / (len(sell_tp) + len(sell_sl)) * 100 if (sell_tp or sell_sl) else 0.0

    st.divider()
    st.markdown(
        f"#### 결과 요약  "
        f"<span style='color:#8B949E;font-size:0.85rem;font-weight:400;'>"
        f"{ec[0]['date']} ~ {ec[-1]['date']}  ({n_years:.1f}년)  "
        f"/ 종목 {len(selected_stocks)}개</span>",
        unsafe_allow_html=True,
    )

    # 지표 카드
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("펀드 수익률",    f"{final_return:+.2f}%")
    m2.metric("CAGR",           f"{cagr:+.2f}%")
    m3.metric("최대 낙폭(MDD)", f"{-mdd:.2f}%")
    m4.metric("승률",           f"{win_rate:.1f}%")
    m5.metric("총 매수 횟수",   f"{len(buys)}회")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("초기 투자",  f"{first_invested:,.0f}원")
    c2.metric("추가 입금",  f"{invested - first_invested:,.0f}원")
    c3.metric("최종 자산",  f"{final_asset:,.0f}원")
    c4.metric("현금 잔고",  f"{cash:,.0f}원")

    # 에쿼티 커브
    st.markdown("#### 수익률 추이")
    ec_df = pd.DataFrame(ec)
    ec_df["date"] = pd.to_datetime(ec_df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ec_df["date"], y=ec_df["return_pct"],
        mode="lines", name="펀드 수익률",
        line=dict(color="#4FC3F7", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>수익률: <b>%{y:.2f}%</b><extra></extra>",
    ))

    if bm_df is not None and not bm_df.empty:
        bm_pd = pd.DataFrame(bm_df)
        bm_pd["date"] = pd.to_datetime(bm_pd["date"])
        fig.add_trace(go.Scatter(
            x=bm_pd["date"], y=bm_pd["return_pct"],
            mode="lines", name=bm_label,
            line=dict(color="#FF8A65", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>" + bm_label + ": <b>%{y:.2f}%</b><extra></extra>",
        ))

    for dep_date in deposit_schedule:
        try:
            dep_ts = pd.Timestamp(dep_date)
            row = ec_df[ec_df["date"] >= dep_ts].head(1)
            if not row.empty:
                fig.add_vline(x=dep_ts.timestamp() * 1000,
                              line=dict(color="#66BB6A", width=1, dash="dash"))
                fig.add_annotation(
                    x=dep_ts, y=float(row["return_pct"].iloc[0]),
                    text="입금", showarrow=True, arrowhead=2,
                    arrowcolor="#66BB6A", font=dict(size=10, color="#66BB6A"),
                    ax=0, ay=-30,
                )
        except Exception:
            pass

    fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dot"))
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)",
        xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
        yaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E"), ticksuffix="%"),
        legend=dict(font=dict(color="#C9D1D9"), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 총자산 추이
    with st.expander("📈 총 자산 규모 추이", expanded=False):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ec_df["date"], y=ec_df["total_asset"],
            mode="lines", name="총 자산",
            line=dict(color="#4FC3F7", width=2),
            fill="tozeroy", fillcolor="rgba(79,195,247,0.1)",
            hovertemplate="%{x|%Y-%m-%d}<br>총 자산: <b>%{y:,.0f}원</b><extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=ec_df["date"], y=ec_df["invested"],
            mode="lines", name="투자 원금",
            line=dict(color="#66BB6A", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>원금: <b>%{y:,.0f}원</b><extra></extra>",
        ))
        fig2.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,27,34,0.8)",
            xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
            yaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E"), tickformat=",.0f"),
            legend=dict(font=dict(color="#C9D1D9"), bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 선정 종목 요약
    with st.expander(f"📋 선정 종목 ({len(selected_stocks)}개)", expanded=False):
        df_sel = pd.DataFrame(selected_stocks)[
            ["ticker", "name", "momentum", "volatility", "avg_volume"]
        ]
        df_sel.index = range(1, len(df_sel) + 1)
        df_sel.columns = ["티커", "종목명", "3개월 수익률(%)", "변동성(%)", "일평균 거래량"]
        df_sel["3개월 수익률(%)"] = df_sel["3개월 수익률(%)"].apply(lambda x: f"{x:+.1f}%")
        df_sel["변동성(%)"]       = df_sel["변동성(%)"].apply(lambda x: f"{x:.2f}%")
        df_sel["일평균 거래량"]   = df_sel["일평균 거래량"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_sel, use_container_width=True)

    # 거래 내역
    with st.expander(f"📋 거래 내역 ({len(trade_log)}건)", expanded=False):
        if trade_log:
            df_tr = pd.DataFrame(trade_log)
            df_tr["amount"] = df_tr["amount"].apply(lambda x: f"{x:,.0f}원")
            df_tr["price"]  = df_tr["price"].apply(lambda x: f"{x:,.0f}원")
            icon_map = {"BUY": "🛒", "SELL_TP": "💵✅", "SELL_SL": "💵❌"}
            df_tr["action"] = df_tr["action"].apply(
                lambda a: f"{icon_map.get(a, '↩️')} {a}"
            )
            df_tr.columns = ["날짜", "종목", "액션", "체결가", "수량", "거래금액", "사유"]
            st.dataframe(df_tr, use_container_width=True, hide_index=True)
        else:
            st.info("거래 내역이 없습니다.")

    # 실행 로그
    with st.expander("🖥️ 실행 로그", expanded=False):
        st.code(log_text, language=None)
