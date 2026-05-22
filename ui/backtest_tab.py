"""
ui/backtest_tab.py — 백테스트 탭 렌더링
"""
from __future__ import annotations

import contextlib
import io
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


def render_backtest_tab(tab) -> None:
    with tab:
        st.subheader("🔬 백테스트")
        st.caption(
            "TradingStrategy(SMA 골든크로스 + ATR 익절·손절 + 분할매수)를 과거 일봉 데이터에 적용해 "
            "수익률을 검증합니다."
        )

        # ── 설정 폼 ────────────────────────────────────────────────────────
        with st.form("backtest_form"):
            col_l, col_r = st.columns([1, 1], gap="large")

            with col_l:
                st.markdown("#### 종목 & 기간")
                tickers_raw = st.text_input(
                    "종목 코드 (쉼표 구분)",
                    value="005930.KS, 000660.KS",
                    help="예: 005930.KS, 000660.KS, AAPL",
                )
                c1, c2 = st.columns(2)
                start_date = c1.date_input("시작일", value=date(2020, 1, 1))
                end_date   = c2.date_input("종료일", value=date(2024, 12, 31))

                benchmark_ticker = st.selectbox(
                    "벤치마크 지수",
                    options=["^KS11 (KOSPI)", "^KQ11 (KOSDAQ)", "^IXIC (NASDAQ)", "^GSPC (S&P500)", "없음"],
                    index=0,
                )

            with col_r:
                st.markdown("#### 자본 설정")
                initial_capital = st.number_input(
                    "초기 자본금 (원)",
                    min_value=1_000_000,
                    max_value=10_000_000_000,
                    value=10_000_000,
                    step=1_000_000,
                    format="%d",
                )
                position_pct = st.slider(
                    "1회 매수 비중 (총자산 대비 %)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="총자산의 몇 %를 1회 매수에 사용할지 설정합니다.",
                )
                st.markdown("#### 추가 입금 일정")
                deposit_text = st.text_area(
                    "날짜,금액 (한 줄에 하나씩)",
                    value="2021-01-04,5000000\n2022-01-03,5000000\n2023-01-02,5000000",
                    height=110,
                    help="형식: YYYY-MM-DD,금액\n예: 2021-01-04,5000000",
                )

            submitted = st.form_submit_button(
                "▶ 백테스트 실행",
                use_container_width=True,
                type="primary",
            )

        # ── 실행 ──────────────────────────────────────────────────────────
        if submitted:
            tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
            if not tickers:
                st.error("종목 코드를 입력하세요.")
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

            bm_map = {
                "^KS11 (KOSPI)":  "^KS11",
                "^KQ11 (KOSDAQ)": "^KQ11",
                "^IXIC (NASDAQ)": "^IXIC",
                "^GSPC (S&P500)": "^GSPC",
                "없음": None,
            }
            bm_ticker = bm_map.get(benchmark_ticker)

            _run_and_display(
                tickers=tickers,
                initial_capital=float(initial_capital),
                start_date=str(start_date),
                end_date=str(end_date),
                deposit_schedule=deposit_schedule,
                position_sizing_pct=position_pct / 100,
                benchmark_ticker=bm_ticker,
            )

        # ── 이전 결과 복원 ─────────────────────────────────────────────────
        elif "bt_result" in st.session_state:
            _display_results(**st.session_state["bt_result"])


# ── 백테스트 실행 & 저장 ────────────────────────────────────────────────────────

def _run_and_display(
    tickers, initial_capital, start_date, end_date,
    deposit_schedule, position_sizing_pct, benchmark_ticker,
):
    from backtest import BacktestEngine

    log_buf = io.StringIO()
    engine  = BacktestEngine(
        tickers             = tickers,
        initial_capital     = initial_capital,
        start_date          = start_date,
        end_date            = end_date,
        deposit_schedule    = deposit_schedule,
        position_sizing_pct = position_sizing_pct,
    )

    with st.spinner("백테스트 실행 중... (수년치 데이터 다운로드 포함, 최대 1분 소요)"):
        with contextlib.redirect_stdout(log_buf):
            engine.run()

    equity_curve = engine.equity_curve
    trade_log    = engine.trade_log
    log_text     = log_buf.getvalue()

    # 벤치마크 데이터 수집
    bm_df = None
    if benchmark_ticker and equity_curve:
        try:
            raw_bm = yf.download(
                benchmark_ticker,
                start=equity_curve[0]["date"],
                end=equity_curve[-1]["date"],
                auto_adjust=True,
                progress=False,
            )
            if not raw_bm.empty:
                bm_close = raw_bm["Close"].dropna()
                first_p  = float(bm_close.iloc[0])
                bm_df = pd.DataFrame({
                    "date":       [d.strftime("%Y-%m-%d") for d in bm_close.index],
                    "return_pct": [(float(p) / first_p - 1) * 100 for p in bm_close],
                })
        except Exception:
            bm_df = None

    result = dict(
        equity_curve    = equity_curve,
        trade_log       = trade_log,
        log_text        = log_text,
        bm_df           = bm_df,
        bm_label        = benchmark_ticker or "",
        tickers         = tickers,
        deposit_schedule= deposit_schedule,
    )
    st.session_state["bt_result"] = result
    _display_results(**result)


# ── 결과 렌더링 ─────────────────────────────────────────────────────────────────

def _display_results(
    equity_curve, trade_log, log_text, bm_df, bm_label,
    tickers, deposit_schedule, **_,
):
    if not equity_curve:
        st.warning("결과 데이터가 없습니다.")
        return

    ec   = equity_curve
    last = ec[-1]
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
    st.markdown(f"#### 결과 요약  <span style='color:#8B949E;font-size:0.85rem;font-weight:400;'>{ec[0]['date']} ~ {ec[-1]['date']}  ({n_years:.1f}년)</span>", unsafe_allow_html=True)

    # ── 지표 카드 ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("펀드 수익률",  f"{final_return:+.2f}%")
    m2.metric("CAGR",          f"{cagr:+.2f}%")
    m3.metric("최대 낙폭 (MDD)", f"{-mdd:.2f}%")
    m4.metric("승률",           f"{win_rate:.1f}%")
    m5.metric("총 매수 횟수",   f"{len(buys)}회")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("초기 투자",      f"{first_invested:,.0f}원")
    c2.metric("추가 입금",      f"{invested - first_invested:,.0f}원")
    c3.metric("최종 자산",      f"{final_asset:,.0f}원")
    c4.metric("현금 잔고",      f"{cash:,.0f}원")

    # ── 에쿼티 커브 차트 ──────────────────────────────────────────────
    st.markdown("#### 수익률 추이")
    ec_df  = pd.DataFrame(ec)
    ec_df["date"] = pd.to_datetime(ec_df["date"])

    fig = go.Figure()

    # 펀드 수익률 선
    fig.add_trace(go.Scatter(
        x=ec_df["date"],
        y=ec_df["return_pct"],
        mode="lines",
        name="펀드 수익률",
        line=dict(color="#4FC3F7", width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>수익률: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # 벤치마크 선
    if bm_df is not None and not bm_df.empty:
        bm_pd = pd.DataFrame(bm_df)
        bm_pd["date"] = pd.to_datetime(bm_pd["date"])
        fig.add_trace(go.Scatter(
            x=bm_pd["date"],
            y=bm_pd["return_pct"],
            mode="lines",
            name=bm_label,
            line=dict(color="#FF8A65", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>" + bm_label + ": <b>%{y:.2f}%</b><extra></extra>",
        ))

    # 추가 입금 시점 세로선
    for dep_date in deposit_schedule:
        try:
            dep_ts = pd.Timestamp(dep_date)
            # 입금 시점 수익률 찾기
            ec_row = ec_df[ec_df["date"] >= dep_ts].head(1)
            if not ec_row.empty:
                y_val = float(ec_row["return_pct"].iloc[0])
                fig.add_vline(
                    x=dep_ts.timestamp() * 1000,
                    line=dict(color="#66BB6A", width=1, dash="dash"),
                )
                fig.add_annotation(
                    x=dep_ts,
                    y=y_val,
                    text="입금",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#66BB6A",
                    font=dict(size=10, color="#66BB6A"),
                    ax=0, ay=-30,
                )
        except Exception:
            pass

    fig.add_hline(y=0, line=dict(color="#666", width=1, dash="dot"))
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.8)",
        xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
        yaxis=dict(
            gridcolor="#21262D",
            tickfont=dict(color="#8B949E"),
            ticksuffix="%",
        ),
        legend=dict(
            font=dict(color="#C9D1D9"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 자산 규모 차트 ────────────────────────────────────────────────
    with st.expander("📈 총 자산 규모 추이", expanded=False):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ec_df["date"],
            y=ec_df["total_asset"],
            mode="lines",
            name="총 자산",
            line=dict(color="#4FC3F7", width=2),
            fill="tozeroy",
            fillcolor="rgba(79,195,247,0.1)",
            hovertemplate="%{x|%Y-%m-%d}<br>총 자산: <b>%{y:,.0f}원</b><extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=ec_df["date"],
            y=ec_df["invested"],
            mode="lines",
            name="투자 원금",
            line=dict(color="#66BB6A", width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>원금: <b>%{y:,.0f}원</b><extra></extra>",
        ))
        fig2.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,27,34,0.8)",
            xaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E")),
            yaxis=dict(gridcolor="#21262D", tickfont=dict(color="#8B949E"), tickformat=",.0f"),
            legend=dict(font=dict(color="#C9D1D9"), bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── 거래 내역 ─────────────────────────────────────────────────────
    with st.expander(f"📋 거래 내역  ({len(trade_log)}건)", expanded=False):
        if trade_log:
            df_trades = pd.DataFrame(trade_log)
            df_trades["amount"] = df_trades["amount"].apply(lambda x: f"{x:,.0f}원")
            df_trades["price"]  = df_trades["price"].apply(lambda x: f"{x:,.0f}원")
            action_icon = {
                "BUY": "🛒", "SELL_TP": "💵✅", "SELL_SL": "💵❌",
                "REBUY#1": "↩️", "REBUY#2": "↩️", "REBUY#3": "↩️",
            }
            df_trades["action"] = df_trades["action"].apply(
                lambda a: f"{action_icon.get(a, '')} {a}"
            )
            df_trades.columns = ["날짜", "종목", "액션", "체결가", "수량", "거래금액", "사유"]
            st.dataframe(df_trades, use_container_width=True, hide_index=True)
        else:
            st.info("거래 내역이 없습니다.")

    # ── 실행 로그 ─────────────────────────────────────────────────────
    with st.expander("🖥️ 실행 로그", expanded=False):
        st.code(log_text, language=None)
