import os, time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from src.graph import build_research_graph
from src.ledger import log_paper_trade, evaluate_ledger, close_paper_trade, import_ledger_csv, close_paper_trade, import_ledger_csv
from src.backtester import compile_strategy, run_backtest, critique_backtest, optimize_strategy
from src.tools import fetch_intraday_pulse

load_dotenv()
st.set_page_config(page_title='AI Trading Terminal', layout='wide')
st.title('AI Trading & Equity Research Terminal')
st.caption('Live Intraday Pulse | Screener-Grade Fundamentals | LangGraph AI Analyst')

if not os.getenv('GROQ_API_KEY') or not os.getenv('TAVILY_API_KEY'):
    st.error('Missing GROQ_API_KEY or TAVILY_API_KEY in .env or Streamlit Secrets.')
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Live Market Pulse', 'Deep Fundamental & AI Research', 'Portfolio Batch Screener', "AI Strategy Lab (Backtester)", "Paper-Trading Ledger"])

with tab1:
    a1, a2, a3, a4 = st.columns([2, 1, 1, 1])
    live_tk = a1.text_input('Live Ticker (e.g., NVDA, TCS.NS, RELIANCE.BO)', value='NVDA').strip().upper()
    interval = a2.selectbox('Intraday Interval', ['1m', '5m', '15m', '1h'], index=1)
    a3.write('')
    auto_refresh = a3.checkbox('Live Auto-Refresh (30s)', value=False)
    a4.write('')
    a4.button('Refresh Now', width='stretch')
    if live_tk:
        df, pulse = fetch_intraday_pulse(live_tk, interval)
        if df is not None:
            cur = 'INR ' if live_tk.endswith(('.NS', '.BO')) else '$'
            st.caption(f"Last Tick: {pulse['timestamp']} (Live Intraday Feed)")
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric('Live Price', f"{cur}{pulse['last_price']}", f"{pulse['change_pct']}%")
            p2.metric('Intraday VWAP', f"{cur}{pulse['vwap']}")
            p3.metric('Volume Surge', f"{pulse['vol_surge']}x")
            p4.metric('Intraday RSI (14)', f"{pulse['rsi']}")
            bb_w = round(float(df['BB_Upper'].iloc[-1] - df['BB_Lower'].iloc[-1]), 2) if len(df) > 20 else 'N/A'
            p5.metric('Bollinger Width', f'{cur}{bb_w}')
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='cyan', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='#4b96ff'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='orange', width=1.5)), row=3, col=1)
            fig.add_hline(y=70, line_dash='dash', line_color='red', row=3, col=1)
            fig.add_hline(y=30, line_dash='dash', line_color='green', row=3, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False, height=650, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning('No intraday data returned for this ticker.')

with tab2:
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input('Stock Ticker (e.g., NVDA, TCS.NS)', value='NVDA').strip().upper()
    c2.write('')
    run_single = c2.button('Run AI Analysis', type='primary', width='stretch')
    if run_single and ticker:
        with st.spinner(f'Running 5-node research pipeline for {ticker}...'):
            st.session_state["tab2_result"] = build_research_graph().invoke({'ticker': ticker})
    if "tab2_result" in st.session_state:
        result = st.session_state["tab2_result"]
        m = result['market_data']
        t = result['technical_indicators']
        cur = 'INR ' if ticker.endswith(('.NS', '.BO')) else '$'
        st.subheader(f"{m.get('name', ticker)} ({result.get("ticker", ticker)}) -- [{t.get('trend_signal', 'N/A')}]")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric('Price', f"{cur}{m.get('current_price')}")
        k2.metric('Trailing P/E', f"{m.get('trailing_pe')}")
        k3.metric('ROE %', f"{m.get('roe')}%")
        k4.metric('Profit Margin %', f"{m.get('profit_margin')}%")
        k5.metric('Debt / Equity', f"{m.get('debt_to_equity')}")
        fund_df = pd.DataFrame([
            {'Metric': 'Industry', 'Value': str(m.get('industry')), 'Metric 2': 'Price to Book', 'Value 2': str(m.get('price_to_book')), 'Metric 3': 'ROA %', 'Value 3': str(m.get('roa'))},
            {'Metric': 'Revenue Growth %', 'Value': str(m.get('revenue_growth')), 'Metric 2': 'RSI (14)', 'Value 2': str(t.get('rsi_14')), 'Metric 3': 'MACD / Signal', 'Value 3': f"{t.get('macd')} / {t.get('macd_signal')}"}
        ])
        st.table(fund_df)
        mtf = result.get("mtf_confluence", {})
        if mtf:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Weekly Macro", mtf.get("weekly", {}).get("bias", "N/A"), f"RSI {mtf.get('weekly', {}).get('rsi', 'N/A')}")
            m2.metric("Daily Swing", mtf.get("daily", {}).get("bias", "N/A"), f"RSI {mtf.get('daily', {}).get('rsi', 'N/A')}")
            m3.metric("1H Execution", mtf.get("hourly", {}).get("bias", "N/A"), f"RSI {mtf.get('hourly', {}).get('rsi', 'N/A')}")
            m4.metric("MTF Confluence", f"{mtf.get('confluence_score', 0)} / 3", mtf.get("verdict", "N/A"))
        tp = result.get("trade_plan", {})
        st.success(f"PM VERDICT: {tp.get('action', 'N/A')}  --  Conviction: {tp.get('conviction_score', 0)}/100  --  Allocation: {tp.get('position_size_pct', 0)}%")
        st.info(f"PM Rationale: {tp.get('rationale', '')}")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Entry Zone", str(tp.get("entry_zone", "N/A")))
        t2.metric("Stop-Loss", str(tp.get("stop_loss", "N/A")))
        t3.metric("Target Price", str(tp.get("target_price", "N/A")))
        t4.metric("Risk / Reward", str(tp.get("risk_reward_ratio", "N/A")))
        t5.metric("ATR (14-Day)", f"{cur}{t.get('atr_14', 'N/A')}")
        st.button("Log Paper Trade to Ledger", key=f"log_{result.get('ticker')}", on_click=lambda t=result.get("ticker", ""), p=tp: (lambda ok: st.session_state.update({"last_logged": t if ok else None, "log_err": None if ok else t, "ledger_df": evaluate_ledger()}))(log_paper_trade(t, p)))
        if st.session_state.get("last_logged") == result.get("ticker"):
            st.success(f"Logged {result.get('ticker')} to Paper-Trading Ledger (Tab 5)!")
        elif st.session_state.get("log_err") == result.get("ticker"):
            st.warning("Invalid or inverted Stop-Loss / Target levels — trade not logged.")
        bull_col, bear_col = st.columns(2)
        with bull_col:
            st.markdown('### Bull Case')
            st.markdown(result['bull_case'])
            st.markdown("**Bull Counter-Rebuttal:**")
            st.markdown(result.get('bull_rebuttal', ''))
        with bear_col:
            st.markdown('### Bear Case & Risks')
            st.markdown(result['bear_case'])
            st.markdown("**Bear Counter-Rebuttal:**")
            st.markdown(result.get('bear_rebuttal', ''))
        st.expander('Recent News Catalysts').markdown(result['news_summary'])
        st.download_button('Download Research Memo (.md)', data=result['final_report'], file_name=f'{ticker}_research_report.md')

with tab3:
    raw = st.text_input('Enter comma-separated tickers', value='NVDA, AAPL, TCS.NS')
    run_batch = st.button('Screen Portfolio', type='primary')
    if run_batch and raw:
        tickers = [t.strip().upper() for t in raw.split(',') if t.strip()]
        rows, reports = [], {}
        prog = st.progress(0)
        app = build_research_graph()
        for i, tk in enumerate(tickers):
            with st.spinner(f'Analyzing {tk} ({i+1}/{len(tickers)})...'):
                try:
                    res = app.invoke({'ticker': tk})
                    m = res['market_data']
                    t = res['technical_indicators']
                    rows.append({'Ticker': tk, 'Company': m.get('name', 'N/A'), 'Price': m.get('current_price', 'N/A'), '1M Return %': float(m.get('1m_return_pct', 0) or 0), 'ROE %': m.get('roe', 'N/A'), 'Trailing P/E': m.get('trailing_pe', 'N/A'), 'RSI (14)': t.get('rsi_14', 'N/A'), 'Trend': t.get('trend_signal', 'N/A')})
                    reports[tk] = res['final_report']
                except Exception as e:
                    st.warning(f'Skipped {tk}: {e}')
            prog.progress((i + 1) / len(tickers))
        if rows:
            df = pd.DataFrame(rows).sort_values(by='1M Return %', ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            st.dataframe(df, width='stretch')
            st.download_button('Download Screener Table (.csv)', data=df.to_csv(), file_name='portfolio_screener.csv')
            for tk, md in reports.items():
                with st.expander(f'Full Research Memo: {tk}'):
                    st.markdown(md)

if auto_refresh:
    time.sleep(30)
    st.rerun()

with tab4:
    st.subheader("AI Natural-Language Strategy Compiler & Backtester")
    bc1, bc2 = st.columns([1, 2])
    bt_ticker = bc1.text_input("Backtest Ticker", value="NVDA", key="bt_ticker").strip().upper()
    bt_period = bc1.selectbox("Historical Window", ["1y", "2y", "5y"], index=1, key="bt_period")
    bt_prompt = bc2.text_area("Describe Your Strategy in Plain English", value="Buy when RSI 14 drops below 35 with a 1.5x ATR stop-loss, 2.5x ATR target, and max 15 day hold", height=105)
    if st.button("Compile & Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Compiling strategy rules and simulating historical trades..."):
            spec = compile_strategy(bt_prompt)
            eq_df, m, trades = run_backtest(bt_ticker, spec, period=bt_period)
        if eq_df is None:
            st.error("Insufficient historical data for this ticker.")
        else:
            with st.spinner("Running AI Quant Post-Mortem Critique..."):
                critique = critique_backtest(bt_ticker, spec, m, trades)
            st.session_state["bt_state"] = {"ticker": bt_ticker, "period": bt_period, "spec": spec, "eq_df": eq_df, "metrics": m, "trades": trades, "critique": critique}
            st.session_state.pop("opt_state", None)
    if "bt_state" in st.session_state:
        bs = st.session_state["bt_state"]
        m = bs["metrics"]
        st.success(f"Baseline Strategy: {bs['spec'].name} ({bs['ticker']} - {bs['period']})")
        st.json(bs["spec"].model_dump())
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Trades", m["trades"])
        k2.metric("Win Rate", f"{m['win_rate']}%")
        k3.metric("Profit Factor", m["profit_factor"])
        k4.metric("Strategy Return", f"{m['strat_ret']}%", delta=f"{round(m['strat_ret'] - m['bh_ret'], 2)}% vs B&H")
        k5.metric("Max Drawdown", f"{m['max_dd']}%")
        st.line_chart(bs["eq_df"][["Strategy", "BuyHold"]])
        st.info(bs["critique"])
        if st.button("1-Click AI Auto-Optimize Strategy", use_container_width=True):
            with st.spinner("AI Quant Optimizer tuning rules based on post-mortem..."):
                opt_spec = optimize_strategy(bs["ticker"], bs["spec"], m, bs["trades"], bs["critique"])
                opt_eq, opt_m, opt_trades = run_backtest(bs["ticker"], opt_spec, period=bs["period"])
                st.session_state["opt_state"] = {"spec": opt_spec, "eq_df": opt_eq, "metrics": opt_m, "trades": opt_trades}
        if "opt_state" in st.session_state:
            os_res = st.session_state["opt_state"]
            om = os_res["metrics"]
            st.divider()
            st.success(f"AI-Optimized Strategy: {os_res['spec'].name}")
            st.json(os_res["spec"].model_dump())
            o1, o2, o3, o4, o5 = st.columns(5)
            o1.metric("Trades", om["trades"], delta=om["trades"] - m["trades"])
            o2.metric("Win Rate", f"{om['win_rate']}%", delta=f"{round(om['win_rate'] - m['win_rate'], 1)}%")
            o3.metric("Profit Factor", om["profit_factor"], delta=round(om["profit_factor"] - m["profit_factor"], 2))
            o4.metric("Strategy Return", f"{om['strat_ret']}%", delta=f"{round(om['strat_ret'] - m['strat_ret'], 2)}% vs Baseline")
            o5.metric("Max Drawdown", f"{om['max_dd']}%", delta=f"{round(om['max_dd'] - m['max_dd'], 2)}%")
            comp_df = bs["eq_df"][["Strategy", "BuyHold"]].rename(columns={"Strategy": "Baseline"})
            comp_df["AI_Optimized"] = os_res["eq_df"]["Strategy"]
            st.line_chart(comp_df)
            if os_res["trades"]:
                st.dataframe(pd.DataFrame(os_res["trades"]), use_container_width=True)
        elif bs["trades"]:
            st.dataframe(pd.DataFrame(bs["trades"]), use_container_width=True)

with tab5:
    st.subheader("PM Paper-Trading Ledger & Closed-Loop Track Record")
    b1, b2, b3 = st.columns([1.2, 1, 1.5])
    if b1.button("Refresh Live Prices & Evaluate", type="primary", use_container_width=True):
        st.session_state["ledger_df"] = evaluate_ledger()
    ldf = st.session_state.get("ledger_df", evaluate_ledger())
    if not ldf.empty:
        b2.download_button("Download Ledger CSV", data=ldf.to_csv(index=False), file_name="pm_paper_trades.csv", mime="text/csv", use_container_width=True)
    up_csv = b3.file_uploader("Restore / Merge Ledger CSV", type=["csv"], label_visibility="collapsed")
    if up_csv is not None:
        cnt = import_ledger_csv(up_csv)
        if cnt:
            st.session_state["ledger_df"] = evaluate_ledger()
            st.success(f"Restored {cnt} trades into Paper-Trading Ledger!")
    if ldf.empty:
        st.info("No paper trades logged yet. Run a deep analysis in Tab 2 and click Log Paper Trade to Ledger.")
    else:
        closed = ldf[ldf["Status"].isin(["TARGET HIT", "STOPPED OUT", "CLOSED MANUAL"])]
        wins = int((closed["R_Multiple"] > 0).sum()) if not closed.empty else 0
        wr = round(wins / len(closed) * 100, 1) if not closed.empty else 0.0
        avg_r = round(float(ldf["R_Multiple"].mean()), 2)
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Total Logged Trades", len(ldf))
        l2.metric("Open Positions", int((ldf["Status"] == "OPEN").sum()))
        l3.metric("Closed Win Rate", f"{wr}%")
        l4.metric("Avg R-Multiple", f"{avg_r}R")
        st.dataframe(ldf, use_container_width=True)
        open_idx = ldf[ldf["Status"] == "OPEN"].index.tolist()
        if open_idx:
            mc1, mc2 = st.columns([2, 1])
            sel_idx = mc1.selectbox("Select Open Position to Close at Market", open_idx, format_func=lambda i: f"#{i} - {ldf.at[i, 'Ticker']} ({ldf.at[i, 'Action']} @ {ldf.at[i, 'Entry']} - R: {ldf.at[i, 'R_Multiple']}R)")
            if mc2.button("Close Selected Position", use_container_width=True):
                if close_paper_trade(sel_idx):
                    st.session_state["ledger_df"] = evaluate_ledger()
                    st.rerun()
