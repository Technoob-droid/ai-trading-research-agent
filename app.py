import os
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from dotenv import load_dotenv
from src.graph import build_research_graph

load_dotenv()
st.set_page_config(page_title="AI Trading Research Agent", layout="wide")

st.title("AI Trading Research Agent")
st.caption("Autonomous LangGraph Equity Research Pipeline (Groq + Tavily + yfinance)")

if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    st.error("Missing GROQ_API_KEY or TAVILY_API_KEY in .env or Streamlit Secrets.")
    st.stop()

tab1, tab2 = st.tabs(["Single-Stock Deep Dive", "Portfolio Batch Screener"])

with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    ticker = c1.text_input("Stock Ticker (e.g., NVDA, TCS.NS)", value="NVDA").strip().upper()
    period = c2.selectbox("Chart Period", ["3mo", "6mo", "1y"], index=2)
    c3.write("")
    c3.write("")
    run_single = c3.button("Run Analysis", type="primary", use_container_width=True)
    if run_single and ticker:
        with st.spinner(f"Running 5-node research pipeline for {ticker}..."):
            result = build_research_graph().invoke({"ticker": ticker})
        m = result["market_data"]
        t = result["technical_indicators"]
        st.subheader(f"{m.get('name', ticker)} ({ticker}) -- [{t.get('trend_signal', 'N/A')}]")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Price", f"${m.get('current_price')}")
        k2.metric("1M Return", f"{m.get('1m_return_pct')}%")
        k3.metric("Trailing P/E", f"{m.get('trailing_pe')}")
        k4.metric("RSI (14)", f"{t.get('rsi_14')}")
        k5.metric("MACD", f"{t.get('macd')}")
        hist = yf.Ticker(ticker).history(period=period)
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price")])
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="SMA 50", line=dict(color="orange", width=1.5)))
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(), name="SMA 200", line=dict(color="cyan", width=1.5)))
            fig.update_layout(xaxis_rangeslider_visible=False, height=430, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        bull_col, bear_col = st.columns(2)
        with bull_col:
            st.markdown("### Bull Case")
            st.markdown(result["bull_case"])
        with bear_col:
            st.markdown("### Bear Case & Risks")
            st.markdown(result["bear_case"])
        st.expander("Recent News Catalysts").markdown(result["news_summary"])
        st.download_button("Download Research Memo (.md)", data=result["final_report"], file_name=f"{ticker}_research_report.md")

with tab2:
    raw = st.text_input("Enter comma-separated tickers", value="NVDA, AAPL, TCS.NS")
    run_batch = st.button("Screen Portfolio", type="primary")
    if run_batch and raw:
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        rows, reports = [], {}
        prog = st.progress(0)
        app = build_research_graph()
        for i, tk in enumerate(tickers):
            with st.spinner(f"Analyzing {tk} ({i+1}/{len(tickers)})..."):
                try:
                    res = app.invoke({"ticker": tk})
                    m = res["market_data"]
                    t = res["technical_indicators"]
                    rows.append({"Ticker": tk, "Company": m.get("name", "N/A"), "Price": m.get("current_price", "N/A"), "1M Return %": float(m.get("1m_return_pct", 0) or 0), "Trailing P/E": m.get("trailing_pe", "N/A"), "RSI (14)": t.get("rsi_14", "N/A"), "Trend": t.get("trend_signal", "N/A")})
                    reports[tk] = res["final_report"]
                except Exception as e:
                    st.warning(f"Skipped {tk}: {e}")
            prog.progress((i + 1) / len(tickers))
        if rows:
            df = pd.DataFrame(rows).sort_values(by="1M Return %", ascending=False).reset_index(drop=True)
            df.index = df.index + 1
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Screener Table (.csv)", data=df.to_csv(), file_name="portfolio_screener.csv")
            for tk, md in reports.items():
                with st.expander(f"Full Research Memo: {tk}"):
                    st.markdown(md)
