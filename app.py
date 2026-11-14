import os
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from dotenv import load_dotenv
from src.graph import build_research_graph

load_dotenv()
st.set_page_config(page_title="AI Trading Research Agent", layout="wide")

st.title("AI Trading Research Agent")
st.caption("Autonomous LangGraph Equity Research Pipeline (Groq + Tavily + yfinance)")

with st.sidebar:
    st.header("Research Config")
    ticker = st.text_input("Stock Ticker", value="NVDA").strip().upper()
    period = st.selectbox("Chart Period", ["3mo", "6mo", "1y"], index=2)
    run_btn = st.button("Run Research Agent", type="primary", use_container_width=True)

if run_btn and ticker:
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Missing GROQ_API_KEY or TAVILY_API_KEY in .env")
        st.stop()
    with st.spinner(f"Running 5-node research pipeline for {ticker}..."):
        result = build_research_graph().invoke({"ticker": ticker})
    m = result["data"] if "data" in result else result["market_data"]
    t = result["technical_indicators"]
    st.subheader(f"{m.get('name', ticker)} ({ticker}) -- [{t.get('trend_signal', 'N/A')}]")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"${m.get('current_price')}")
    c2.metric("1M Return", f"{m.get('1m_return_pct')}%")
    c3.metric("Trailing P/E", f"{m.get('trailing_pe')}")
    c4.metric("RSI (14)", f"{t.get('rsi_14')}")
    c5.metric("MACD", f"{t.get('macd')}")
    hist = yf.Ticker(ticker).history(period=period)
    if not hist.empty:
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price")])
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="SMA 50", line=dict(color="orange", width=1.5)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(), name="SMA 200", line=dict(color="cyan", width=1.5)))
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Bull Case")
        st.markdown(result["bull_case"])
    with col2:
        st.markdown("### Bear Case & Risks")
        st.markdown(result["bear_case"])
    st.expander("Recent News Catalysts").markdown(result["news_summary"])
    st.download_button("Download Research Memo (.md)", data=result["final_report"], file_name=f"{ticker}_research_report.md")
