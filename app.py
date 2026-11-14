import os, time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from src.graph import build_research_graph
from src.tools import fetch_intraday_pulse

load_dotenv()
st.set_page_config(page_title='AI Trading Terminal', layout='wide')
st.title('AI Trading & Equity Research Terminal')
st.caption('Live Intraday Pulse | Screener-Grade Fundamentals | LangGraph AI Analyst')

if not os.getenv('GROQ_API_KEY') or not os.getenv('TAVILY_API_KEY'):
    st.error('Missing GROQ_API_KEY or TAVILY_API_KEY in .env or Streamlit Secrets.')
    st.stop()

tab1, tab2, tab3 = st.tabs(['Live Market Pulse', 'Deep Fundamental & AI Research', 'Portfolio Batch Screener'])

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
            result = build_research_graph().invoke({'ticker': ticker})
        m = result['market_data']
        t = result['technical_indicators']
        cur = 'INR ' if ticker.endswith(('.NS', '.BO')) else '$'
        st.subheader(f"{m.get('name', ticker)} ({ticker}) -- [{t.get('trend_signal', 'N/A')}]")
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
        bull_col, bear_col = st.columns(2)
        with bull_col:
            st.markdown('### Bull Case')
            st.markdown(result['bull_case'])
        with bear_col:
            st.markdown('### Bear Case & Risks')
            st.markdown(result['bear_case'])
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
