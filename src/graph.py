from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from src.state import ResearchState
from src.tools import fetch_market_metrics, fetch_technical_indicators, fetch_latest_news

def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

def market_data_node(state: ResearchState):
    print(f"[1/5] Fetching market metrics for {state['ticker']}...")
    return {"market_data": fetch_market_metrics(state["ticker"])}

def technicals_node(state: ResearchState):
    print(f"[2/5] Calculating technical indicators for {state['ticker']}...")
    return {"technical_indicators": fetch_technical_indicators(state["ticker"])}

def news_research_node(state: ResearchState):
    print(f"[3/5] Searching recent financial news for {state['ticker']}...")
    company_name = state["market_data"].get("name", state["ticker"])
    return {"news_summary": fetch_latest_news(state["ticker"], company_name)}

def thesis_generation_node(state: ResearchState):
    print("[4/5] Generating Bull and Bear cases...")
    llm = get_llm()
    context = f"Metrics: {state['market_data']}\nTechnicals: {state['technical_indicators']}\n\nNews:\n{state['news_summary']}"
    bull = llm.invoke([SystemMessage(content="Write a concise, data-backed Bull Case (3 bullets) incorporating fundamentals, technicals, and news."), HumanMessage(content=context)])
    bear = llm.invoke([SystemMessage(content="Write a concise, data-backed Bear Case and risks (3 bullets) incorporating fundamentals, technicals, and news."), HumanMessage(content=context)])
    return {"bull_case": bull.content, "bear_case": bear.content}

def report_compiler_node(state: ResearchState):
    print("[5/5] Compiling final research memo...")
    m = state["market_data"]
    cur = "₹" if state["ticker"].endswith((".NS", ".BO")) else "$"
    t = state["technical_indicators"]
    report = f"""# Equity Research Memo: {m.get('name')} ({state['ticker']})

## 1. Market Snapshot
- **Sector:** {m.get('sector')}
- +*Current Price:** {cur}{m.get('current_price')}
- **1-Month Return:** {m.get('1m_return_pct')}%
- **Trailing / Forward P/E:** {m.get('trailing_pe')} / {m.get('forward_pe')}
- **52-Week Range:** {cur}{m.get('52w_low')} - {cur}{m.get('52w_high')}

## 2. Technical Indicators
- +*RSI (14-day):** {t.get('rsi_14')}
- **MACD / Signal:** {t.get('macd')} / {t.get('macd_signal')}
- **SMA 50 / 200:** {t.get('sma_50')} / {t.get('sma_200')}
- +*Trend Signal:** {t.get('trend_signal')}

## 3. Bull Case
{state['bull_case']}

## 4. Bear Case & Risk Factors
{state['bear_case']}

## 5. Recent News Catalysts
{state['news_summary']}
"""
    return {"final_report": report}

def build_research_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("market_data", market_data_node)
    workflow.add_node("technicals", technicals_node)
    workflow.add_node("news_research", news_research_node)
    workflow.add_node("thesis_gen", thesis_generation_node)
    workflow.add_node("compile_report", report_compiler_node)
    workflow.set_entry_point("market_data")
    workflow.add_edge("market_data", "technicals")
    workflow.add_edge("technicals", "news_research")
    workflow.add_edge("news_research", "thesis_gen")
    workflow.add_edge("thesis_gen", "compile_report")
    workflow.add_edge("compile_report", END)
    return workflow.compile()
