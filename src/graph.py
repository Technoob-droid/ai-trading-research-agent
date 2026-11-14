from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from src.state import ResearchState, TradeDecision
from src.tools import fetch_market_metrics, fetch_technical_indicators, fetch_latest_news

def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

def market_data_node(state: ResearchState):
    print(f"[1/7] Fetching market metrics for {state['ticker']}...")
    return {"market_data": fetch_market_metrics(state["ticker"])}

def technicals_node(state: ResearchState):
    print(f"[2/7] Calculating technical indicators and ATR for {state['ticker']}...")
    return {"technical_indicators": fetch_technical_indicators(state["ticker"])}

def news_research_node(state: ResearchState):
    print(f"[3/7] Searching recent financial news for {state['ticker']}...")
    company_name = state["market_data"].get("name", state["ticker"])
    return {"news_summary": fetch_latest_news(state["ticker"], company_name)}

def thesis_generation_node(state: ResearchState):
    print("[4/7] Generating initial Bull and Bear theses...")
    llm = get_llm()
    context = f"Metrics: {state['market_data']}\nTechnicals: {state['technical_indicators']}\n\nNews:\n{state['news_summary']}"
    bull = llm.invoke([SystemMessage(content="You are a Lead Equity Bull Analyst. Write a concise, data-backed Bull Case (3 bullets)."), HumanMessage(content=context)])
    bear = llm.invoke([SystemMessage(content="You are a Lead Short-Seller / Risk Analyst. Write a concise, data-backed Bear Case (3 bullets)."), HumanMessage(content=context)])
    return {"bull_case": bull.content, "bear_case": bear.content}
