from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from src.state import ResearchState, TradeDecision
from src.tools import fetch_market_metrics, fetch_technical_indicators, fetch_latest_news, fetch_mtf_confluence

def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

def market_data_node(state: ResearchState):
    print(f"[1/7] Fetching market metrics for {state['ticker']}...")
    return {"market_data": fetch_market_metrics(state["ticker"])}

def technicals_node(state: ResearchState):
    print(f"[2/7] Calculating technical indicators and ATR for {state['ticker']}...")
    return {"technical_indicators": fetch_technical_indicators(state["ticker"]), "mtf_confluence": fetch_mtf_confluence(state["ticker"])}

def news_research_node(state: ResearchState):
    print(f"[3/7] Searching recent financial news for {state['ticker']}...")
    company_name = state["market_data"].get("name", state["ticker"])
    return {"news_summary": fetch_latest_news, fetch_mtf_confluence(state["ticker"], company_name)}

def thesis_generation_node(state: ResearchState):
    print("[4/7] Generating initial Bull and Bear theses...")
    llm = get_llm()
    context = f"Metrics: {state['market_data']}\nTechnicals: {state['technical_indicators']}\nMTF Confluence: {state.get('mtf_confluence')}\n\nNews:\n{state['news_summary']}"
    bull = llm.invoke([SystemMessage(content="You are a Lead Equity Bull Analyst. Write a concise, data-backed Bull Case (3 bullets)."), HumanMessage(content=context)])
    bear = llm.invoke([SystemMessage(content="You are a Lead Short-Seller / Risk Analyst. Write a concise, data-backed Bear Case (3 bullets)."), HumanMessage(content=context)])
    return {"bull_case": bull.content, "bear_case": bear.content}

def adversarial_debate_node(state: ResearchState):
    print("[5/7] Running cross-examination rebuttal round...")
    llm = get_llm()
    base = f"Metrics: {state['market_data']}\nTechnicals: {state['technical_indicators']}\nMTF Confluence: {state.get('mtf_confluence')}"
    bull_reb = llm.invoke([SystemMessage(content="You are the Bull Analyst. Directly refute the Bear Analyst's weakest assumptions using the market/technical data in 2 sharp bullets."), HumanMessage(content=f"{base}\n\nBear Thesis to Refute:\n{state['bear_case']}")])
    bear_reb = llm.invoke([SystemMessage(content="You are the Bear Analyst. Directly expose flaws or traps in the Bull Analyst's thesis using the market/technical data in 2 sharp bullets."), HumanMessage(content=f"{base}\n\nBull Thesis to Refute:\n{state['bull_case']}")])
    return {"bull_rebuttal": bull_reb.content, "bear_rebuttal": bear_reb.content}

def pm_judge_node(state: ResearchState):
    print("[6/7] Portfolio Manager Judge synthesizing trade plan...")
    llm = get_llm().with_structured_output(TradeDecision, method="json_schema")
    cur = "INR " if state["ticker"].endswith((".NS", ".BO")) else "$"
    prompt = f"Ticker: {state['ticker']} (Currency: {cur})\nFundamentals: {state['market_data']}\nTechnicals (including ATR-14): {state['technical_indicators']}\nMTF Confluence (Weekly/Daily/1H): {state.get('mtf_confluence')}\n\nBull Thesis:\n{state['bull_case']}\nBull Rebuttal:\n{state['bull_rebuttal']}\n\nBear Thesis:\n{state['bear_case']}\nBear Rebuttal:\n{state['bear_rebuttal']}"
    decision = llm.invoke([SystemMessage(content="You are a Senior Hedge Fund Portfolio Manager. Evaluate both sides of the debate objectively and output a risk-managed TradeDecision using ATR-14 for realistic stop-loss and target levels."), HumanMessage(content=prompt)])
    return {"trade_plan": decision.model_dump()}

def report_compiler_node(state: ResearchState):
    print("[7/7] Compiling Investment Committee Memo...")
    m, t, pr = state["market_data"], state["technical_indicators"], state["trade_plan"]
    cur = "INR " if state["ticker"].endswith((".NS", ".BO")) else "$"
    lines = [f"# Investment Committee Memo: {m.get('name')} ({state['ticker']})", f"## PM Verdict: {pr.get('action')} ({pr.get('conviction_score')}/100)", f"- Entry: {pr.get('entry_zone')} / Stop: {pr.get('stop_loss')} / Target: {pr.get('target_price')} / R:R: {pr.get('risk_reward_ratio')}", f"- Rationale: {pr.get('rationale')}", f"- Price: {cur}{m.get('current_price')} / RSI: {t.get('rsi_14')} / ATR: {cur}{t.get('atr_14')}", "## Bull Case", state['bull_case'], state['bull_rebuttal'], "## Bear Case", state['bear_case'], state['bear_rebuttal']]
    return {"final_report": "\n\n".join(lines)}

def build_research_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("market_data", market_data_node)
    workflow.add_node("technicals", technicals_node)
    workflow.add_node("news_research", news_research_node)
    workflow.add_node("thesis_gen", thesis_generation_node)
    workflow.add_node("adversarial_debate", adversarial_debate_node)
    workflow.add_node("pm_judge", pm_judge_node)
    workflow.add_node("compile_report", report_compiler_node)
    workflow.set_entry_point("market_data")
    workflow.add_edge("market_data", "technicals")
    workflow.add_edge("technicals", "news_research")
    workflow.add_edge("news_research", "thesis_gen")
    workflow.add_edge("thesis_gen", "adversarial_debate")
    workflow.add_edge("adversarial_debate", "pm_judge")
    workflow.add_edge("pm_judge", "compile_report")
    workflow.add_edge("compile_report", END)
    return workflow.compile()
