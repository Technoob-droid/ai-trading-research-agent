from typing import TypedDict, Dict, Any

class ResearchState(TypedDict):
    ticker: str
    market_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    news_summary: str
    bull_case: str
    bear_case: str
    final_report: str
