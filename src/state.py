from typing import TypedDict, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

class TradeDecision(BaseModel):
    action: Literal["STRONG BUY", "ACCUMULATE", "HOLD", "REDUCE", "SHORT"] = Field(description="Final investment committee action")
    conviction_score: int = Field(ge=0, le=100, description="Confidence score from 0 to 100")
    entry_zone: str = Field(description="Recommended entry price range")
    stop_loss: str = Field(description="Invalidation stop-loss price level based on ATR and support")
    target_price: str = Field(description="Primary upside/downside price target")
    risk_reward_ratio: str = Field(description="Estimated reward-to-risk ratio, e.g. 2.4:1")
    position_size_pct: float = Field(ge=0.0, le=15.0, description="Suggested portfolio allocation percentage (0-15%)")
    @field_validator("entry_zone", "stop_loss", "target_price", "risk_reward_ratio", mode="before")
    @classmethod
    def _as_str(cls, v):
        return str(v) if v is not None else ""

    rationale: str = Field(description="2-3 sentence executive synthesis explaining why one side won the debate")

class ResearchState(TypedDict):
    ticker: str
    market_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    mtf_confluence: Dict[str, Any]
    news_summary: str
    bull_case: str
    bear_case: str
    bull_rebuttal: str
    bear_rebuttal: str
    trade_plan: Dict[str, Any]
    final_report: str
