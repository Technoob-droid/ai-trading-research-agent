from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

class StrategySpec(BaseModel):
    name: str = Field(description="Strategy name")
    rsi_below: Optional[float] = Field(default=None, description="Buy when RSI 14 is below value")
    rsi_above: Optional[float] = Field(default=None, description="Buy when RSI 14 is above value")
    above_sma200: bool = Field(default=False, description="Require Close above SMA 200")
    below_bb_lower: bool = Field(default=False, description="Require Close at or below Lower BB 20")
    atr_stop_mult: float = Field(default=1.5, description="Stop-loss in ATR multiples")
    atr_target_mult: float = Field(default=2.5, description="Take-profit in ATR multiples")
    max_hold_days: int = Field(default=15, description="Max holding days")

def compile_strategy(prompt: str) -> StrategySpec:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0).with_structured_output(StrategySpec, method="json_schema")
    return llm.invoke([SystemMessage(content="Convert trading rule into StrategySpec JSON."), HumanMessage(content=prompt)])

def critique_backtest(ticker: str, spec: StrategySpec, metrics: dict, trades: list) -> str:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
    prompt = f"Ticker: {ticker}\nSpec: {spec.model_dump()}\nMetrics: {metrics}\nRecent Trades: {trades[-10:]}"
    return llm.invoke([SystemMessage(content="Diagnose this backtest in 3 concise bullets: (1) Edge vs Buy-and-Hold, (2) Primary failure mode, (3) One concrete parameter tweak."), HumanMessage(content=prompt)]).content

def run_backtest(ticker: str, spec: StrategySpec, period: str = "2y"):
    df = yf.Ticker(ticker).history(period=period)
    if df.empty or len(df) < 60:
        return None, {}, []
    c = df["Close"]
    d = c.diff()
    g = d.where(d > 0, 0.0).rolling(14).mean()
    l = (-d.where(d < 0, 0.0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    df["SMA200"] = c.rolling(200).mean()
    df["BB_Low"] = c.rolling(20).mean() - 2 * c.rolling(20).std()
    tr = (df["High"] - df["Low"]).combine((df["High"] - c.shift()).abs(), max).combine((df["Low"] - c.shift()).abs(), max)
    df["ATR"] = tr.rolling(14).mean()
    cond = pd.Series(True, index=df.index)
    if spec.rsi_below is not None:
        cond = cond.mul(df["RSI"].lt(spec.rsi_below))
    if spec.rsi_above is not None:
        cond = cond.mul(df["RSI"].gt(spec.rsi_above))
    if spec.above_sma200:
        cond = cond.mul(c.gt(df["SMA200"]))
    if spec.below_bb_lower:
        cond = cond.mul(c.le(df["BB_Low"]))
    df["Signal"] = cond.astype(bool)
    equity, trades, in_pos, eq_curve = 100.0, [], False, []
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        if not in_pos and prev["Signal"] and pd.notna(prev["ATR"]):
            in_pos, entry_px, entry_dt, hold = True, float(row["Open"]), df.index[i], 0
            stop_px = entry_px - spec.atr_stop_mult * float(prev["ATR"])
            targ_px = entry_px + spec.atr_target_mult * float(prev["ATR"])
        if in_pos:
            hold += 1
            exit_px, reason = None, None
            if float(row["Low"]) <= stop_px:
                exit_px, reason = stop_px, "Stop-Loss"
            elif float(row["High"]) >= targ_px:
                exit_px, reason = targ_px, "Target Hit"
            elif hold >= spec.max_hold_days or i == len(df) - 1:
                exit_px, reason = float(row["Close"]), "Time Exit"
            if exit_px is not None:
                ret = (exit_px - entry_px) / entry_px
                equity *= (1 + ret)
                trades.append({"Entry Date": str(entry_dt.date()), "Exit Date": str(df.index[i].date()), "Entry": round(entry_px, 2), "Exit": round(exit_px, 2), "Return %": round(ret * 100, 2), "Reason": reason})
                in_pos = False
        eq_curve.append({"Date": df.index[i], "Strategy": round(equity, 2), "BuyHold": round(100.0 * float(row["Close"]) / float(df["Close"].iloc[1]), 2)})
    eq_df = pd.DataFrame(eq_curve).set_index("Date")
    wins = [t["Return %"] for t in trades if t["Return %"] > 0]
    losses = [abs(t["Return %"]) for t in trades if t["Return %"] <= 0]
    dd = ((eq_df["Strategy"] - eq_df["Strategy"].cummax()) / eq_df["Strategy"].cummax() * 100).min()
    metrics = {"trades": len(trades), "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0.0, "profit_factor": round(sum(wins) / sum(losses), 2) if losses and sum(losses) > 0 else (99.0 if wins else 0.0), "strat_ret": round(equity - 100.0, 2), "bh_ret": round(float(eq_df["BuyHold"].iloc[-1]) - 100.0, 2), "max_dd": round(float(dd), 2)}
    return eq_df, metrics, trades

def optimize_strategy(ticker: str, spec: StrategySpec, metrics: dict, trades: list, critique: str) -> StrategySpec:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2).with_structured_output(StrategySpec, method="json_schema")
    sys_msg = "You are a Quantitative Strategy Optimizer. Modify the StrategySpec parameters to fix the failure modes identified in the backtest critique and improve Profit Factor and Max Drawdown while still generating trades. Give the optimized strategy a clear updated name."
    prompt = f"Ticker: {ticker}\nCurrent Spec: {spec.model_dump()}\nMetrics: {metrics}\nCritique: {critique}\nSample Trades: {trades[-10:]}"
    return llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=prompt)])
