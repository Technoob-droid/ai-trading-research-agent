import os, re
from datetime import datetime
import pandas as pd
import yfinance as yf

LEDGER_PATH = "data/paper_trades.csv"
COLS = ["Date", "Ticker", "Action", "Entry", "Stop_Loss", "Target", "Conviction", "Status", "Current_Exit", "R_Multiple"]

def _parse_num(val) -> float:
    nums = re.findall(r"[0-9]+\.?[0-9]*", str(val))
    if not nums:
        return 0.0
    vals = [float(x) for x in nums[:2]]
    return round(sum(vals) / len(vals), 2)

def load_ledger() -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LEDGER_PATH):
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(LEDGER_PATH)

def log_paper_trade(ticker: str, tp: dict) -> bool:
    df = load_ledger()
    entry = _parse_num(tp.get("entry_zone", 0))
    stop = _parse_num(tp.get("stop_loss", 0))
    targ = _parse_num(tp.get("target_price", 0))
    if entry <= 0 or stop <= 0 or targ <= 0:
        return False
    row = {"Date": datetime.now().strftime("%Y-%m-%d %H:%M"), "Ticker": ticker.upper(), "Action": tp.get("action", "BUY"), "Entry": entry, "Stop_Loss": stop, "Target": targ, "Conviction": tp.get("conviction_score", 0), "Status": "OPEN", "Current_Exit": entry, "R_Multiple": 0.0}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LEDGER_PATH, index=False)
    return True

def evaluate_ledger() -> pd.DataFrame:
    df = load_ledger()
    if df.empty:
        return df
    for idx, r in df.iterrows():
        if r["Status"] != "OPEN":
            continue
        hist = yf.Ticker(r["Ticker"]).history(period="5d")
        if hist.empty:
            continue
        last_px = round(float(hist["Close"].iloc[-1]), 2)
        hi_px = float(hist["High"].max())
        lo_px = float(hist["Low"].min())
        risk = abs(float(r["Entry"]) - float(r["Stop_Loss"])) or 1.0
        is_short = "SELL" in str(r["Action"]).upper()
        if (not is_short and lo_px <= float(r["Stop_Loss"])) or (is_short and hi_px >= float(r["Stop_Loss"])):
            df.at[idx, "Status"], df.at[idx, "Current_Exit"], df.at[idx, "R_Multiple"] = "STOPPED OUT", float(r["Stop_Loss"]), -1.0
        elif (not is_short and hi_px >= float(r["Target"])) or (is_short and lo_px <= float(r["Target"])):
            r_mult = round(abs(float(r["Target"]) - float(r["Entry"])) / risk, 2)
            df.at[idx, "Status"], df.at[idx, "Current_Exit"], df.at[idx, "R_Multiple"] = "TARGET HIT", float(r["Target"]), r_mult
        else:
            pnl = (float(r["Entry"]) - last_px) if is_short else (last_px - float(r["Entry"]))
            df.at[idx, "Current_Exit"], df.at[idx, "R_Multiple"] = last_px, round(pnl / risk, 2)
    df.to_csv(LEDGER_PATH, index=False)
    return df

def get_pm_lessons() -> str:
    df = load_ledger()
    closed = df[df["Status"].isin(["TARGET HIT", "STOPPED OUT"])]
    if closed.empty:
        return "No closed paper trades yet."
    wins = (closed["Status"] == "TARGET HIT").sum()
    avg_r = round(float(closed["R_Multiple"].mean()), 2)
    recent = closed.tail(5)[["Ticker", "Action", "Status", "R_Multiple"]].to_dict(orient="records")
    return f"Historical PM Track Record: {wins}/{len(closed)} wins, Avg R={avg_r}. Recent closed trades: {recent}"
