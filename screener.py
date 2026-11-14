import os, sys
from dotenv import load_dotenv
from src.graph import build_research_graph

load_dotenv()

def run_batch_screener(tickers: list[str]):
    app = build_research_graph()
    os.makedirs("reports", exist_ok=True)
    results = []
    for idx, ticker in enumerate(tickers, 1):
        print(f"\n=== [{idx}/{len(tickers)}] Analyzing {ticker} ===")
        try:
            res = app.invoke({"ticker": ticker})
            report_path = os.path.join("reports", f"{ticker}_research_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(res["final_report"])
            results.append(res)
        except Exception as e:
            print(f"Failed on {ticker}: {e}")
    return results

def save_summary_report(results: list[dict]):
    results.sort(key=lambda x: float(x["market_data"].get("1m_return_pct", 0) or 0), reverse=True)
    lines = [
        "# Portfolio Batch Screener Summary\n",
        "| Rank | Ticker | Company | Price | 1M Return | Trailing P/E | RSI (14) | Trend Signal |",
        "| ---- | ------ | ------- | ----- | --------- | ------------ | -------- | ------------ |"
    ]
    for rank, r in enumerate(results, 1):
        m = r["market_data"]
        t = r["technical_indicators"]
        lines.append(f"| {rank} | {r['ticker']} | {m.get('name', 'N/A')} | ${m.get('current_price', 'N/A')} | {m.get('1m_return_pct', 0)}% | {m.get('trailing_pe', 'N/A')} | {t.get('rsi_14', 'N/A')} | {t.get('trend_signal', 'N/A')} |")
    out_path = os.path.join("reports", "portfolio_screener_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary table saved to {out_path}")

def main():
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("ERROR: Set GROQ_API_KEY and TAVILY_API_KEY in .env")
        sys.exit(1)
    raw = input("Enter comma-separated tickers (e.g., NVDA, AAPL, TCS.NS): ")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        sys.exit(1)
    results = run_batch_screener(tickers)
    if results:
        save_summary_report(results)

if __name__ == "__main__":
    main()
