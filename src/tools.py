import os
import yfinance as yf
from tavily import TavilyClient

def fetch_market_metrics(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1mo")
    pct_change_1m = 0.0
    if not hist.empty and len(hist) > 1:
        pct_change_1m = round(((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100, 2)
    return {
        "name": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
        "market_cap": info.get("marketCap", "N/A"),
        "trailing_pe": info.get("trailingPE", "N/A"),
        "forward_pe": info.get("forwardPE", "N/A"),
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        "1m_return_pct": pct_change_1m
    }

def fetch_latest_news(ticker: str, company_name: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"{company_name} {ticker} stock financial news earnings analyst rating"
    response = client.search(query=query, topic="news", max_results=5)
    articles = [f"- {r['title']}: {r['content']} (Source: {r'url']})" for r in response.get("results", [])]
    return "\n".join(articles) if articles else "No recent news found."
