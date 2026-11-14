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

def fetch_technical_indicators(ticker: str) -> dict:
    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty or len(hist) < 30:
        return {"status": "Insufficient historical data"}
    close = hist["Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_14 = round(float((100 - (100 / (1 + rs))).fillna(50).iloc[-1]), 2)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else "N/A"
    sma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else "N/A"
    last_price = float(close.iloc[-1])
    trend = "Bullish (Above SMA200)" if isinstance(sma200, float) and last_price > sma200 else "Bearish / Neutral"
    return {
        "rsi_14": rsi_14,
        "macd": round(float(macd_line.iloc[-1]), 3),
        "macd_signal": round(float(signal_line.iloc[-1]), 3),
        "sma_50": sma50,
        "sma_200": sma200,
        "trend_signal": trend
    }

def fetch_latest_news(ticker: str, company_name: str) -> str:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    query = f"{company_name} {ticker} stock financial news earnings analyst rating"
    response = client.search(query=query, topic="news", max_results=5)
    articles = [f"- {r['title']}: {r['content']} (Source: {r['url']})" for r in response.get("results", [])]
    return "\n".join(articles) if articles else "No recent news found."
