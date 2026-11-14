# AI Trading Research Agent

An autonomous multi-agent equity research pipeline built with **LangGraph**, **Groq()**, **Tavily Search**, and **yfinance**.

## Architecture
1. **Market Data Node:** Fetches live price, valuation metrics (P/E, 52w range), and 30-day returns.
2. **Technicals Node:** Calculates RSI (14-day), MACD, SMA-50, SMA-200, and trend signals.
3. **News Catalysts Node:** Aggregates real-time financial news via Tavily.
4. **Thesis Synthesis Node:** Generates data-backed Bull vs. Bear investment theses.
5. **Report Compiler Node:** Outputs structured Markdown research memos.

## Usage

### 1. Single-Ticker CLI Research
```bat
python main.py
```

### 2. Interactive Streamlit Dashboard
```bat
streamlit run app.py
```

### 3. Portfolio Batch Screener
```bat
python screener.py
```
