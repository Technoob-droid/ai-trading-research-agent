import os, sys
from dotenv import load_dotenv
from src.graph import build_research_graph

load_dotenv()

def main():
    if not os.getenv("GROQ_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("ERROR: Set GROQ_API_KEY and TAVILY_API_KEY in .env")
        sys.exit(1)
    ticker = input("Enter stock ticker ^(e.g., NVDA, TCS.NS^): ").strip().upper()
    result = build_research_graph().invoke({"ticker": ticker})
    os.makedirs("reports", exist_ok=True)
    out = os.path.join("reports", f"{ticker}_research_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(result["final_report"])
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
