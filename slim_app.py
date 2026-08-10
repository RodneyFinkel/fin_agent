import asyncio
import urllib.parse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

from stock_service import StockDataService
from agent5_async import ShortResearchAgent
from analytical_engine import build_analytical_picture



app = FastAPI(title="Stock Market Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PATH = "index.html"

# Services
stock_service = StockDataService(
    sqlite_path="fused_database4.db",
    parquet_path="stock_cache_fin4.parquet",
)
research_agent = ShortResearchAgent()


@app.get("/")
def serve_frontend():
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    return {"error": f"{HTML_PATH} not found in the root directory."}


@app.get("/api/tickers")
def get_all_tickers():
    return stock_service.get_all_tickers()


@app.post("/api/ticker/add")
def add_ticker(payload: dict):
    ticker = payload.get("ticker", "").strip().upper()
    message = stock_service.add_ticker(ticker)
    return {"status": "success", "message": message}


@app.get("/api/stock/{ticker}")
async def get_stock_data(ticker: str):
    """
    Thin orchestration layer.
    Chart/metric data comes from StockDataService.
    Research comes from ShortResearchAgent.
    """
    try:
        # 1. Chart + metrics (blocking → thread)
        db_task = asyncio.to_thread(stock_service.get_chart_data, ticker)

        # 2. Research
        query = f"{ticker} stock latest financial news and analysis"
        research_task = research_agent.run(query)

        db_results, raw_research = await asyncio.gather(db_task, research_task)
        
        # NEW: Analytical engine (deterministic)
        analytical_picture = build_analytical_picture(
            db_results["data"],
            include_charts=True
        )

        # 3. Format research for frontend
        unique_urls = set()
        sources = []
        for p in raw_research.get("passages", []):
            url = p["url"]
            if url not in unique_urls:
                unique_urls.add(url)
                domain = urllib.parse.urlparse(url).netloc.replace("www.", "")
                sources.append({"title": domain, "url": url})

        formatted_research = {
            "summary": raw_research.get("summary", "No recent insights found for this asset."),
            "sources": sources,
            "telemetry": {
                "execution_time": round(raw_research.get("time", 0.0), 2),
                "vectors_matched": len(raw_research.get("passages", [])),
            },
        }

        return {
            "ticker": ticker.upper(),
            "company": db_results["company"],
            "sector": db_results["sector"],
            "chart_data": db_results["data"],
            "research_summary": formatted_research,
            "analytical_picture": analytical_picture,   # ← NEW
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/analyze")
async def analyze(payload: dict):
    prompt = payload.get("prompt", "").strip()
    ticker = payload.get("ticker", "TSLA").upper()

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        # 1. Get the same chart data the frontend already uses
        db_results = stock_service.get_chart_data(ticker)

        # 2. Run the analytical engine
        picture = build_analytical_picture(
            db_results["data"],
            include_charts=True
        )

        # 3. Build a simple text analysis from the numbers
        #    (we can replace this with LLM synthesis later)
        analysis_parts = [
            f"**Technical Snapshot for {ticker}** (as of {picture.get('as_of')})",
            "",
            f"• RSI: {picture['rsi'].get('latest_rsi')} — {picture['rsi'].get('regime')}",
            f"• Moving Averages: {picture['moving_averages'].get('structure')}",
            f"• Bollinger: {picture['bollinger'].get('state')} (Width: {picture['bollinger'].get('band_width')})",
            f"• Returns → 5d: {picture['performance'].get('return_5d')}% | "
            f"20d: {picture['performance'].get('return_20d')}% | "
            f"60d: {picture['performance'].get('return_60d')}%",
            "",
            f"Question: {prompt}",
            "",
            "(Charts generated from the same data shown above)"
        ]
        analysis = "\n".join(analysis_parts)

        # 4. Collect charts in the order we want to show them
        charts = []
        if "charts" in picture:
            for key in ["rsi", "ma_structure", "bollinger", "returns"]:
                if key in picture["charts"]:
                    charts.append(picture["charts"][key])

        return {
            "status": "success",
            "analysis": analysis,
            "charts": charts
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("slim_app:app", host="127.0.0.1", port=8000, reload=True)