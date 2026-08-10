import asyncio
import logging
import os
import urllib.parse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from agent5_async import ShortResearchAgent
from analytical_engine import build_analytical_picture
from stock_service import StockDataService
from llm_synthesis import LLM_Synthesis  # ← Dedicated LLM service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("StockAPI")

app = FastAPI(title="Stock Market Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PATH = "index3.html"

# Initialize Services
stock_service = StockDataService(
    sqlite_path="fused_database4.db",
    parquet_path="stock_cache_fin4.parquet",
)
research_agent = ShortResearchAgent()
analysis_service = LLM_Synthesis()


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
    try:
        db_task = asyncio.to_thread(stock_service.get_chart_data, ticker)
        query = f"{ticker} stock latest financial news and analysis"
        research_task = research_agent.run(query)

        db_results, raw_research = await asyncio.gather(db_task, research_task)

        analytical_picture = build_analytical_picture(
            db_results["data"], include_charts=True
        )

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
            "analytical_picture": analytical_picture,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"CRITICAL ERROR in get_stock_data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze(payload: dict):
    prompt = payload.get("prompt", "").strip()
    ticker = payload.get("ticker", "UNKNOWN").upper()
    research_summary = payload.get("research", "No recent fundamental news available.")
    metrics = payload.get("metrics", {})

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        # 1. Fetch raw data to generate base64 chart plots
        db_results = stock_service.get_chart_data(ticker)
        picture = build_analytical_picture(db_results["data"], include_charts=True)

        # 2. Delegate synthesis to AnalysisService
        llm_analysis = await LLM_Synthesis.generate_synthesis(
            ticker=ticker,
            prompt=prompt,
            metrics=metrics,
            research_summary=research_summary
        )

        # 3. Collect charts for payload
        charts = []
        if "charts" in picture:
            for key in ["rsi", "ma_structure", "bollinger", "returns"]:
                if key in picture["charts"]:
                    charts.append(picture["charts"][key])

        return {
            "status": "success",
            "analysis": llm_analysis,
            "charts": charts
        }

    except Exception as e:
        logger.exception(f"Error in /api/analyze for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("slim_app_2:app", host="127.0.0.1", port=8000, reload=True)