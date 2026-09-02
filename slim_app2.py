import asyncio
import logging
import json
import os
import urllib.parse
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
# APP Modules
from prompt_manager import prompt_manager
from agent5_async import ShortResearchAgent
from analytical_engine import build_analytical_picture
from stock_service import StockDataService
from llm_synthesis import LLM_Synthesis  # ← Dedicated LLM service
from schema_layer import build_df_schema, schema_to_prompt_block # <- NEW schema layer for LLM routing
from sandbox_engine import CodeSandbox

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

HTML_PATH = "index_llm.html"

# Initialize Services
stock_service = StockDataService(
    sqlite_path="fused_database5.db",
    parquet_path="parquet_cache/stock_cache_fin5.parquet",
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



#________NEW ENDPOINT WITH SANDBOX FOR CUSTOM CODE EXECUTION________

@app.post("/api/analyze")
async def analyze(payload: dict):
    prompt = payload.get("prompt", "").strip()
    ticker = payload.get("ticker", "UNKNOWN").upper()
    research_summary = payload.get("research", "")

    try:
        # 1. Fetch raw data & generate the deterministic picture immediately
        db_results = stock_service.get_chart_data(ticker)
        df = pd.DataFrame(db_results["data"])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        logging.info(f"Fetched {len(df)} rows for {ticker} from database.")
        
        # # Programmatically extract live DataFrame metadata
        # df_metadata = {
        #     "row_count": len(df),
        #     "columns": df.columns.tolist(),
        #     "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        #     "date_range": [str(df["time"].min()), str(df["time"].max())] if "time" in df.columns else "Unknown"
        # }
        
        schema = build_df_schema(df, ticker)
        schema_block = schema_to_prompt_block(schema)
        logging.info(f"Schema block for {ticker}:\n{schema_block}")
        picture = build_analytical_picture(db_results["data"], include_charts=False)
        logging.info(f"Generated analytical picture for {ticker}.")

        async def event_generator():
            # Initial progress update
            yield f"data: {json.dumps({'type': 'token', 'content': ' *Evaluating analytical state...*\\n\\n'})}\n\n"

            # 2. Ask LLM to evaluate the picture vs the user's prompt
            router_response = await analysis_service.evaluate_and_generate_code(
                ticker=ticker, 
                prompt=prompt, 
                picture=picture,
                schema_block=schema_block,
                research_summary=research_summary,
            )
            logging.info(f"Router response for {ticker}: {router_response[:200]}...")  # Log first 200 chars
            code_context = "No custom execution required. Baseline metrics used."

            # 3. Dynamic Routing: Check for the bypass keyword
            if "SKIP_EXECUTION" not in router_response:
            #if router_response and router_response != "SKIP_EXECUTION":
                yield f"data: {json.dumps({'type': 'token', 'content': ' *Running custom quantitative sandbox analysis...*\\n\\n'})}\n\n"
                
                
                # ---> SHOW THE CODE IN THE UI <---
                code_display = f"```python\n{router_response}\n```\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': code_display})}\n\n"
                
                # Extract code and execute in sandbox
                sandbox = CodeSandbox(timeout_seconds=8, persist_artifacts=True)
                logging.info(f"Executing sandbox code for {ticker}...")
                execution_res = await asyncio.to_thread(sandbox.execute_pandas_code, router_response, df, ticker,)
                logging.info(f"--- SANDBOX DEBUG --- Success: {execution_res['success']} | Artifact: {execution_res.get('artifact_parquet')} | Error: {execution_res['error']}")
                
                # Stream custom charts if generated
                if execution_res.get("chart"):
                    yield f"data: {json.dumps({'type': 'charts', 'charts': [execution_res['chart']]})}\n\n"

                ###NEW
                if execution_res["success"]:
                    code_context = execution_res["output"]  # already compact JSON
                    if execution_res.get("artifact_parquet"):
                        # Inform the final LLM that a longer series was archived
                        code_context += (
                            f"\n\n[Note: a longer intermediate series was archived to "
                            f"{execution_res['artifact_parquet']}; only the summary metrics above "
                            f"are available for narrative analysis.]"
                        )
                else:
                    code_context = f"Execution Error: {execution_res['error']}"
                
                #code_context = execution_res["output"] if execution_res["success"] else f"Execution Error: {execution_res['error']}"

            # 4. Final Synthesis: Pass all context to the LLM for the final stream
            async for token in analysis_service.generate_synthesis_stream(
                ticker=ticker,
                prompt=prompt,
                picture=picture,
                code_output=code_context,
                research_summary=research_summary
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logging.exception(f"Error in /api/analyze for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    

# --- STREAMING RESEARCH & STOCK DATA ENDPOINT ---
@app.get("/api/stock/{ticker}/stream")
async def stream_stock_data(ticker: str):
    queue = asyncio.Queue()

    async def progress_callback(msg: str):
        await queue.put({"type": "progress", "msg": msg})

    async def run_research_task():
        try:
            query = f"{ticker} stock latest financial news and analysis"
            raw_research = await research_agent.run(query, progress_callback=progress_callback)

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
            await queue.put({"type": "research_complete", "data": formatted_research})
        except Exception as e:
            logger.exception(f"Research streaming error for {ticker}: {e}")
            await queue.put({"type": "error", "msg": str(e)})
        finally:
            await queue.put(None)  # Sentinel to close queue

    async def event_generator():
        # 1. Immediately fetch SQLite chart data & technical indicators
        await queue.put({"type": "progress", "msg": "Querying database & computing technical indicators..."})
        db_results = await asyncio.to_thread(stock_service.get_chart_data, ticker)
        analytical_picture = build_analytical_picture(db_results["data"], include_charts=True)

        # 2. Emit base stock data immediately so charts render right away
        await queue.put({
            "type": "stock_base",
            "company": db_results["company"],
            "sector": db_results["sector"],
            "chart_data": db_results["data"],
            "analytical_picture": analytical_picture
        })

        # 3. Kick off async research task in background
        asyncio.create_task(run_research_task())

        # 4. Stream events from queue to SSE client
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


#####hot-swappable prompt system.

@app.get("/api/prompts")
def list_prompts():
    return {"prompts": prompt_manager.list_prompts()}


@app.get("/api/prompts/{name}")
def get_prompt(name: str):
    try:
        return {"name": name, "content": prompt_manager.get(name)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prompt '{name}' not found")


@app.put("/api/prompts/{name}")
async def update_prompt(name: str, payload: dict):
    content = payload.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    prompt_manager.set(name, content)
    return {"status": "ok", "name": name}


if __name__ == "__main__":
    uvicorn.run("slim_app2:app", host="127.0.0.1", port=8000, reload=True)