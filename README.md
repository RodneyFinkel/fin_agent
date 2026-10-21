---
title: Stock Ticker Agent
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Stock Ticker Agent

**Live demo:** [https://selahf-stock-ticker-agent.hf.space/](https://selahf-stock-ticker-agent.hf.space/)

A hybrid deterministic + LLM agent system for quantitative stock analysis.  
Users ask natural-language questions about a ticker. The system decides whether the question can be answered from pre-computed technical indicators or whether custom Python code must be generated and executed in a restricted sandbox. Results are grounded in real market data, research summaries, and deterministic calculations, then streamed back as a narrative with charts.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  User Prompt    │────▶│  Schema Layer    │────▶│  LLM Router        │
│  + Ticker       │     │  (DataFrame      │     │  (skip vs code)    │
└─────────────────┘     │   introspection) │     └─────────┬──────────┘
                        └──────────────────┘               │
                                                           │
                     ┌─────────────────────────────────────┼─────────────────────────┐
                     │                                     ▼                         │
                     │                          ┌────────────────────┐                │
                     │                          │  Code Generation   │                │
                     │                          │  (structured out)  │                │
                     │                          └─────────┬──────────┘                │
                     │                                    │                          │
                     │                                    ▼                          │
┌────────────────┐   │                          ┌────────────────────┐   ┌──────────┴─────────┐
│ Market Data    │───┼─────────────────────────▶│  Code Sandbox      │   │ Research Agent    │
│ Pipeline       │   │                          │  (Pandas + Matplotlib)│  │ (web search +     │
│ (Yahoo → SQLite│   │                          └─────────┬──────────┘   │  embeddings +     │
│  + Parquet)    │   │                                    │              │  ChromaDB)        │
└────────────────┘   │                                    ▼              └──────────┬─────────┘
                     │                          ┌────────────────────┐              │
                     │                          │ Deterministic      │              │
                     │                          │ Technical Picture  │              │
                     │                          │ (RSI, MAs, BB...)  │              │
                     │                          └─────────┬──────────┘              │
                     │                                    │                          │
                     │                                    ▼                          │
                     │                          ┌────────────────────┐              │
                     │                          │  LLM Synthesis     │◀─────────────┘
                     │                          │  (streamed narrative)│
                     │                          └─────────┬──────────┘
                     │                                    │
                     │                                    ▼
                     │                          ┌────────────────────┐
                     │                          │  LLM-as-Judge      │
                     │                          │  Evaluation        │
                     └──────────────────────────┴────────────────────┘
```

**Flow summary**

1. **Data layer** – Historical OHLCV + technical indicators are stored in SQLite/Parquet and served by `StockDataService`.
2. **Schema layer** – The live DataFrame is introspected (columns, dtypes, stats, date range, sample rows) and turned into a compact text block for the LLM. Full rows are never sent.
3. **Router** – A structured LLM call decides `skip` (answer from the deterministic picture alone) or `code` (custom calculation required).
4. **Code generation + sandbox** – If needed, the LLM emits a complete Pandas script that must end with a `SandboxOutputSchema`. The sandbox executes it, captures metrics and any Matplotlib figures, and archives long series to Parquet.
5. **Research agent** – Parallel async web research (DuckDuckGo → scrape → embed → rank → ChromaDB) produces a short grounded news summary.
6. **Synthesis** – A final LLM call streams a narrative that treats sandbox results as ground truth.
7. **Evaluation** – An optional LLM-as-judge scores the run on relevance, code correctness, narrative consistency, and efficiency.

---



## Example Queries

These illustrate the range of questions the agent can handle:

- *Calculate the 30-day rolling annualized volatility of daily returns and plot the rolling volatility trend line for the given ticker.*
- *What is the latest RSI and MA structure for the given ticker?*
- *What is the correlation of RSI vs Returns? Show charts and use lagged features.*

**Stress-test prompt** (exercises the full sandbox + strategy simulation path):

> Identify every instance where the stock experienced a "Volatility Squeeze"—defined as Bollinger Band Width `(upperBand - lowerBand) / ma20` dropping into its lowest 10th percentile over the preceding 60 trading days—while RSI was under 45. For each squeeze event, simulate entering a trade on the following day's open and exiting when either RSI crosses above 65 **or** after 15 trading days (whichever occurs first). Calculate the total strategy win rate, average trade duration, cumulative return, and the worst single trade's maximum underwater drawdown during its holding window.

---

## Screenshots

**Hot-swappable prompts**  
<img width="995" height="968" alt="Prompt Studio" src="https://github.com/user-attachments/assets/a150d59a-ab73-43e7-8f20-7cb475d5d7c2" />

**Deterministic technical snapshot + schema layer**  
<img width="792" height="966" alt="Technical Snapshot" src="https://github.com/user-attachments/assets/29c016eb-6828-43c1-baf4-7aee51254598" />

**Generated code, sandbox execution, and research synthesis**  
<img width="1050" height="966" alt="Sandbox + Synthesis" src="https://github.com/user-attachments/assets/695dadfd-6238-4f01-b798-ceffdf01a985" />
<img width="1434" height="598" alt="Evaluation / metrics" src="https://github.com/user-attachments/assets/35217504-7559-4501-8c5c-378be2ef76d9" />

---

## Run Locally (Docker)

You do **not** need to install Python or any other developer tools. Docker Desktop is enough.

**Requirements**
- Docker Desktop (Mac or Windows, ~8 GB RAM recommended)
- A free [Groq API key](https://console.groq.com/)
- This project folder (including `Dockerfile`, `docker-compose.yml`, and the SQLite database)

### Setup steps

1. **Install and start Docker Desktop**  
   - [Mac](https://docs.docker.com/desktop/setup/install/mac-install/)  
   - [Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

2. **Create a `.env` file** in the project root:

   ```
   GROQ_API_KEY=your_key_here
   ```

3. **Build and run**

   ```bash
   docker compose up --build
   ```

   The first run downloads base images and packages (10–15 minutes).  
   When ready you will see something like: `Uvicorn running on http://0.0.0.0:8000`

4. **Open the app**  
   [http://localhost:8000](http://localhost:8000)

Leave the terminal open while using the app. Press `Ctrl+C` to stop.

---

## Project Structure (key files)

| File / Module              | Role                                              |
|----------------------------|---------------------------------------------------|
| `slim_app2.py` / `app.py`  | FastAPI application + streaming endpoints         |
| `pipeline.py`              | Market data ingestion + indicator calculation     |
| `stock_service.py`         | SQLite / Parquet data access layer                |
| `schema_layer.py`          | DataFrame introspection for the LLM               |
| `analytical_engine.py`     | Deterministic technical picture + base64 charts   |
| `sandbox_engine.py`        | Restricted code execution environment             |
| `llm_synthesis.py`         | Router + code generation + narrative synthesis    |
| `agent5_async.py`          | Async research agent (search → scrape → embed)    |
| `prompt_manager.py`        | Hot-swappable system prompts                      |
| `evaluation_orchestrator.py` / `evaluator_agent2.py` | Run tracing + LLM-as-judge |
| `index_llm.html`           | Frontend (Alpine.js + Lightweight Charts)         |

---

## Limitations

- **Sandbox isolation** – Code runs via `exec` with restricted globals, not a full container or RestrictedPython sandbox. Suitable for personal / demo use; not recommended for untrusted multi-tenant production.
- **Data coverage** – Historical data starts around 2019. No fundamentals, earnings, options, or multi-asset data beyond what is stored in the local database.
- **Research quality** – Web research uses simple chunking + cosine ranking. Results can vary with search engine availability and page structure.
- **Model dependency** – Relies on Groq-hosted models. Latency and quality depend on the chosen model and free-tier rate limits.
- **No automated tests** – The evaluation harness exists but unit/integration tests are not yet present.
- **Single-user design** – No authentication, rate limiting, or multi-user session isolation.

---

## License

Personal / portfolio project. Use and modify freely for non-commercial purposes.
