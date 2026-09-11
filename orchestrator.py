from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any
import pandas as pd

from stock_service import StockDataService
from llm_synthesis import LLM_Synthesis
from schema_layer import build_df_schema, schema_to_prompt_block
from analytical_engine import build_analytical_picture
from sandbox_engine import CodeSandbox

logger = logging.getLogger("AnalysisOrchestrator")


class AnalysisOrchestrator:
    def __init__(
        self,
        stock_service: StockDataService,
        analysis_service: LLM_Synthesis,
        sandbox_timeout: int = 8,
    ):
        self.stock_service = stock_service
        self.analysis_service = analysis_service
        self.sandbox_timeout = sandbox_timeout

    async def run_analysis(
        self,
        ticker: str,
        prompt: str,
        research_summary: str = "",
        session_id: str = "default_session",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.perf_counter()

        # 1. Pipeline Initialization & Context Loading
        yield {
            "type": "trace",
            "phase": "FETCH",
            "message": f"Querying SQLite & Parquet store for {ticker} chart data...",
        }
        
        db_results = self.stock_service.get_chart_data(ticker)
        df = pd.DataFrame(db_results["data"])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        yield {
            "type": "trace",
            "phase": "SCHEMA",
            "message": f"Loaded {len(df)} candles into frame. Extracting memory schema & distribution stats...",
        }

        # Yield frame schema for CLI rendering
        yield {
            "type": "schema",
            "df": df,
            "table_name": f"sqlite.{ticker}_daily",
        }

        schema = build_df_schema(df, ticker)
        schema_block = schema_to_prompt_block(schema)
        picture = build_analytical_picture(db_results["data"], include_charts=False)

        if research_summary:
            yield {
                "type": "trace",
                "phase": "RESEARCH",
                "message": f"Injected research context vector ({len(research_summary)} chars).",
            }

        # 2. LLM Routing & Code Synthesis
        yield {
            "type": "trace",
            "phase": "ROUTER",
            "message": "Sending prompt & analytical context to LLM code routing engine...",
        }

        router_response = await self.analysis_service.evaluate_and_generate_code(
            ticker=ticker,
            prompt=prompt,
            picture=picture,
            schema_block=schema_block,
            research_summary=research_summary,
        )

        code_context = "No custom execution required. Baseline metrics used."
        code_generated = False
        sandbox_time = 0.0

        # 3. Code Execution Phase
        if "SKIP_EXECUTION" not in router_response:
            code_generated = True
            yield {
                "type": "trace",
                "phase": "ROUTER",
                "message": "Custom quantitative logic generated. Dispatching to isolated sandbox...",
            }
            yield {"type": "code", "content": router_response}

            yield {
                "type": "trace",
                "phase": "SANDBOX",
                "message": f"Executing dynamic Pandas script in isolated thread (Timeout: {self.sandbox_timeout}s)...",
            }

            sb_start = time.perf_counter()
            sandbox = CodeSandbox(
                timeout_seconds=self.sandbox_timeout, persist_artifacts=True
            )
            execution_res = await asyncio.to_thread(
                sandbox.execute_pandas_code,
                router_response,
                df,
                ticker,
            )
            sandbox_time = time.perf_counter() - sb_start

            charts = execution_res.get("charts") or []
            if not charts and execution_res.get("chart"):
                charts = [execution_res["chart"]]

            if charts:
                yield {
                    "type": "trace",
                    "phase": "SANDBOX",
                    "message": f"Rendered {len(charts)} chart artifact(s).",
                }
                yield {"type": "charts", "charts": charts}

            if execution_res["success"]:
                yield {
                    "type": "trace",
                    "phase": "SANDBOX",
                    "message": f"Sandbox execution SUCCESS ({sandbox_time:.2f}s). Extracting output metrics...",
                }
                code_context = execution_res["output"]
                if execution_res.get("artifact_parquet"):
                    code_context += (
                        f"\n\n[Note: intermediate series archived to "
                        f"{execution_res['artifact_parquet']}]"
                    )
            else:
                yield {
                    "type": "trace",
                    "phase": "SANDBOX_ERR",
                    "message": f"Execution error: {execution_res['error']}",
                }
                code_context = f"Execution Error: {execution_res['error']}"

        # 4. Final Narrative Synthesis
        yield {
            "type": "trace",
            "phase": "SYNTHESIS",
            "message": "Streaming final narrative synthesis from LLM...",
        }

        async for token in self.analysis_service.generate_synthesis_stream(
            ticker=ticker,
            prompt=prompt,
            picture=picture,
            code_output=code_context,
            research_summary=research_summary,
        ):
            yield {"type": "token", "content": token}

        total_time = time.perf_counter() - start_time
        yield {
            "type": "metrics",
            "ticker": ticker,
            "latency": round(total_time, 2),
            "sandbox_time": round(sandbox_time, 2),
            "code_executed": code_generated,
        }


# =====================================================================
# Rich CLI Entry Point & Presentation Helpers
# =====================================================================
if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.status import Status

    console = Console()

    def print_schema_summary(
        console: Console, df: pd.DataFrame, table_name: str = "market_data"
    ) -> None:
        """Prints a styled Rich table detailing the DB schema, data types, and value bounds."""
        if df.empty:
            console.print(f"[bold red]Schema Error:[/bold red] Table '{table_name}' is empty.")
            return

        row_count = f"{len(df):,}"
        col_count = len(df.columns)

        date_str = "N/A"
        if "time" in df.columns and not df["time"].dropna().empty:
            start_t = str(df["time"].min())[:10]
            end_t = str(df["time"].max())[:10]
            date_str = f"{start_t} ➔ {end_t}"

        meta_text = (
            f"[bold cyan]Source Table:[/bold cyan] {table_name}  |  "
            f"[bold green]Shape:[/bold green] ({row_count} rows, {col_count} cols)  |  "
            f"[bold yellow]Span:[/bold yellow] {date_str}"
        )
        console.print(
            Panel(meta_text, title="🗄️ Database & Frame Schema", expand=False)
        )

        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Column Name", style="bold cyan", no_wrap=True)
        table.add_column("Dtype", style="green")
        table.add_column("Nulls", style="red", justify="right")
        table.add_column("Sample / Range (Min ➔ Max)", style="dim")

        for col in df.columns:
            dtype = str(df[col].dtype)
            null_cnt = f"{df[col].isnull().sum():,}"

            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].dropna().empty:
                    val_range = "N/A (All NaN)"
                else:
                    val_range = f"[{df[col].min():.2f} ➔ {df[col].max():.2f}]"
            elif (
                pd.api.types.is_datetime64_any_dtype(df[col]) or col.lower() == "time"
            ):
                valid_dates = df[col].dropna()
                if not valid_dates.empty:
                    val_range = (
                        f"[{str(valid_dates.iloc[0])[:10]} ➔ {str(valid_dates.iloc[-1])[:10]}]"
                    )
                else:
                    val_range = "N/A"
            else:
                valid_vals = df[col].dropna()
                val_range = f"Sample: '{valid_vals.iloc[0]}'" if not valid_vals.empty else "N/A"

            table.add_row(col, dtype, null_cnt, val_range)

        console.print(table)
        console.print()

    def print_banner():
        console.clear()
        banner = r"""
   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗    ███████╗███╗   ██╗██████╗ 
  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔════╝████╗  ██║██╔══██╗
  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       █████╗  ██╔██╗ ██║██║  ██║
  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔══╝  ██║╚██╗██║██║  ██║
  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ███████╗██║ ╚████║██████╔╝
   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══════╝╚═╝  ╚═══╝╚═════╝ 
        """
        console.print(Panel(banner, title="[bold cyan]QUANT ENGINE CLI[/bold cyan]", border_style="cyan"))

    async def main():
        stock_svc = StockDataService(
            sqlite_path="fused_database5.db",
            parquet_path="parquet_cache/stock_cache_fin5.parquet",
        )
        llm_svc = LLM_Synthesis()
        orchestrator = AnalysisOrchestrator(stock_svc, llm_svc)

        print_banner()

        while True:
            ticker = Prompt.ask("\n[bold green]Ticker[/bold green]", default="NVDA").upper()
            if ticker in ["EXIT", "QUIT", "Q"]:
                break

            prompt = Prompt.ask(
                "[bold green]Prompt[/bold green]",
                default="Calculate 20-day rolling volatility and identify worst drawdown date."
            )

            console.print("\n" + "─" * 70)

            synthesis_text = ""
            metrics = None

            with Status("[bold cyan]Initializing pipeline...", spinner="dots") as status:
                async for event in orchestrator.run_analysis(ticker, prompt):
                    event_type = event.get("type")

                    if event_type == "trace":
                        phase = event["phase"]
                        msg = event["message"]
                        status.update(f"[bold yellow][{phase}][/bold yellow] {msg}")
                        console.print(f"[dim font-mono]  └─ [{phase}] {msg}[/dim font-mono]")

                    elif event_type == "schema":
                        status.stop()
                        print_schema_summary(
                            console, event["df"], table_name=event["table_name"]
                        )
                        status.start()

                    elif event_type == "code":
                        status.stop()
                        console.print(Panel(
                            Syntax(event["content"].strip(), "python", theme="monokai", line_numbers=True),
                            title="[bold green]⚡ Dynamic Sandbox Python Code[/bold green]",
                            border_style="green"
                        ))
                        status.start()

                    elif event_type == "token":
                        if status.start:
                            status.stop()
                        content = event["content"]
                        synthesis_text += content
                        sys.stdout.write(content)
                        sys.stdout.flush()

                    elif event_type == "metrics":
                        metrics = event

            if metrics:
                console.print("\n")
                table = Table(title="[bold cyan]Execution Diagnostic Report[/bold cyan]", border_style="dim")
                table.add_column("Metric", style="bold white")
                table.add_column("Value", style="bold green")
                table.add_row("Ticker", metrics["ticker"])
                table.add_row("Total Pipeline Latency", f"{metrics['latency']}s")
                table.add_row("Sandbox Time", f"{metrics['sandbox_time']}s")
                table.add_row("Code Executed", "Yes" if metrics["code_executed"] else "No")
                console.print(table)

            console.print("─" * 70)

    asyncio.run(main())