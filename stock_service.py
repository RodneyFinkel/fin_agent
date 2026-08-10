import sqlite3
import os
import pandas as pd
from fastapi import HTTPException
from fin_tool_last import MarketDataPipelineTool


class StockDataService:
    def __init__(self, sqlite_path: str = "fused_database4.db", parquet_path: str = "stock_cache_fin4.parquet"):
        self.sqlite_path = sqlite_path
        self.parquet_path = parquet_path
        self.pipeline_tool = MarketDataPipelineTool()

    def get_all_tickers(self) -> list[dict]:
        if not os.path.exists(self.sqlite_path):
            return []

        try:
            conn = sqlite3.connect(self.sqlite_path)
            query = "SELECT DISTINCT Ticker, Company, Sector FROM stock_metrics"
            df = pd.read_sql_query(query, conn)
            conn.close()

            tickers_list = []
            for _, row in df.iterrows():
                tickers_list.append({
                    "ticker": row["Ticker"],
                    "company": row["Company"] if pd.notna(row["Company"]) else row["Ticker"],
                    "sector": row["Sector"] if pd.notna(row["Sector"]) else "Unknown",
                })
            return tickers_list
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_chart_data(self, ticker: str) -> dict:
        """
        Returns the exact payload structure the frontend expects for chart + meta.
        """
        if not os.path.exists(self.sqlite_path):
            raise HTTPException(
                status_code=404,
                detail="SQLite database not found. Run pipeline first.",
            )

        conn = sqlite3.connect(self.sqlite_path)
        query = """
            SELECT 
                Date as time, Open as open, High as high, Low as low, Close as close, 
                Volume as volume, MA20 as ma20, MA50 as ma50, RSI as rsi, 
                MACD as macd, Signal as signal, UpperBand as upperBand, LowerBand as lowerBand,
                Company as company, Sector as sector
            FROM stock_metrics 
            WHERE UPPER(Ticker) = UPPER(?)
            ORDER BY Date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for ticker: {ticker}",
            )

        company = df["company"].iloc[0]
        sector = df["sector"].iloc[0]
        records = df.drop(columns=["company", "sector"]).to_dict(orient="records")

        return {
            "company": company,
            "sector": sector,
            "data": records,
        }

    def add_ticker(self, ticker: str) -> str:
        ticker = ticker.strip().upper()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker symbol is required.")

        try:
            result_msg = self.pipeline_tool.run(
                tickers=[ticker],
                parquet_path=self.parquet_path,
                sqlite_path=self.sqlite_path,
            )
            return result_msg
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))