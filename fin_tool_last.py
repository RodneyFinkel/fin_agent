import os
import sqlite3
import time
import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests
from pydantic import BaseModel, Field
import warnings

# Silence internal yfinance/pandas deprecation warnings
warnings.filterwarnings("ignore")

class MarketDataPipelineInput(BaseModel):
    tickers: list[str] = Field(
        description="List of ticker symbols to process (e.g., ['AAPL', 'MSFT', 'GOOGL'])."
    )
    parquet_path: str = Field(
        default="stock_cache.parquet", 
        description="File path to save the master Parquet cache."
    )
    sqlite_path: str = Field(
        default="stock_database.db", 
        description="SQLite database file path for the NL2SQL agent."
    )

class MarketDataPipelineTool:
    name: str = "run_market_data_pipeline"
    description: str = (
        "Runs the end-to-end data pipeline for specified stock tickers. It first checks the local "
        "SQLite database to avoid redundant API calls. If new data is needed, it fetches history via "
        "yfinance using curl_cffi impersonation to bypass rate limits."
    )
    args_schema: type[BaseModel] = MarketDataPipelineInput

    def _check_cached_tickers(self, sqlite_path, tickers):
        """Checks which tickers already exist in the SQLite database."""
        if not os.path.exists(sqlite_path):
            return []
            
        try:
            conn = sqlite3.connect(sqlite_path)
            query = "SELECT DISTINCT Ticker FROM stock_metrics"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df['Ticker'].tolist()
        except Exception as e:
            print(f"[Cache Check] Error checking database: {e}")
            return []

    def _get_sp_sector_mapping(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        try:
            # We can use curl_cffi here as well just to be safe
            session = cffi_requests.Session(impersonate="chrome")
            response = session.get(url)
            tables = pd.read_html(response.text)
            sp_df = tables[0]
            
            mapping = {}
            for _, row in sp_df.iterrows():
                ticker = str(row['Symbol']).replace('.', '-')
                mapping[ticker] = {
                    'company': str(row['Security']),
                    'sector': str(row['GICS Sector'])
                }
            return mapping
        except Exception as e:
            print(f"Warning: Could not fetch Wikipedia sector mapping: {e}")
            return {}

    def _fetch_historical_yfinance_safely(self, ticker):
        """Fetches historical data using curl_cffi to spoof TLS fingerprints."""
        try:
            # Create a session that impersonates Google Chrome's network signature
            session = cffi_requests.Session(impersonate="chrome")
            
            # Pass the impersonating session directly into yfinance
            stock = yf.Ticker(ticker, session=session)
            df = stock.history(start="2019-01-01")
            
            if df is None or df.empty:
                return None, 'Unknown', 'Unknown'
            
            info = stock.info
            company = info.get('longName', ticker)
            sector = info.get('sector', 'Unknown')
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            return df.sort_index(), company, sector
        except Exception as e:
            print(f"yfinance error for {ticker}: {e}")
            return None, 'Unknown', 'Unknown'

    def _add_technical_indicators(self, data):
        prices = data.copy()
        if len(prices) < 50:
            return None
        
        prices['MA20'] = prices['Close'].rolling(window=20).mean()
        prices['MA50'] = prices['Close'].rolling(window=50).mean()
        
        delta = prices['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        prices['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = prices['Close'].ewm(span=12, adjust=False).mean()
        exp2 = prices['Close'].ewm(span=26, adjust=False).mean()
        prices['MACD'] = exp1 - exp2
        prices['Signal'] = prices['MACD'].ewm(span=9, adjust=False).mean()
        
        prices['20STD'] = prices['Close'].rolling(window=20).std()
        prices['UpperBand'] = prices['MA20'] + (prices['20STD'] * 2)
        prices['LowerBand'] = prices['MA20'] - (prices['20STD'] * 2)
        
        return prices.dropna()

    def run(self, tickers: list[str], parquet_path: str = "stock_cache.parquet", sqlite_path: str = "stock_database.db") -> str:
        
        # 1. Check which tickers we already have
        cached_tickers = self._check_cached_tickers(sqlite_path, tickers)
        tickers_to_fetch = [t.upper() for t in tickers if t.upper() not in cached_tickers]
        
        if not tickers_to_fetch:
            return f"Pipeline skipped API requests: All requested tickers ({tickers}) are already cached in the database."

        # 2. Only fetch metadata if we actually have tickers to download
        sector_mapping = self._get_sp_sector_mapping()
        new_records = []
        
        for idx, ticker in enumerate(tickers_to_fetch):
            if idx > 0:
                print(f"[Pipeline] Applying rate-limit delay for yfinance...")
                time.sleep(2)
                
            print(f"[Pipeline] Fetching fresh data for ticker: {ticker}...")
            
            raw_data, company, sector = self._fetch_historical_yfinance_safely(ticker)
            
            if raw_data is not None:
                processed_df = self._add_technical_indicators(raw_data)
                if processed_df is not None:
                    meta = sector_mapping.get(ticker, {'company': company, 'sector': sector})
                    
                    processed_df['Ticker'] = ticker
                    processed_df['Company'] = meta['company']
                    processed_df['Sector'] = meta['sector']
                    processed_df['Date'] = processed_df.index.strftime('%Y-%m-%d')
                    
                    new_records.append(processed_df.reset_index(drop=True))

        if not new_records:
            return "Pipeline executed, but no new valid data was retrieved for the requested tickers."

        # 3. Append the new data to the existing databases
        new_df = pd.concat(new_records, ignore_index=True)
        
        # Append to Parquet
        if os.path.exists(parquet_path):
            existing_parquet = pd.read_parquet(parquet_path)
            master_df = pd.concat([existing_parquet, new_df], ignore_index=True)
        else:
            master_df = new_df
        master_df.to_parquet(parquet_path, index=False)
        
        # Append to SQLite
        conn = sqlite3.connect(sqlite_path)
        new_df.to_sql("stock_metrics", conn, if_exists="append", index=False)
        conn.close()

        return (
            f"Pipeline successfully completed. "
            f"Fetched {len(tickers_to_fetch)} new tickers: {tickers_to_fetch}. "
            f"Added {len(new_df)} new rows. "
            f"Databases '{parquet_path}' and '{sqlite_path}' are fully synced."
        )

if __name__ == "__main__":
    pipeline_tool = MarketDataPipelineTool()
    
    # Run it once (it will probably skip because of your previous successful run)
    result = pipeline_tool.run(
        tickers=["PLTR"],
        parquet_path="stock_cache_fin.parquet",
        sqlite_path="stock_database_fin.db"
    )
    print(result)
    
    # # Try running it with a new ticker to see the impersonation in action
    # print("\n--- Testing a new ticker ---")
    # result2 = pipeline_tool.run(
    #     tickers=["TSLA"],
    #     parquet_path="stock_cache_fin.parquet",
    #     sqlite_path="stock_database_fin.db"
    # )
    # print(result2)