import sqlite3
import pandas as pd


def get_tables(db_path):
  """Helper function to list all tables in a SQLite database."""
  try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE"
        " 'sqlite_%';"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables
  except sqlite3.OperationalError:
    return []


def fuse_dbs_with_pandas(
    stock_database_fin1,
    stock_database_fin2,
    fused_database_path,
    table_name="stock_metrics",
):
  tables1 = get_tables(stock_database_fin1)
  tables2 = get_tables(stock_database_fin2)

  print(f"Tables in {stock_database_fin1}: {tables1}")
  print(f"Tables in {stock_database_fin2}: {tables2}")

  dfs = []

  # Load from DB1 if the table exists
  if table_name in tables1:
    conn1 = sqlite3.connect(stock_database_fin1)
    df1 = pd.read_sql(f"SELECT * FROM {table_name}", conn1)
    conn1.close()
    dfs.append(df1)
    print(f"Loaded {len(df1)} rows from {stock_database_fin1}")
  else:
    print(
        f"Warning: Table '{table_name}' not found in {stock_database_fin1} (or"
        " database is empty)."
    )

  # Load from DB2 if the table exists
  if table_name in tables2:
    conn2 = sqlite3.connect(stock_database_fin2)
    df2 = pd.read_sql(f"SELECT * FROM {table_name}", conn2)
    conn2.close()
    dfs.append(df2)
    print(f"Loaded {len(df2)} rows from {stock_database_fin2}")
  else:
    print(f"Warning: Table '{table_name}' not found in {stock_database_fin2}.")

  if not dfs:
    raise ValueError(
        f"Table '{table_name}' could not be found in either database!"
    )

  # Combine available DataFrames
  combined_df = pd.concat(dfs, ignore_index=True)

  # Drop duplicates if a ticker and date combination appears multiple times
  if "Date" in combined_df.columns and "Ticker" in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset=["Date", "Ticker"])

  # Write the fused data to a new SQLite database
  output_conn = sqlite3.connect(fused_database_path)
  combined_df.to_sql(table_name, output_conn, if_exists="replace", index=False)
  output_conn.close()

  print(
      f"Successfully merged {len(combined_df)} total rows into"
      f" {fused_database_path} using table '{table_name}'"
  )


if __name__ == "__main__":
  fuse_dbs_with_pandas(
      "fused_database3.db", "stock_database_fin4.db", "fused_database4.db"
  )