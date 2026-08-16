"""
Schema layer that sits *before* the LLM router.

It programmatically inspects the loaded ticker DataFrame and produces a compact,
LLM-safe description of the time-series database under investigation.
No full rows are ever sent to any language model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("SchemaLayer")


def build_df_schema(df: pd.DataFrame, ticker: str, max_sample_rows: int = 3) -> dict[str, Any]:
    """
    Build a rich but compact schema object for the router LLM.

    Contains:
    - column names + dtypes
    - null counts
    - numeric column stats (min / max / mean / std)
    - date range
    - a few sample rows (head + tail) so the model understands the shape
    - explicit contract reminder about the `time` column
    """
    if df is None or df.empty:
        return {
            "ticker": ticker,
            "row_count": 0,
            "error": "DataFrame is empty or missing",
        }

    work = df.copy()
    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"], errors="coerce")

    # Column inventory
    columns_info = []
    for col in work.columns:
        dtype = str(work[col].dtype)
        nulls = int(work[col].isna().sum())
        entry: dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "null_count": nulls,
        }
        if pd.api.types.is_numeric_dtype(work[col]):
            s = work[col].dropna()
            if len(s) > 0:
                entry["stats"] = {
                    "min": float(np.round(s.min(), 6)),
                    "max": float(np.round(s.max(), 6)),
                    "mean": float(np.round(s.mean(), 6)),
                    "std": float(np.round(s.std(), 6)) if len(s) > 1 else 0.0,
                }
        columns_info.append(entry)

    # Date range
    date_range = "Unknown"
    if "time" in work.columns and work["time"].notna().any():
        tmin = work["time"].min()
        tmax = work["time"].max()
        date_range = [str(tmin.date()) if hasattr(tmin, "date") else str(tmin),
                      str(tmax.date()) if hasattr(tmax, "date") else str(tmax)]

    # Tiny sample (head + tail) – still tiny so it stays safe for context
    sample_rows = []
    try:
        head = work.head(max_sample_rows).copy()
        tail = work.tail(max_sample_rows).copy()
        for part in (head, tail):
            for _, row in part.iterrows():
                sample = {}
                for c in work.columns:
                    val = row[c]
                    if pd.isna(val):
                        sample[c] = None
                    elif hasattr(val, "isoformat"):
                        sample[c] = val.isoformat()[:10]
                    elif isinstance(val, (np.floating, float)):
                        sample[c] = round(float(val), 4)
                    elif isinstance(val, (np.integer, int)):
                        sample[c] = int(val)
                    else:
                        sample[c] = str(val)[:80]
                sample_rows.append(sample)
    except Exception as e:
        logger.warning(f"Could not build sample rows: {e}")

    schema = {
        "ticker": ticker,
        "row_count": int(len(work)),
        "date_range": date_range,
        "columns": columns_info,
        "sample_rows": sample_rows[: max_sample_rows * 2],
        "contracts": {
            "time_column": "time",
            "time_note": "Always extract dates via str(df.loc[idx, 'time']). Never return raw integer indices.",
            "primary_price": "close",
            "available_indicators": [c for c in work.columns if c.lower() in
                                     {"rsi", "ma20", "ma50", "macd", "signal", "upperband", "lowerband", "volume"}],
        },
    }
    return schema


def schema_to_prompt_block(schema: dict[str, Any]) -> str:
    """Render the schema as a clean text block for the router system prompt."""
    if not schema or schema.get("row_count", 0) == 0:
        return "DATAFRAME SCHEMA: empty or unavailable."

    lines = [
        f"TICKER UNDER INVESTIGATION: {schema.get('ticker', 'UNKNOWN')}",
        f"ROWS: {schema.get('row_count')}",
        f"DATE RANGE: {schema.get('date_range')}",
        "",
        "COLUMNS (name | dtype | nulls | numeric stats):",
    ]
    for col in schema.get("columns", []):
        stats = col.get("stats")
        stats_str = ""
        if stats:
            stats_str = f"  min={stats['min']} max={stats['max']} mean={stats['mean']} std={stats['std']}"
        lines.append(f"  - {col['name']:12} | {col['dtype']:12} | nulls={col['null_count']}{stats_str}")

    lines.append("")
    lines.append("CONTRACTS:")
    contracts = schema.get("contracts", {})
    for k, v in contracts.items():
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("SAMPLE ROWS (first/last few – for shape awareness only):")
    for i, row in enumerate(schema.get("sample_rows", [])[:4]):
        lines.append(f"  [{i}] {row}")

    return "\n".join(lines)