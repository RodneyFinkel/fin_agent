import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
from typing import Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AnalyticsEngine")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _prepare_df(chart_data: list[dict]) -> pd.DataFrame:
    logger.info(f"Received raw data for preparation. Record count: {len(chart_data)}")
    if not chart_data:
        logger.warning("chart_data is empty! No data received from database.")
        return pd.DataFrame()
    
    df = pd.DataFrame(chart_data)
    logger.info(f"Initial DataFrame columns from DB: {list(df.columns)}")
    # Map SQLite schema columns to the names expected by the analytics engine
    column_mapping = {
        "Date": "time",
        "Close": "close",
        "RSI": "rsi",
        "MA20": "ma20",
        "MA50": "ma50",
        "UpperBand": "upperBand",
        "LowerBand": "lowerBand"
    }
    df = df.rename(columns=column_mapping)
    logger.info(f"DataFrame columns after mapping: {list(df.columns)}")
    
    # Ensure time is parsed correctly and sort
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    
    df = df.sort_values("time").reset_index(drop=True)
    logger.info(f"DataFrame successfully prepared. Shape: {df.shape}")
    df["time"] = pd.to_datetime(df["time"])
    return df


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


# ─────────────────────────────────────────────
# Analytical pictures (numbers)
# ─────────────────────────────────────────────

def rsi_picture(df: pd.DataFrame) -> dict[str, Any]:
    if "rsi" not in df.columns:
        return {}
    latest = float(df["rsi"].iloc[-1])
    recent = df["rsi"].tail(20)

    regime = "neutral"
    if latest >= 70:
        regime = "overbought"
    elif latest <= 30:
        regime = "oversold"
    elif latest >= 55:
        regime = "bullish bias"
    elif latest <= 45:
        regime = "bearish bias"

    return {
        "latest_rsi": round(latest, 1),
        "regime": regime,
        "avg_rsi_20d": round(float(recent.mean()), 1),
    }


def ma_picture(df: pd.DataFrame) -> dict[str, Any]:
    if not {"ma20", "ma50", "close"}.issubset(df.columns):
        return {}
    row = df.iloc[-1]
    close, ma20, ma50 = float(row["close"]), float(row["ma20"]), float(row["ma50"])

    if close > ma20 > ma50:
        structure = "bullish alignment"
    elif close < ma20 < ma50:
        structure = "bearish alignment"
    else:
        structure = "mixed"

    return {
        "close": round(close, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "structure": structure,
        "dist_to_ma20_pct": round((close - ma20) / ma20 * 100, 2),
        "dist_to_ma50_pct": round((close - ma50) / ma50 * 100, 2),
    }


def bollinger_picture(df: pd.DataFrame) -> dict[str, Any]:
    if not {"upperBand", "lowerBand", "close", "ma20"}.issubset(df.columns):
        return {}
    row = df.iloc[-1]
    close = float(row["close"])
    upper = float(row["upperBand"])
    lower = float(row["lowerBand"])
    width = upper - lower
    pct_b = (close - lower) / width if width else 0.5

    state = "inside bands"
    if close > upper:
        state = "above upper band"
    elif close < lower:
        state = "below lower band"

    return {
        "percent_b": round(pct_b, 2),
        "band_width": round(width, 2),
        "state": state,
    }


def performance_picture(df: pd.DataFrame) -> dict[str, Any]:
    close = df["close"]
    latest = float(close.iloc[-1])

    def ret(days: int):
        if len(close) <= days:
            return None
        return round((latest / float(close.iloc[-days - 1]) - 1) * 100, 2)

    return {
        "return_5d": ret(5),
        "return_20d": ret(20),
        "return_60d": ret(60),
    }


# ─────────────────────────────────────────────
# Charts (base64)
# ─────────────────────────────────────────────

def chart_rsi(df: pd.DataFrame, lookback: int = 120) -> str:
    data = df.tail(lookback)
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#1f2937")
    ax.set_facecolor("#1f2937")

    ax.plot(data["time"], data["rsi"], color="#60a5fa", linewidth=1.6, label="RSI")
    ax.axhline(70, color="#ef4444", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(30, color="#22c55e", linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(50, color="#6b7280", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.fill_between(data["time"], 70, 100, color="#ef4444", alpha=0.08)
    ax.fill_between(data["time"], 0, 30, color="#22c55e", alpha=0.08)

    ax.set_ylim(0, 100)
    ax.set_title("RSI (14)", color="#e5e7eb", fontsize=11, pad=8)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, facecolor="#374151", edgecolor="none", labelcolor="#e5e7eb")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.15, color="#6b7280")

    return _fig_to_base64(fig)


def chart_ma_structure(df: pd.DataFrame, lookback: int = 120) -> str:
    data = df.tail(lookback)
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#1f2937")
    ax.set_facecolor("#1f2937")

    ax.plot(data["time"], data["close"], color="#e5e7eb", linewidth=1.5, label="Close")
    ax.plot(data["time"], data["ma20"], color="#f59e0b", linewidth=1.3, label="MA20")
    ax.plot(data["time"], data["ma50"], color="#3b82f6", linewidth=1.3, label="MA50")

    # Simple alignment shading
    bull = (data["close"] > data["ma20"]) & (data["ma20"] > data["ma50"])
    bear = (data["close"] < data["ma20"]) & (data["ma20"] < data["ma50"])
    ax.fill_between(data["time"], data["close"].min() * 0.98, data["close"].max() * 1.02,
                    where=bull, color="#22c55e", alpha=0.07, interpolate=True)
    ax.fill_between(data["time"], data["close"].min() * 0.98, data["close"].max() * 1.02,
                    where=bear, color="#ef4444", alpha=0.07, interpolate=True)

    ax.set_title("Price + Moving Average Structure", color="#e5e7eb", fontsize=11, pad=8)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, facecolor="#374151", edgecolor="none", labelcolor="#e5e7eb")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.15, color="#6b7280")

    return _fig_to_base64(fig)


def chart_bollinger(df: pd.DataFrame, lookback: int = 120) -> str:
    data = df.tail(lookback)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), facecolor="#1f2937",
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    for ax in (ax1, ax2):
        ax.set_facecolor("#1f2937")

    ax1.plot(data["time"], data["close"], color="#e5e7eb", linewidth=1.4, label="Close")
    ax1.plot(data["time"], data["ma20"], color="#f59e0b", linewidth=1.1, label="MA20")
    ax1.plot(data["time"], data["upperBand"], color="#6b7280", linewidth=1, linestyle="--")
    ax1.plot(data["time"], data["lowerBand"], color="#6b7280", linewidth=1, linestyle="--")
    ax1.fill_between(data["time"], data["lowerBand"], data["upperBand"], color="#3b82f6", alpha=0.1)

    ax1.set_title("Bollinger Bands", color="#e5e7eb", fontsize=11, pad=8)
    ax1.tick_params(colors="#9ca3af", labelsize=8)
    ax1.legend(loc="upper left", fontsize=8, facecolor="#374151", edgecolor="none", labelcolor="#e5e7eb")
    ax1.grid(True, alpha=0.15, color="#6b7280")

    width = data["upperBand"] - data["lowerBand"]
    ax2.plot(data["time"], width, color="#a78bfa", linewidth=1.3)
    ax2.set_ylabel("Width", color="#9ca3af", fontsize=8)
    ax2.tick_params(colors="#9ca3af", labelsize=8)
    ax2.grid(True, alpha=0.15, color="#6b7280")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.autofmt_xdate(rotation=30)

    return _fig_to_base64(fig)


def chart_returns(df: pd.DataFrame) -> str:
    close = df["close"]
    latest = float(close.iloc[-1])

    periods = {"5D": 5, "20D": 20, "60D": 60}
    labels, values, colors = [], [], []

    for name, days in periods.items():
        if len(close) > days:
            ret = (latest / float(close.iloc[-days - 1]) - 1) * 100
            labels.append(name)
            values.append(ret)
            colors.append("#22c55e" if ret >= 0 else "#ef4444")

    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="#1f2937")
    ax.set_facecolor("#1f2937")

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="none")
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_title("Recent Returns (%)", color="#e5e7eb", fontsize=11, pad=8)
    ax.tick_params(colors="#9ca3af", labelsize=9)
    ax.grid(True, axis="y", alpha=0.15, color="#6b7280")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.4 if val >= 0 else -1.2),
                f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                color="#e5e7eb", fontsize=9)

    return _fig_to_base64(fig)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def build_analytical_picture(chart_data: list[dict], include_charts: bool = True) -> dict[str, Any]:
    """
    Main entry point.
    Returns numbers + optional base64 charts.
    """
    
    if not chart_data:
        return {}

    df = _prepare_df(chart_data)

    result = {
        "rsi": rsi_picture(df),
        "moving_averages": ma_picture(df),
        "bollinger": bollinger_picture(df),
        "performance": performance_picture(df),
        "as_of": str(df["time"].iloc[-1].date()),
        "rows_analyzed": len(df),
    }

    if include_charts:
        result["charts"] = {
            "rsi": chart_rsi(df),
            "ma_structure": chart_ma_structure(df),
            "bollinger": chart_bollinger(df),
            "returns": chart_returns(df),
        }

    return result