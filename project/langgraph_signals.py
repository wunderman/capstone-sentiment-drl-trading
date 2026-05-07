"""
LangGraph Signal Cache
======================
Scans Capstone/reports/*_trade_recommendation_*.md markdown reports produced
by graph/trade_generation_pipeline.py and builds a DataFrame of
(ticker, date, recommendation, conviction, sentiment_score, justification).

Used by pipeline.py's gating layer to modify DRL trade actions.
"""
import os
import re
import glob
from typing import Optional, Dict, Any, List

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "Capstone", "reports")
CACHE_PATH = os.path.join(BASE_DIR, "dashboard_data", "langgraph_signals.csv")

VALID_RECS = {"BUY", "SELL", "HOLD"}

_TICKER_FILE_RE = re.compile(r"^([A-Z][A-Z0-9\.\-]*)_trade_recommendation", re.IGNORECASE)
_DATE_LINE_RES = [
    re.compile(r"\*\*Trade Date:\*\*\s*(\d{4}-\d{2}-\d{2})"),
    re.compile(r"\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})"),
    re.compile(r"^\s*Date\s*:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE),
]
_REC_RES = [
    re.compile(r"###\s*Final Investment Recommendation\s*:\s*\*?\*?\[?\s*(BUY|SELL|HOLD)\s*\]?", re.IGNORECASE),
    re.compile(r"Final Investment Recommendation\s*:\s*\*?\*?\[?\s*(BUY|SELL|HOLD)\s*\]?", re.IGNORECASE),
    re.compile(r"Final Trade Recommendation\s*:\s*\*?\*?\[?\s*(BUY|SELL|HOLD)\s*\]?", re.IGNORECASE),
]
_CONVICTION_RE = re.compile(r"Conviction Level\s*:\s*\*?\*?\s*(High|Medium|Low)", re.IGNORECASE)
_SENTIMENT_RES = [
    re.compile(r"\*\*Sentiment Score\s*:?\*\*\s*(-?\d+\.\d+|-?\d+)", re.IGNORECASE),
    re.compile(r"Sentiment Score\s*:\s*(-?\d+\.\d+|-?\d+)", re.IGNORECASE),
    re.compile(r"Final weighted sentiment score(?:\s*\(news\))?\s*:\s*\*?\*?\s*(-?\d+\.\d+|-?\d+)", re.IGNORECASE),
]


def _parse_report(path: str) -> Optional[Dict[str, Any]]:
    name = os.path.basename(path)
    m = _TICKER_FILE_RE.match(name)
    ticker = m.group(1).upper() if m else None

    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    if not ticker:
        # Fallback: find ticker in first heading
        m2 = re.search(r"Trade Generation Report:\s*([A-Z][A-Z0-9\.\-]*)", text)
        ticker = m2.group(1).upper() if m2 else None
    if not ticker:
        return None

    date_str = None
    for rx in _DATE_LINE_RES:
        mm = rx.search(text)
        if mm:
            date_str = mm.group(1)
            break
    if not date_str:
        return None

    rec = None
    for rx in _REC_RES:
        mm = rx.search(text)
        if mm:
            rec = mm.group(1).upper()
            break
    if rec not in VALID_RECS:
        return None

    conviction_match = _CONVICTION_RE.search(text)
    conviction = conviction_match.group(1).capitalize() if conviction_match else "Medium"

    sentiment_score = 0.0
    for rx in _SENTIMENT_RES:
        mm = rx.search(text)
        if mm:
            try:
                sentiment_score = float(mm.group(1))
                break
            except ValueError:
                pass

    return {
        "ticker": ticker,
        "date": date_str,
        "recommendation": rec,
        "conviction": conviction,
        "sentiment_score": sentiment_score,
        "source_file": name,
    }


def build_signals_cache(reports_dir: str = REPORTS_DIR, cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """Scan all trade-recommendation markdowns in reports_dir, write cache CSV, return DataFrame."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.isdir(reports_dir):
        df = pd.DataFrame(columns=["ticker", "date", "recommendation", "conviction", "sentiment_score", "source_file"])
        df.to_csv(cache_path, index=False)
        return df

    paths = sorted(glob.glob(os.path.join(reports_dir, "*trade_recommendation*.md")))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        parsed = _parse_report(p)
        if parsed is not None:
            rows.append(parsed)

    df = pd.DataFrame(rows, columns=["ticker", "date", "recommendation", "conviction", "sentiment_score", "source_file"])
    if not df.empty:
        df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def load_signals(cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """Load cached signals, rebuilding if the cache file is missing."""
    if not os.path.exists(cache_path):
        return build_signals_cache(cache_path=cache_path)
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return pd.DataFrame(columns=["ticker", "date", "recommendation", "conviction", "sentiment_score", "source_file"])
    return df


def get_signal(signals_df: pd.DataFrame, ticker: str, date) -> Optional[Dict[str, Any]]:
    """
    Return the most-recent signal for `ticker` on or before `date`.
    `date` can be a string (YYYY-MM-DD), pd.Timestamp, or datetime.
    Returns None if no signal exists.
    """
    if signals_df is None or signals_df.empty:
        return None

    ticker_u = str(ticker).upper()
    sub = signals_df[signals_df["ticker"].str.upper() == ticker_u].copy()
    if sub.empty:
        return None

    target = pd.to_datetime(date).normalize()
    sub["_dt"] = pd.to_datetime(sub["date"]).dt.normalize()
    eligible = sub[sub["_dt"] <= target].sort_values("_dt")
    if eligible.empty:
        return None

    row = eligible.iloc[-1]
    return {
        "ticker": row["ticker"],
        "date": str(row["date"]),
        "recommendation": str(row["recommendation"]).upper(),
        "conviction": str(row["conviction"]),
        "sentiment_score": float(row["sentiment_score"]) if pd.notna(row["sentiment_score"]) else 0.0,
        "source_file": str(row.get("source_file", "")),
    }


if __name__ == "__main__":
    df = build_signals_cache()
    print(f"Parsed {len(df)} LangGraph signal(s) → {CACHE_PATH}")
    if not df.empty:
        print(df.to_string(index=False))
