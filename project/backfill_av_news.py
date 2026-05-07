"""
Alpha Vantage NEWS_SENTIMENT backfill
=====================================
Extends sentiment-drl-trading-main/datasets/dow30_monthly_news_sentiment.csv
forward in monthly chunks, per ticker, so pipeline.py can be run on a wider
END_DATE without losing sentiment coverage.

Usage
-----
    ALPHAVANTAGE_API_KEY=... python backfill_av_news.py
    # optional flags:
    python backfill_av_news.py --until 2026-04-21
    python backfill_av_news.py --tickers AAPL,MSFT --until 2026-04-21
    python backfill_av_news.py --max-calls 25   # free-tier safety cap

Notes
-----
* AV free tier = 25 calls/day. Each (ticker, month) is 1 call. With 27 tickers
  and ~12 months to backfill, you need ~324 calls → a premium key, or run over
  multiple days (the script resumes from wherever the CSV leaves off).
* Rate-limit hits stop the run cleanly; already-written rows survive.
* Dedupe key is (ticker, published_time, url).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from typing import List

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(
    BASE_DIR, "sentiment-drl-trading-main", "datasets", "dow30_monthly_news_sentiment.csv"
)

SCHEMA = [
    "ticker", "published_time", "title", "summary", "source", "url",
    "overall_sentiment_score", "overall_sentiment_label",
    "ticker_relevance_score", "ticker_sentiment_score", "ticker_sentiment_label",
]


def month_chunks(start: dt.date, end: dt.date):
    """Yield (time_from, time_to) tuples covering [start, end] in monthly windows."""
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        # last day of current month
        if cur.month == 12:
            nxt = dt.date(cur.year + 1, 1, 1)
        else:
            nxt = dt.date(cur.year, cur.month + 1, 1)
        hi = min(nxt - dt.timedelta(days=1), end)
        lo = max(cur, start)
        yield lo, hi
        cur = nxt


def fetch_month(ticker: str, lo: dt.date, hi: dt.date, api_key: str) -> List[dict]:
    tf = lo.strftime("%Y%m%dT0000")
    tt = hi.strftime("%Y%m%dT2359")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}"
        f"&time_from={tf}&time_to={tt}"
        f"&limit=1000&apikey={api_key}"
    )
    resp = requests.get(url, timeout=30).json()
    if "Note" in resp or "Information" in resp:
        raise RuntimeError(resp.get("Note") or resp.get("Information"))
    feed = resp.get("feed", []) or []

    rows = []
    for item in feed:
        pub = item.get("time_published", "")
        title = item.get("title", "")
        summary = item.get("summary", "")
        source = item.get("source", "")
        u = item.get("url", "")
        overall_score = item.get("overall_sentiment_score")
        overall_label = item.get("overall_sentiment_label")
        for ts in item.get("ticker_sentiment", []) or []:
            if str(ts.get("ticker", "")).upper() != ticker.upper():
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "published_time": pub,
                    "title": title,
                    "summary": summary,
                    "source": source,
                    "url": u,
                    "overall_sentiment_score": overall_score,
                    "overall_sentiment_label": overall_label,
                    "ticker_relevance_score": ts.get("relevance_score"),
                    "ticker_sentiment_score": ts.get("ticker_sentiment_score"),
                    "ticker_sentiment_label": ts.get("ticker_sentiment_label"),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=dt.date.today().isoformat(),
                    help="Backfill end date (YYYY-MM-DD, inclusive). Default: today.")
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated override. Default: unique tickers already in the CSV.")
    ap.add_argument("--max-calls", type=int, default=10_000,
                    help="Hard cap on API calls this run (free-tier safety).")
    ap.add_argument("--sleep", type=float, default=0.8,
                    help="Seconds between calls (premium: 0.15, free: 0.8).")
    args = ap.parse_args()

    api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("ERROR: ALPHAVANTAGE_API_KEY not set in environment / .env", file=sys.stderr)
        sys.exit(1)

    until = dt.date.fromisoformat(args.until)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading existing CSV: {CSV_PATH}")
    df_old = pd.read_csv(CSV_PATH)
    df_old["_dt"] = pd.to_datetime(df_old["published_time"], errors="coerce", format="%Y%m%dT%H%M%S")

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted(df_old["ticker"].dropna().unique().tolist())

    print(f"Tickers ({len(tickers)}): {tickers}")
    print(f"CSV max date: {df_old['_dt'].max()}")
    print(f"Backfilling up to: {until}")

    # Per-ticker resume point: day after this ticker's last row
    per_ticker_start = {}
    for t in tickers:
        tmax = df_old.loc[df_old["ticker"] == t, "_dt"].max()
        if pd.isna(tmax):
            per_ticker_start[t] = dt.date(2022, 3, 1)  # AV floor
        else:
            per_ticker_start[t] = (tmax + pd.Timedelta(days=1)).date()

    calls = 0
    new_rows: List[dict] = []
    try:
        for t in tickers:
            start = per_ticker_start[t]
            if start > until:
                print(f"  {t}: up to date")
                continue
            print(f"  {t}: {start} → {until}")
            for lo, hi in month_chunks(start, until):
                if calls >= args.max_calls:
                    print(f"  [cap] reached --max-calls={args.max_calls}, stopping")
                    raise KeyboardInterrupt
                try:
                    rows = fetch_month(t, lo, hi, api_key)
                except RuntimeError as e:
                    print(f"    [rate-limit] {e}")
                    raise KeyboardInterrupt
                except Exception as e:
                    print(f"    [error] {t} {lo}..{hi}: {e}")
                    rows = []
                calls += 1
                print(f"    {t} {lo}..{hi}: +{len(rows)} rows  (call {calls})")
                new_rows.extend(rows)
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("Stopping early; flushing collected rows.")

    if not new_rows:
        print("No new rows collected.")
        return

    df_new = pd.DataFrame(new_rows, columns=SCHEMA)
    df_old = df_old.drop(columns=["_dt"])
    combined = pd.concat([df_old, df_new], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "published_time", "url"], keep="first")
    print(f"Deduped {before - len(combined)} rows; final: {len(combined)}")

    # Sort for readability
    combined["_dt"] = pd.to_datetime(combined["published_time"], errors="coerce", format="%Y%m%dT%H%M%S")
    combined = combined.sort_values(["ticker", "_dt"]).drop(columns=["_dt"])

    combined.to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH} ({len(combined):,} rows)")
    print(f"Total API calls this run: {calls}")


if __name__ == "__main__":
    main()
