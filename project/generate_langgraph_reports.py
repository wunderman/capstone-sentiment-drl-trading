"""
Batch-generate LangGraph trade-recommendation reports for the DRL test window
so the gate layer actually has signals to override with.

Iterates (ticker × month), invokes the trade-generation graph, writes one
markdown per report into Capstone/reports/. langgraph_signals.py picks them up
automatically next time pipeline.py runs.

Usage:
    python generate_langgraph_reports.py
    python generate_langgraph_reports.py --tickers AAPL,MSFT,NVDA
    python generate_langgraph_reports.py --start 2025-01-01 --end 2026-04-21
    python generate_langgraph_reports.py --freq MS   # month-start dates
    python generate_langgraph_reports.py --max-reports 30  # cost cap for testing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import pipeline as pl  # noqa: E402  (TEST_START / TEST_END / EXPANDED_TICKERS)

from Capstone.graph.trade_generation_pipeline import (  # noqa: E402
    build_trade_generation_graph,
    extract_final_trade_recommendation,
)

REPORTS_DIR = BASE_DIR / "Capstone" / "reports"


def existing_report_keys() -> set[tuple[str, str]]:
    """Set of (TICKER, YYYY-MM-DD) pairs already on disk so we skip re-runs."""
    keys: set[tuple[str, str]] = set()
    if not REPORTS_DIR.exists():
        return keys
    for p in REPORTS_DIR.glob("*_trade_recommendation_*.md"):
        name = p.name
        try:
            # <TICKER>_trade_recommendation_<YYYY-MM-DD>[_timestamp].md
            head, _, rest = name.partition("_trade_recommendation_")
            date_part = rest.split("_")[0].replace(".md", "")
            keys.add((head.upper(), date_part))
        except Exception:
            continue
    return keys


def generate_one(agent, ticker: str, trade_date: str) -> dict | None:
    initial_state = {
        "messages": [HumanMessage(content=f"Analyze {ticker} and provide investment recommendation")],
        "trade_date": trade_date,
        "company_of_interest": ticker,
        "fundamentals_report": "",
        "sentiment_report": "",
        "sentiment_score": 0.0,
        "csv_path": "",
        "investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": "", "count": 0,
        },
        "final_recommendation": "",
    }
    try:
        result = agent.invoke(initial_state)
    except Exception as e:
        print(f"    ERROR {ticker} {trade_date}: {e}")
        return None
    return result


def write_report(ticker: str, trade_date: str, result: dict) -> Path:
    rec = extract_final_trade_recommendation(result.get("final_recommendation", ""))
    sentiment_score = result.get("sentiment_score", 0.0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"{ticker}_trade_recommendation_{trade_date}_{ts}.md"
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    body = f"""# Trade Generation Report: {ticker}

Ticker: {ticker}
Date: {trade_date}

**Trade Date:** {trade_date}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Sentiment Score:** {sentiment_score}

---

## Fundamentals Analysis
{result.get('fundamentals_report', 'Not available')}

---

## Sentiment Analysis
{result.get('sentiment_report', 'Not available')}

---

## Debate History
{result.get('investment_debate_state', {}).get('history', 'Not available')}

---

### Final Investment Recommendation: **{rec}**

{result.get('final_recommendation', 'Not available')}
"""
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None,
                    help="Comma-separated override. Default: pl.EXPANDED_TICKERS (all).")
    ap.add_argument("--start", default=pl.TEST_START, help="Report window start (YYYY-MM-DD).")
    ap.add_argument("--end", default=pl.TEST_END, help="Report window end (YYYY-MM-DD).")
    ap.add_argument("--freq", default="MS",
                    help="pandas date_range freq: MS (month-start, default), W-MON, 2W, etc.")
    ap.add_argument("--max-reports", type=int, default=10_000,
                    help="Hard cap this run (cost safety).")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Seconds between reports to avoid OpenRouter bursts.")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip (ticker, date) pairs already written.")
    ap.add_argument("--dow-only", action="store_true", default=False,
                    help="Use only 27 DRL tickers (skip expanded universe).")
    args = ap.parse_args()

    # The 27 tickers with Alpha Vantage sentiment coverage — the only universe
    # the LangGraph sentiment analyst can actually reason about.
    SENTIMENT_COVERED = [
        'AAPL', 'ABBV', 'ACN', 'ADBE', 'AMZN', 'AVGO', 'BAC', 'COST', 'CRM', 'CVX',
        'GOOG', 'HD', 'KO', 'LLY', 'MA', 'META', 'MSFT', 'NVDA', 'ORCL', 'PEP',
        'PG', 'TMO', 'TSLA', 'UNH', 'V', 'WMT', 'XOM',
    ]

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.dow_only:
        tickers = SENTIMENT_COVERED
    else:
        tickers = SENTIMENT_COVERED  # default to the 27 covered tickers — reports on uncovered names have no sentiment to analyze

    dates = [d.strftime("%Y-%m-%d")
             for d in pd.date_range(start=args.start, end=args.end, freq=args.freq)]

    total = len(tickers) * len(dates)
    print(f"Tickers ({len(tickers)}): {tickers[:8]}{'...' if len(tickers) > 8 else ''}")
    print(f"Dates ({len(dates)}): {dates[:6]}{'...' if len(dates) > 6 else ''}")
    print(f"Potential reports: {total} (hard cap: {args.max_reports})")
    print(f"Reports dir: {REPORTS_DIR}")

    existing = existing_report_keys() if args.skip_existing else set()
    if existing:
        print(f"Skipping {len(existing)} existing (ticker, date) pairs")

    print("Building LangGraph pipeline...")
    agent, _memory = build_trade_generation_graph()

    done = 0
    skipped = 0
    started = time.time()
    try:
        for t in tickers:
            for d in dates:
                if done >= args.max_reports:
                    print(f"[cap] reached --max-reports={args.max_reports}, stopping")
                    raise KeyboardInterrupt
                if (t, d) in existing:
                    skipped += 1
                    continue

                t0 = time.time()
                result = generate_one(agent, t, d)
                if result is None:
                    continue
                path = write_report(t, d, result)
                done += 1
                dt = time.time() - t0
                rec = extract_final_trade_recommendation(result.get("final_recommendation", ""))
                print(f"  [{done}/{total}] {t} {d}: {rec}  ({dt:.1f}s)  → {path.name}")
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - started
    print(f"\nDONE. Wrote {done} report(s), skipped {skipped}. Elapsed: {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
