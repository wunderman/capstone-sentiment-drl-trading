"""
Integrated Pipeline: Sentiment + DRL Trading
=============================================
Downloads OHLCV data, merges with sentiment, loads pre-trained agents,
runs backtesting, and saves results for the dashboard.
"""
import os, sys, datetime, warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

import pandas as pd
import numpy as np
import torch

# FinRL imports
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, get_baseline
from finrl import config

# Stable Baselines
from stable_baselines3 import PPO, A2C, DDPG, TD3

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRL_DIR = os.path.join(BASE_DIR, "sentiment-drl-trading-main")
OUTPUT_DIR = os.path.join(BASE_DIR, "dashboard_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2022-03-02"
END_DATE = "2026-04-21"
TRAIN_START = START_DATE
TRAIN_END = "2025-01-01"
TEST_START = "2025-01-01"
TEST_END = END_DATE
INITIAL_AMOUNT = 1_000_000

# Risk-free rate that idle cash earns each day (≈ 2025 SOFR / 3-mo T-bill avg).
# Without this, strategies that hold large cash buffers are unfairly compared
# against fully-invested benchmarks — cash had a real yield in this regime.
RISK_FREE_RATE = 0.045
DAILY_CASH_YIELD = (1 + RISK_FREE_RATE) ** (1 / 252) - 1
# Trailing stop = peak - TRAILING_ATR_MULT * ATR(14). Per-strategy, because
# RSI dip-buys want fast exits if the bounce fails (mean-reversion thesis is
# either right in days or wrong), while trend-following SMA-based names need
# wider stops so the trend can breathe. Sweep: 2.5/2.5 vs 2.5/3.0 vs 3.0/3.0
# — the asymmetric (2.5 RSI, 3.0 SMA) gave the best ensemble Sharpe.
TRAILING_ATR_MULT_RSI = 2.5
TRAILING_ATR_MULT_TREND = 3.0
TRAILING_ATR_MULT = 2.5  # legacy default for ensemble (uses RSI as primary voter)
# Profit ladder: scale out 1/3 at +PROFIT_T1, another 1/3 at +PROFIT_T2.
PROFIT_T1 = 0.15
PROFIT_T2 = 0.30
# Momentum gate for RSI dip-buys: skip names whose 63-day return is negative
# (avoids catching falling knives — the classic RSI mean-reversion failure).
MOMENTUM_LOOKBACK = 63

INDICATORS = config.INDICATORS  # ['macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma']

# Expanded ticker universe for rule-based strategies
EXPANDED_TICKERS = sorted([
    # Original 27 (from Alpha Vantage sentiment dataset)
    'AAPL','ABBV','ACN','ADBE','AMZN','AVGO','BAC','COST','CRM','CVX','GOOG','HD',
    'KO','LLY','MA','META','MSFT','NVDA','ORCL','PEP','PG','TMO','TSLA','UNH','V','WMT','XOM',
    # DJIA additions
    'AMGN','AXP','BA','CAT','CSCO','DIS','DOW','GS','HON','IBM','INTC','JNJ','JPM',
    'MCD','MMM','MRK','NKE','TRV','VZ',
    # Major S&P 500 / high-sentiment-coverage names
    'NFLX','AMD','QCOM','TXN','PYPL','UBER','PLTR','COIN','PANW','CRWD','ARM',
    'ABNB','SQ','SHOP','SNOW','ZS','DDOG','NET','RIVN','LCID',
])

# ---------------------------------------------------------------------------
# STEP 1 — Prepare sentiment data (Alpha Vantage news-based, from notebook)
# ---------------------------------------------------------------------------
def prepare_sentiment_data():
    """Load and clean the Alpha Vantage news sentiment CSV used by the DRL bot."""
    # The notebook uses a HuggingFace-hosted CSV: Natty6418/dow30_monthly_news_sentiment
    # Try local first, then download
    local_path = os.path.join(DRL_DIR, "datasets", "dow30_monthly_news_sentiment.csv")
    if not os.path.exists(local_path):
        print("Downloading sentiment data from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("Natty6418/dow30_monthly_news_sentiment",
                          data_files="dow30_monthly_news_sentiment.csv", split="train")
        df_sentiment = pd.DataFrame(ds)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df_sentiment.to_csv(local_path, index=False)
    else:
        df_sentiment = pd.read_csv(local_path)

    # Clean — same steps as notebook
    df_sentiment.drop(columns=["title", "summary", "source", "url"], inplace=True, errors="ignore")
    df_sentiment["date"] = pd.to_datetime(df_sentiment["published_time"]).dt.date
    df_sentiment.drop(columns=["published_time"], inplace=True, errors="ignore")

    df_sentiment["weighted_score"] = (
        df_sentiment["ticker_sentiment_score"] * df_sentiment["ticker_relevance_score"]
    )

    agg_df = (
        df_sentiment
        .groupby(["date", "ticker"])
        .agg(
            raw_avg_sentiment=("ticker_sentiment_score", "mean"),
            total_relevance=("ticker_relevance_score", "sum"),
            weighted_sum=("weighted_score", "sum"),
            article_count=("ticker_sentiment_score", "count"),
        )
        .reset_index()
    )
    agg_df["weighted_avg_sentiment"] = agg_df["weighted_sum"] / agg_df["total_relevance"]
    agg_df["weighted_avg_sentiment"] = agg_df["weighted_avg_sentiment"].fillna(0)
    agg_df.sort_values(by=["date", "ticker"], inplace=True)

    # Fill missing dates
    full_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D").date
    combos = pd.MultiIndex.from_product(
        [agg_df["ticker"].unique(), full_dates], names=["ticker", "date"]
    ).to_frame(index=False)
    merged = pd.merge(combos, agg_df, how="left", on=["ticker", "date"])
    merged["date"] = merged["date"] + pd.Timedelta(days=1)
    merged = merged.fillna(0).sort_values(["date", "ticker"])

    print(f"  Sentiment data: {len(merged)} rows, {merged['ticker'].nunique()} tickers")
    return merged


# ---------------------------------------------------------------------------
# STEP 1b — Collect Telegram sentiment via FinBERT
# ---------------------------------------------------------------------------
def collect_telegram_sentiment(tickers):
    """
    Collect social media sentiment from Telegram (+ StockTwits via Apify),
    run FinBERT analysis, return DataFrame matching Alpha Vantage format.
    Falls back to web-scraping if Apify is unavailable.
    """
    cache = os.path.join(OUTPUT_DIR, "telegram_sentiment.csv")
    if os.path.exists(cache):
        print("  Loading cached Telegram sentiment...")
        return pd.read_csv(cache)

    # Add Capstone/ to path first so its social_media_sentiment takes priority
    capstone_path = os.path.join(BASE_DIR, "Capstone")
    if capstone_path in sys.path:
        sys.path.remove(capstone_path)
    sys.path.insert(0, capstone_path)

    from social_media_sentiment.sentiment_analyzer import SentimentAnalyzer

    import re
    import yfinance as yf
    ticker_set = set(t.upper() for t in tickers)
    all_posts = []  # list of {"ticker": ..., "text": ..., "date": date}

    # Company name → ticker mapping
    company_map = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN', 'nvidia': 'NVDA',
        'tesla': 'TSLA', 'meta platforms': 'META', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'jpmorgan': 'JPM', 'goldman sachs': 'GS', 'disney': 'DIS', 'boeing': 'BA',
        'walmart': 'WMT', 'coca-cola': 'KO', 'nike': 'NKE', 'intel': 'INTC',
        'cisco': 'CSCO', 'salesforce': 'CRM', 'chevron': 'CVX', 'merck': 'MRK',
        'caterpillar': 'CAT', 'home depot': 'HD', 'amgen': 'AMGN', 'verizon': 'VZ',
        'honeywell': 'HON', 'mcdonald': 'MCD',
    }

    # --- Source 1: m-ric/financial-news-2024 (timestamped, Apr-Oct 2024) ---
    try:
        from datasets import load_dataset
        print("  Loading m-ric/financial-news-2024 from HuggingFace...")
        ds = load_dataset('m-ric/financial-news-2024', split='train')
        hf_count = 0
        for row in ds:
            title = row.get('title', '')
            date_val = row.get('date')
            if not title or not date_val:
                continue
            # Convert date
            if hasattr(date_val, 'date'):
                d = date_val if isinstance(date_val, datetime.date) else date_val.date()
            else:
                d = pd.to_datetime(str(date_val)).date()

            # Match tickers via uppercase words and company names
            found = set()
            for t in re.findall(r'\b([A-Z]{2,5})\b', title):
                if t in ticker_set:
                    found.add(t)
            text_lower = title.lower()
            for name, tic in company_map.items():
                if name in text_lower and tic in ticker_set:
                    found.add(tic)
            for tic in found:
                all_posts.append({"ticker": tic, "text": title[:512], "date": d})
                hf_count += 1
        print(f"  HuggingFace news: {hf_count} ticker-posts, {len(set(p['ticker'] for p in all_posts))} tickers")
    except Exception as e:
        print(f"  HuggingFace dataset failed: {e}")

    # --- Source 2: yfinance news (live, per-ticker, recent) ---
    try:
        print("  Collecting yfinance news for each ticker...")
        yf_count = 0
        for tic in tickers:
            try:
                t = yf.Ticker(tic)
                news = t.news or []
                for n in news:
                    c = n.get('content', {})
                    title = c.get('title', '')
                    pub = c.get('pubDate', '')
                    if not title or not pub:
                        continue
                    d = pd.to_datetime(pub).date()
                    all_posts.append({"ticker": tic, "text": title[:512], "date": d})
                    yf_count += 1
            except Exception:
                continue
        print(f"  yfinance news: {yf_count} articles across {len(tickers)} tickers")
    except Exception as e:
        print(f"  yfinance news failed: {e}")

    # --- Source 3: Telegram public channels (live, web scraping) ---
    try:
        from social_media_sentiment.collectors.telegram_collector import TelegramCollector
        print("  Scraping public Telegram channels...")
        tg = TelegramCollector(use_llm=False)
        msgs = tg.scrape_all_messages(limit=2000)
        tg_matched = 0
        for msg in msgs:
            text = msg.get("text", "")
            detected = msg.get("detected_tickers", [])
            created = msg.get("created_utc", datetime.datetime.now())
            for tic in detected:
                if tic.upper() in ticker_set:
                    d = created.date() if hasattr(created, "date") else created
                    all_posts.append({"ticker": tic.upper(), "text": text[:512], "date": d})
                    tg_matched += 1
        print(f"  Telegram: {tg_matched} matched posts from {len(msgs)} messages")
    except Exception as e:
        print(f"  Telegram scrape failed: {e}")

    # --- Source 4: NewsCollector (Yahoo Finance RSS + Google News + Finnhub) ---
    try:
        from social_media_sentiment.collectors.news_collector import NewsCollector
        print("  Fetching news via NewsCollector (Yahoo+Google+Finnhub)...")
        nc = NewsCollector()
        news_count = 0
        for tic in tickers:
            try:
                articles = nc.search_ticker(tic, limit=100, hours_back=72)
                for art in articles:
                    text = art.get("full_text") or art.get("text") or ""
                    created = art.get("created_utc")
                    if not text or not created:
                        continue
                    d = created.date() if hasattr(created, "date") else created
                    all_posts.append({"ticker": tic.upper(), "text": text[:512], "date": d})
                    news_count += 1
            except Exception:
                continue
        print(f"  NewsCollector: {news_count} articles across {len(tickers)} tickers")
    except Exception as e:
        print(f"  NewsCollector failed: {e}")

    if not all_posts:
        print("  WARNING: No social media posts collected")
        return pd.DataFrame()

    # --- Run FinBERT on all collected texts ---
    print(f"  Loading FinBERT for sentiment analysis on {len(all_posts)} posts...")
    analyzer = SentimentAnalyzer()

    texts = [p["text"] for p in all_posts]
    sentiments = analyzer.analyze_batch(texts, batch_size=32)

    rows = []
    for post, sent in zip(all_posts, sentiments):
        rows.append({
            "ticker": post["ticker"],
            "date": post["date"],
            "ticker_sentiment_score": sent["score"],
            "ticker_relevance_score": sent["confidence"],
        })

    df_sm = pd.DataFrame(rows)
    if df_sm.empty:
        return df_sm

    # Aggregate per (date, ticker) — time-aligned!
    df_sm["date"] = pd.to_datetime(df_sm["date"]).dt.date
    df_sm["weighted_score"] = df_sm["ticker_sentiment_score"] * df_sm["ticker_relevance_score"]

    agg = (
        df_sm.groupby(["date", "ticker"])
        .agg(
            raw_avg_sentiment=("ticker_sentiment_score", "mean"),
            total_relevance=("ticker_relevance_score", "sum"),
            weighted_sum=("weighted_score", "sum"),
            article_count=("ticker_sentiment_score", "count"),
        )
        .reset_index()
    )
    agg["weighted_avg_sentiment"] = agg["weighted_sum"] / agg["total_relevance"]
    agg["weighted_avg_sentiment"] = agg["weighted_avg_sentiment"].fillna(0)

    n_tickers = agg["ticker"].nunique()
    n_dates = agg["date"].nunique()
    print(f"  Social media sentiment: {len(agg)} rows, {n_tickers} tickers, {n_dates} unique dates")
    print(f"  Date range: {agg['date'].min()} to {agg['date'].max()}")
    agg.to_csv(cache, index=False)
    return agg


# ---------------------------------------------------------------------------
# STEP 1c — Alpha Vantage LIVE news sentiment (opt-in via USE_LIVE_ALPHAVANTAGE=1)
# ---------------------------------------------------------------------------
def fetch_alphavantage_live_sentiment(tickers, days_back=90, max_tickers=10, cache_ttl_hours=12):
    """
    Pull live NEWS_SENTIMENT from Alpha Vantage for each ticker and aggregate
    per (date, ticker) to match the sentiment schema. Returns an empty df when
    USE_LIVE_ALPHAVANTAGE env flag is off.

    Free tier = 25 calls/day — defaults to max_tickers=10 and a 12h cache.
    """
    if os.getenv("USE_LIVE_ALPHAVANTAGE", "0") != "1":
        print("  [Alpha Vantage live] skipped — set USE_LIVE_ALPHAVANTAGE=1 to enable")
        return pd.DataFrame()

    api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        print("  [Alpha Vantage live] no ALPHAVANTAGE_API_KEY — skipped")
        return pd.DataFrame()

    cache = os.path.join(OUTPUT_DIR, "alphavantage_live_sentiment.csv")
    if os.path.exists(cache):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(cache))
        if (datetime.datetime.now() - mtime).total_seconds() < cache_ttl_hours * 3600:
            print(f"  [Alpha Vantage live] using cache ({mtime:%Y-%m-%d %H:%M})")
            return pd.read_csv(cache)

    import time
    import requests

    time_from = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y%m%dT0000")
    subset = list(tickers)[:max_tickers]
    print(f"  [Alpha Vantage live] fetching NEWS_SENTIMENT for {len(subset)} tickers (last {days_back}d)...")

    rows = []
    for tic in subset:
        url = ("https://www.alphavantage.co/query"
               f"?function=NEWS_SENTIMENT&tickers={tic}&time_from={time_from}"
               f"&limit=200&apikey={api_key}")
        try:
            resp = requests.get(url, timeout=30).json()
        except Exception as exc:
            print(f"    {tic}: request failed ({exc})")
            continue

        if "Note" in resp or "Information" in resp:
            print(f"    {tic}: rate-limit hit — stopping. {resp.get('Note') or resp.get('Information')}")
            break

        feed = resp.get("feed", [])
        for item in feed:
            pub = item.get("time_published", "")
            if len(pub) < 8:
                continue
            d = datetime.datetime.strptime(pub[:8], "%Y%m%d").date()
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() != tic.upper():
                    continue
                try:
                    score = float(ts.get("ticker_sentiment_score", 0))
                    rel = float(ts.get("relevance_score", 0))
                except (TypeError, ValueError):
                    continue
                rows.append({"ticker": tic, "date": d,
                             "ticker_sentiment_score": score,
                             "ticker_relevance_score": rel})
        print(f"    {tic}: {len(feed)} articles")
        time.sleep(0.8)  # avoid per-minute rate limit

    if not rows:
        print("  [Alpha Vantage live] no rows collected")
        df_empty = pd.DataFrame()
        return df_empty

    df = pd.DataFrame(rows)
    df["weighted_score"] = df["ticker_sentiment_score"] * df["ticker_relevance_score"]
    agg = (df.groupby(["date", "ticker"])
             .agg(raw_avg_sentiment=("ticker_sentiment_score", "mean"),
                  total_relevance=("ticker_relevance_score", "sum"),
                  weighted_sum=("weighted_score", "sum"),
                  article_count=("ticker_sentiment_score", "count"))
             .reset_index())
    agg["weighted_avg_sentiment"] = (agg["weighted_sum"] / agg["total_relevance"]).fillna(0)
    print(f"  [Alpha Vantage live] {len(agg)} (date,ticker) rows across {agg['ticker'].nunique()} tickers")
    agg.to_csv(cache, index=False)
    return agg


# ---------------------------------------------------------------------------
# STEP 1d — Apify multi-platform social scrape (opt-in via USE_APIFY=1)
# ---------------------------------------------------------------------------
def fetch_apify_social_sentiment(tickers, max_tickers=5, limit_per_platform=50, hours_back=72):
    """
    Use ApifyCollector.search_ticker() to pull posts from Reddit/Telegram/YouTube/
    StockTwits (Twitter skipped by default — highest credit cost), score them with
    FinBERT, and aggregate per (date, ticker).

    Paid credits — defaults to 5 tickers. Opt-in via USE_APIFY=1.
    """
    if os.getenv("USE_APIFY", "0") != "1":
        print("  [Apify] skipped — set USE_APIFY=1 to enable")
        return pd.DataFrame()

    if not os.getenv("APIFY_API_KEY"):
        print("  [Apify] no APIFY_API_KEY — skipped")
        return pd.DataFrame()

    cache = os.path.join(OUTPUT_DIR, "apify_social_sentiment.csv")
    if os.path.exists(cache):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(cache))
        if (datetime.datetime.now() - mtime).total_seconds() < 24 * 3600:
            print(f"  [Apify] using cache ({mtime:%Y-%m-%d %H:%M})")
            return pd.read_csv(cache)

    # apify_collector lives in the WORKSPACE-ROOT social_media_sentiment pkg,
    # not the one under Capstone/. Evict any cached version and force root priority.
    for mod in [m for m in list(sys.modules) if m.startswith("social_media_sentiment")]:
        del sys.modules[mod]
    capstone_path = os.path.join(BASE_DIR, "Capstone")
    if capstone_path in sys.path:
        sys.path.remove(capstone_path)
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    try:
        from social_media_sentiment.collectors.apify_collector import ApifyCollector
        from social_media_sentiment.sentiment_analyzer import SentimentAnalyzer
    except Exception as exc:
        print(f"  [Apify] import failed: {exc}")
        return pd.DataFrame()

    collector = ApifyCollector()
    subset = list(tickers)[:max_tickers]
    skip_twitter = os.getenv("USE_APIFY_TWITTER", "0") != "1"

    print(f"  [Apify] scraping {len(subset)} tickers (twitter={'off' if skip_twitter else 'on'})...")
    all_posts = []
    for tic in subset:
        try:
            posts = collector.search_ticker(
                tic,
                limit=limit_per_platform,
                hours_back=hours_back,
                skip_twitter=skip_twitter,
            )
        except Exception as exc:
            print(f"    {tic}: {exc}")
            continue
        for p in posts:
            text = p.get("text", "")
            created = p.get("created_utc")
            if not text or not created:
                continue
            try:
                d = created.date() if hasattr(created, "date") else pd.to_datetime(created).date()
            except Exception:
                continue
            all_posts.append({"ticker": tic, "text": text[:512], "date": d})

    if not all_posts:
        print("  [Apify] no posts collected")
        return pd.DataFrame()

    print(f"  [Apify] scoring {len(all_posts)} posts with FinBERT...")
    analyzer = SentimentAnalyzer()
    sentiments = analyzer.analyze_batch([p["text"] for p in all_posts], batch_size=32)

    rows = [{
        "ticker": p["ticker"], "date": p["date"],
        "ticker_sentiment_score": s["score"],
        "ticker_relevance_score": s["confidence"],
    } for p, s in zip(all_posts, sentiments)]

    df = pd.DataFrame(rows)
    df["weighted_score"] = df["ticker_sentiment_score"] * df["ticker_relevance_score"]
    agg = (df.groupby(["date", "ticker"])
             .agg(raw_avg_sentiment=("ticker_sentiment_score", "mean"),
                  total_relevance=("ticker_relevance_score", "sum"),
                  weighted_sum=("weighted_score", "sum"),
                  article_count=("ticker_sentiment_score", "count"))
             .reset_index())
    agg["weighted_avg_sentiment"] = (agg["weighted_sum"] / agg["total_relevance"]).fillna(0)
    print(f"  [Apify] {len(agg)} (date,ticker) rows across {agg['ticker'].nunique()} tickers")
    agg.to_csv(cache, index=False)
    return agg


# ---------------------------------------------------------------------------
# STEP 2 — Download OHLCV + technical indicators
# ---------------------------------------------------------------------------
def prepare_market_data(tickers):
    cache = os.path.join(OUTPUT_DIR, "market_data.csv")
    if os.path.exists(cache):
        print("  Loading cached market data...")
        return pd.read_csv(cache)

    print("  Downloading OHLCV from Yahoo Finance...")
    import yfinance as yf
    all_dfs = []
    for tic in tickers:
        try:
            raw = yf.download(tic, start=START_DATE, end=END_DATE, progress=False)
            if raw.empty:
                continue
            # Flatten multi-level columns if needed
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            cols = raw.columns.tolist()
            # Handle both old ("Adj Close") and new yfinance column names
            if "Adj Close" in cols:
                tmp = raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
                tmp.columns = ["open", "high", "low", "close", "adjcp", "volume"]
                tmp["close"] = tmp["adjcp"]
            else:
                tmp = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                tmp.columns = ["open", "high", "low", "close", "volume"]
            tmp["tic"] = tic
            tmp["date"] = tmp.index
            tmp = tmp.reset_index(drop=True)
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
            tmp["day"] = pd.to_datetime(tmp["date"]).dt.dayofweek
            tmp = tmp[["date", "open", "high", "low", "close", "volume", "tic", "day"]]
            all_dfs.append(tmp)
        except Exception as e:
            print(f"    WARN: failed to download {tic}: {e}")
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    print("  Adding technical indicators...")
    df = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False
    ).preprocess_data(df.copy())

    df.to_csv(cache, index=False)
    print(f"  Market data: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# STEP 3 — Merge market + sentiment
# ---------------------------------------------------------------------------
def merge_data(df_market, df_sentiment):
    df_sentiment_r = df_sentiment.rename(columns={"ticker": "tic"})
    df_sentiment_r["date"] = pd.to_datetime(df_sentiment_r["date"]).dt.strftime("%Y-%m-%d")
    merged = df_market.merge(df_sentiment_r, on=["date", "tic"], how="inner")
    merged.drop(columns=["raw_avg_sentiment", "total_relevance", "weighted_sum", "article_count"],
                inplace=True, errors="ignore")
    print(f"  Merged data: {len(merged)} rows, {merged['tic'].nunique()} tickers")
    return merged


# ---------------------------------------------------------------------------
# STEP 4 — Build environment and run agent backtesting
# ---------------------------------------------------------------------------
# If True: sentiment is part of the DRL state (feature mode, original).
# If False: sentiment is NOT in the state; it only acts through the gate.
# Literature finding (Ye et al. 2024; our own A2C_Gated result): sentiment is
# more effective as a *gate* than as an input feature.
USE_SENTIMENT_FEATURE = False


def build_env_kwargs(df):
    stock_dim = len(df.tic.unique())
    extra_feats = ["weighted_avg_sentiment"] if USE_SENTIMENT_FEATURE else []
    tech_list = INDICATORS + extra_feats
    state_space = 1 + 2 * stock_dim + len(tech_list) * stock_dim
    return {
        "hmax": 100,
        "initial_amount": INITIAL_AMOUNT,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": tech_list,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
        "print_verbosity": 500,
    }


def run_agent_backtest(model, trade_df, env_kwargs, agent_name):
    """Run a single agent's backtest and return (account_value_df, actions_df)."""
    print(f"  Running backtest: {agent_name}...")
    env = StockTradingEnv(df=trade_df, **env_kwargs)
    df_account, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)
    df_account["agent"] = agent_name
    df_actions["agent"] = agent_name
    return df_account, df_actions


# ---------------------------------------------------------------------------
# LangGraph gating — applies analyst/manager decisions to DRL actions
# ---------------------------------------------------------------------------
# Rules (negative-tail-only, per Lopez-Lira 2023 + Baker-Wurgler):
#   Hard veto:     executor says SELL AND score < NEG_THRESH AND DRL wants BUY → clip to 0
#   Score veto:    score < NEG_THRESH AND DRL wants BUY                         → clip to 0
# Positive sentiment is NOT actioned — it is priced in fast and adds no edge.
# HOLD verdicts are ignored (no dampen) — HOLD is usually low-conviction noise.
# Tickers without a signal: action untouched (pure DRL, backwards-compatible).

GATE_NEG_SENTIMENT_THRESHOLD = -0.10  # only veto buys when sentiment is meaningfully negative


def apply_langgraph_gate(action_vec, current_date, tickers, signals_df, override_log=None):
    """
    Modify a single-step action vector using cached LangGraph signals.
    `action_vec` is a 1-D array with one entry per ticker (order = `tickers`).
    `override_log`, when provided, gets rows appended describing any change.
    Returns the modified action vector (new array, not in-place).
    """
    from langgraph_signals import get_signal  # local import to keep top-of-module light

    arr = np.array(action_vec, dtype=np.float32).copy()
    if signals_df is None or signals_df.empty:
        return arr

    for i, tic in enumerate(tickers):
        sig = get_signal(signals_df, tic, current_date)
        if not sig:
            continue
        rec = sig["recommendation"]
        score = sig["sentiment_score"]
        orig = float(arr[i])
        new = orig
        reason = None

        # Negative-tail veto: only block BUYs when sentiment is meaningfully bad.
        # Positive sentiment is ignored (priced in fast); HOLD is ignored (noise).
        if orig > 0 and (
            (rec == "SELL" and score <= GATE_NEG_SENTIMENT_THRESHOLD)
            or score <= GATE_NEG_SENTIMENT_THRESHOLD
        ):
            new = 0.0
            reason = f"veto_buy_neg_sentiment_{score:.2f}"

        if new != orig:
            arr[i] = new
            if override_log is not None:
                override_log.append({
                    "date": str(current_date),
                    "ticker": tic,
                    "original_action": orig,
                    "gated_action": new,
                    "recommendation": rec,
                    "conviction": sig.get("conviction", ""),
                    "sentiment_score": score,
                    "reason": reason,
                    "signal_date": sig.get("date", ""),
                })

    return arr


def run_agent_backtest_gated(model, trade_df, env_kwargs, signals_df, agent_name):
    """
    FinRL-style prediction loop that applies `apply_langgraph_gate` to each action
    before stepping the environment. Returns (df_account, df_actions, override_log).
    """
    print(f"  Running GATED backtest: {agent_name}...")
    env = StockTradingEnv(df=trade_df, **env_kwargs)
    test_env, test_obs = env.get_sb_env()
    test_env.reset()

    unique_dates = sorted(env.df["date"].unique())
    tickers = sorted(env.df["tic"].unique())  # StockTradingEnv action order matches sorted tic
    n_steps = len(env.df.index.unique())

    override_log = []
    account_memory = []
    actions_memory = []

    for i in range(n_steps):
        action, _states = model.predict(test_obs, deterministic=True)
        current_date = unique_dates[i] if i < len(unique_dates) else unique_dates[-1]
        # action shape is (1, n_tickers) for the VecEnv
        gated_vec = apply_langgraph_gate(action[0], current_date, tickers, signals_df, override_log)
        action = np.array([gated_vec], dtype=action.dtype)

        test_obs, rewards, dones, info = test_env.step(action)
        if i == (n_steps - 2):
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")

    if not account_memory:
        # Fallback: edge case when loop length doesn't hit n-2
        account_memory = test_env.env_method(method_name="save_asset_memory")
        actions_memory = test_env.env_method(method_name="save_action_memory")

    df_account = account_memory[0]
    df_actions = actions_memory[0]
    df_account["agent"] = f"{agent_name}_GATED"
    df_actions["agent"] = f"{agent_name}_GATED"
    return df_account, df_actions, override_log


def load_and_backtest_agents(trade_df, env_kwargs):
    """Load pre-trained agents and run backtests."""
    model_dir = os.path.join(DRL_DIR, "trained_models")
    results = {}

    agent_map = {
        "PPO": ("agent_ppo", PPO),
        "A2C": ("agent_a2c", A2C),
        "DDPG": ("agent_ddpg", DDPG),
    }

    dummy_env = StockTradingEnv(df=trade_df, **env_kwargs)
    dummy_sb_env, _ = dummy_env.get_sb_env()

    for name, (filename, ModelClass) in agent_map.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path + ".zip"):
            print(f"  WARNING: {path}.zip not found, skipping {name}")
            continue
        try:
            model = ModelClass.load(path, device="cpu")
            acct, acts = run_agent_backtest(model, trade_df, env_kwargs, name)
            results[name] = {"account": acct, "actions": acts}
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Earnings dates helper (cached) — used by EarningsSentiment mode
# ---------------------------------------------------------------------------
def _earnings_only_dates(earn_list):
    """Accept either the new [(date, surprise_pct), ...] format or the old
    [date, ...] format and return a list of dates only."""
    if not earn_list:
        return []
    if isinstance(earn_list[0], tuple):
        return [d for d, _ in earn_list]
    return earn_list


def fetch_earnings_dates(tickers, cache_path):
    """
    Cached pull of yfinance.Ticker.earnings_dates. Returns a dict
    {tic: list of (earnings_date, surprise_pct) tuples sorted by date}.
    Surprise(%) is the EPS beat/miss vs analyst consensus, used by the PEAD
    (post-earnings announcement drift) feature in the MetaModel.
    """
    refresh = not os.path.exists(cache_path)
    if not refresh:
        df = pd.read_csv(cache_path, parse_dates=["earnings_date"])
        # Old cache had no surprise column — refresh once to add it.
        if "surprise_pct" not in df.columns:
            print(f"    earnings cache lacks surprise_pct; refetching...")
            os.remove(cache_path)
            refresh = True

    if refresh:
        import yfinance as yf
        rows = []
        print(f"    fetching earnings dates + surprise for {len(tickers)} tickers...")
        for tic in tickers:
            try:
                ed = yf.Ticker(tic).earnings_dates
                if ed is None or ed.empty:
                    continue
                ed = ed.reset_index()
                ed["earnings_date"] = pd.to_datetime(ed["Earnings Date"]).dt.tz_localize(None).dt.normalize()
                ed["tic"] = tic
                ed["surprise_pct"] = pd.to_numeric(ed.get("Surprise(%)"), errors="coerce")
                rows.append(ed[["tic", "earnings_date", "surprise_pct"]])
            except Exception as e:
                print(f"      {tic}: skipped ({e})")
        if not rows:
            return {}
        df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["tic", "earnings_date"])
        df.to_csv(cache_path, index=False)
        print(f"    cached {len(df)} earnings dates to {os.path.basename(cache_path)}")
    out = {}
    for tic, sub in df.groupby("tic"):
        sub = sub.sort_values("earnings_date")
        out[tic] = list(zip(sub["earnings_date"].tolist(),
                            sub["surprise_pct"].fillna(0.0).tolist()))
    return out


# ---------------------------------------------------------------------------
# STEP 4b — Rule-based MultiStrategy bot (from Trading_bot.ipynb)
# ---------------------------------------------------------------------------
def run_multistrategy_backtest(df_market, tickers, mode="SMA_RSI", df_sentiment=None,
                               earnings_by_tic=None):
    """
    Rule-based multi-strategy bot adapted from Trading_bot.ipynb.
    Runs an equal-weight portfolio across all tickers using SMA/RSI signals.

    Key fixes over the original notebook:
    - Compute indicators on FULL history, then trade only in test period
    - Use actual SMA50/SMA200 (not SMA100/SMA200)
    - Use RSI(14) (standard) instead of RSI(21)
    - Add trailing stop-loss (8%)
    - Sentiment-enhanced mode uses sentiment score as additional filter
    """
    agent_name = f"RuleBased ({mode})"
    print(f"  Running backtest: {agent_name}...")

    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])

    # If we have sentiment, merge it for sentiment-enhanced mode
    sent_lookup = {}
    if df_sentiment is not None and ("Sentiment" in mode or "Dynamic" in mode):
        sf = df_sentiment.copy()
        sf["date"] = pd.to_datetime(sf["date"])
        sf = sf.rename(columns={"ticker": "tic"})
        for _, r in sf.iterrows():
            sent_lookup[(r["date"].strftime("%Y-%m-%d"), r["tic"])] = r.get("weighted_avg_sentiment", 0)

    test_start_dt = pd.Timestamp(TEST_START)

    # Per-ticker: compute indicators on FULL history, then slice for trading
    all_signals = {}
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").copy()
        if len(tdf) < 250:
            continue

        # Indicators on full history
        tdf["SMA50"] = tdf["close"].rolling(50).mean()
        tdf["SMA200"] = tdf["close"].rolling(200).mean()
        delta = tdf["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        tdf["RSI"] = 100 - (100 / (1 + rs))
        # ATR(14) for volatility-adjusted trailing stop
        prev_close = tdf["close"].shift()
        tr = pd.concat([
            tdf["high"] - tdf["low"],
            (tdf["high"] - prev_close).abs(),
            (tdf["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        tdf["ATR14"] = tr.rolling(14).mean()
        # Momentum filter for dip-buy modes
        tdf[f"MOM{MOMENTUM_LOOKBACK}"] = tdf["close"].pct_change(MOMENTUM_LOOKBACK)
        tdf.dropna(inplace=True)

        # Now filter to test period for trading
        tdf = tdf[tdf["date"] >= test_start_dt].copy()
        if tdf.empty:
            continue

        # Generate signals based on mode
        if mode == "SMA":
            tdf["Buy"] = (tdf["SMA50"] > tdf["SMA200"]) & (tdf["SMA50"].shift(1) <= tdf["SMA200"].shift(1))
            tdf["Sell"] = (tdf["SMA50"] < tdf["SMA200"]) & (tdf["SMA50"].shift(1) >= tdf["SMA200"].shift(1))
        elif mode == "RSI":
            tdf["Buy"] = tdf["RSI"] < 35
            tdf["Sell"] = tdf["RSI"] > 75
        elif mode == "SMA_RSI":
            # Buy: uptrend confirmed AND pullback (RSI dip)
            uptrend = tdf["SMA50"] > tdf["SMA200"]
            tdf["Buy"] = uptrend & (tdf["RSI"] < 40) & (tdf["RSI"].shift(1) >= 40)
            # Sell: trend reversal only (not just RSI overbought)
            tdf["Sell"] = (~uptrend) & (tdf["SMA50"].shift(1) > tdf["SMA200"].shift(1))
        elif mode == "SMA_RSI_Sentiment":
            uptrend = tdf["SMA50"] > tdf["SMA200"]
            rsi_dip = (tdf["RSI"] < 40) & (tdf["RSI"].shift(1) >= 40)
            # Ablation: gate_pos (sent > +0.1) on 3-day smoothed sentiment is the
            # best mode. Sentiment is buy-gate only, never a stop trigger —
            # veto_neg exits were shown to hurt. When a ticker has no sentiment
            # coverage at all, pass through to the base signal so we don't silently
            # block every trade on uncovered names.
            tdf["_sent_raw"] = tdf.apply(
                lambda r: sent_lookup.get((r["date"].strftime("%Y-%m-%d"), tic), np.nan), axis=1
            )
            has_coverage = tdf["_sent_raw"].notna().mean() > 0.10
            if has_coverage:
                tdf["_sent"] = tdf["_sent_raw"].fillna(0).rolling(3, min_periods=1).mean()
                tdf["Buy"] = uptrend & rsi_dip & (tdf["_sent"] > 0.1)
            else:
                tdf["Buy"] = uptrend & rsi_dip  # fall through: no sentiment data
            tdf["Sell"] = (~uptrend) & (tdf["SMA50"].shift(1) > tdf["SMA200"].shift(1))
        elif mode == "Dynamic":
            # Same signals as SMA_RSI but allocation/sizing handled dynamically
            uptrend = tdf["SMA50"] > tdf["SMA200"]
            tdf["Buy"] = uptrend & (tdf["RSI"] < 40) & (tdf["RSI"].shift(1) >= 40)
            tdf["Sell"] = (~uptrend) & (tdf["SMA50"].shift(1) > tdf["SMA200"].shift(1))
        elif mode == "EarningsSentiment":
            # Hypothesis: news sentiment alpha is concentrated ±5 trading days
            # around earnings; outside that window it's noise. Inside the
            # earnings window, require positive sentiment to fire SMA_RSI buys;
            # outside the window, fall through to plain SMA_RSI.
            uptrend = tdf["SMA50"] > tdf["SMA200"]
            rsi_dip = (tdf["RSI"] < 40) & (tdf["RSI"].shift(1) >= 40)
            tdf["_sent_raw"] = tdf.apply(
                lambda r: sent_lookup.get((r["date"].strftime("%Y-%m-%d"), tic), np.nan),
                axis=1,
            )
            tdf["_sent"] = tdf["_sent_raw"].ffill().fillna(0).rolling(3, min_periods=1).mean()
            # Build the earnings-window mask: True if any earnings date within
            # ±5 trading days (≈ ±7 calendar days) of this row's date.
            earn_dates = _earnings_only_dates((earnings_by_tic or {}).get(tic, []))
            window_td = pd.Timedelta(days=7)
            in_window = pd.Series(False, index=tdf.index)
            for ed_dt in earn_dates:
                in_window |= (tdf["date"] - ed_dt).abs() <= window_td
            base_buy = uptrend & rsi_dip
            tdf["Buy"] = base_buy & (
                (~in_window) | (in_window & (tdf["_sent"] > 0.1))
            )
            tdf["Sell"] = (~uptrend) & (tdf["SMA50"].shift(1) > tdf["SMA200"].shift(1))
        elif mode == "SentimentMomentum":
            # New approach: buy on positive sentiment SURPRISE (5-day MA above
            # 30-day baseline by > 0.15) with uptrend confirmation. Sells on
            # surprise reversal or trend reversal. Skips tickers without
            # sentiment coverage entirely (they sit in cash earning the daily yield).
            uptrend = tdf["SMA50"] > tdf["SMA200"]
            tdf["_sent_raw"] = tdf.apply(
                lambda r: sent_lookup.get((r["date"].strftime("%Y-%m-%d"), tic), np.nan),
                axis=1,
            )
            has_coverage = tdf["_sent_raw"].notna().mean() > 0.10
            if has_coverage:
                # Forward-fill recent sentiment (sticky news effect), then 0-fill
                # for the cold start at the very beginning of the test window.
                sent = tdf["_sent_raw"].ffill().fillna(0)
                sent_short = sent.rolling(5, min_periods=1).mean()
                sent_long = sent.rolling(30, min_periods=5).mean()
                surprise = sent_short - sent_long
                tdf["Buy"] = uptrend & (surprise > 0.05)
                tdf["Sell"] = (~uptrend) | (surprise < -0.10)
            else:
                # No coverage → never buys; bucket sits in cash earning yield.
                tdf["Buy"] = pd.Series(False, index=tdf.index)
                tdf["Sell"] = pd.Series(False, index=tdf.index)

        # Apply 63-day momentum filter to RSI-based dip-buy modes.
        # Pure SMA crossover already requires trend confirmation, so leave it alone.
        if mode in {"RSI", "SMA_RSI", "SMA_RSI_Sentiment", "Dynamic"}:
            mom_col = f"MOM{MOMENTUM_LOOKBACK}"
            tdf["Buy"] = tdf["Buy"] & (tdf[mom_col] >= 0.0)

        # Store RSI/ATR for sizing + stops downstream
        all_signals[tic] = tdf[["date", "close", "Buy", "Sell", "RSI", "ATR14"]].copy()

    if not all_signals:
        print(f"    WARNING: No tickers had enough data for {agent_name}")
        return None

    # ---- Compute per-ticker allocation weights ----
    n_stocks = len(all_signals)
    # Mean-reversion modes (RSI core) get a tight ATR stop; trend-following modes
    # get a wider one so the position can breathe through normal pullbacks.
    if mode in {"RSI", "Dynamic"}:
        atr_mult = TRAILING_ATR_MULT_RSI
    else:
        atr_mult = TRAILING_ATR_MULT_TREND

    if "Dynamic" in mode:
        # 1) Volatility-weighted: inverse of 60-day rolling std → equal risk contribution
        vol_weights = {}
        for tic, sig_df in all_signals.items():
            daily_ret = sig_df["close"].pct_change().dropna()
            vol = daily_ret.std() if len(daily_ret) > 20 else 0.02  # fallback 2%
            vol_weights[tic] = 1.0 / max(vol, 0.005)  # inverse vol, floor at 0.5%

        # 2) Sentiment weight: scale by (1 + sentiment_score), clamped [0.5, 2.0]
        sent_weights = {}
        for tic in all_signals:
            # Average sentiment for this ticker over the test period
            scores = [sent_lookup.get((d.strftime("%Y-%m-%d"), tic), 0)
                      for d in all_signals[tic]["date"]]
            avg_sent = np.mean(scores) if scores else 0
            sent_weights[tic] = np.clip(1.0 + avg_sent * 3, 0.5, 2.0)

        # Combined weight = vol_weight * sentiment_weight
        raw_weights = {tic: vol_weights[tic] * sent_weights[tic] for tic in all_signals}
        total_w = sum(raw_weights.values())
        alloc_weights = {tic: w / total_w for tic, w in raw_weights.items()}

        top5 = sorted(alloc_weights.items(), key=lambda x: -x[1])[:5]
        bot5 = sorted(alloc_weights.items(), key=lambda x: x[1])[:5]
        print(f"    Dynamic allocation — top 5: {[(t, f'{w:.1%}') for t, w in top5]}")
        print(f"    Dynamic allocation — bot 5: {[(t, f'{w:.1%}') for t, w in bot5]}")
    else:
        # Equal weight
        alloc_weights = {tic: 1.0 / n_stocks for tic in all_signals}

    positions = {}
    for tic in all_signals:
        positions[tic] = {
            "shares": 0,
            "cash": INITIAL_AMOUNT * alloc_weights[tic],
            "peak_price": 0.0,
            "base_cash": INITIAL_AMOUNT * alloc_weights[tic],  # remember allocation for sizing
            "entry_price": 0.0,
            "entry_shares": 0,
            "profit_lvl": 0,  # 0 = no scale-out yet, 1 = took T1, 2 = took T2
        }

    all_dates = sorted(set().union(*(sig["date"].tolist() for sig in all_signals.values())))

    account_history = []
    actions_history = []

    for dt in all_dates:
        total_value = 0
        day_actions = {"date": dt.strftime("%Y-%m-%d")}

        for tic, sig_df in all_signals.items():
            pos = positions[tic]
            # Cash earns the daily risk-free rate every trading day, whether or
            # not the ticker has a price today.
            pos["cash"] *= (1 + DAILY_CASH_YIELD)

            row = sig_df[sig_df["date"] == dt]
            if row.empty:
                total_value += pos["cash"]
                day_actions[tic] = 0
                continue

            price = float(row["close"].values[0])
            buy = bool(row["Buy"].values[0])
            sell = bool(row["Sell"].values[0])
            rsi_val = float(row["RSI"].values[0])
            atr_val = float(row["ATR14"].values[0])
            if not np.isfinite(atr_val) or atr_val <= 0:
                atr_val = price * 0.02  # fall back to ~2% if ATR missing
            action = 0
            stop_triggered = False

            # Update trailing stop peak + ATR-based exit
            if pos["shares"] > 0:
                pos["peak_price"] = max(pos["peak_price"], price)
                stop_level = pos["peak_price"] - atr_mult * atr_val
                if price < stop_level:
                    sell = True
                    stop_triggered = True

                # Profit ladder: scale out 1/3 at +T1, another 1/3 at +T2.
                # Skip ladder if a stop was triggered — prefer the full exit.
                if not stop_triggered and pos["entry_price"] > 0:
                    gain = (price - pos["entry_price"]) / pos["entry_price"]
                    third = max(1, pos["entry_shares"] // 3)
                    if pos["profit_lvl"] == 0 and gain >= PROFIT_T1 and pos["shares"] >= third:
                        proceeds = third * price * 0.999
                        pos["cash"] += proceeds
                        pos["shares"] -= third
                        pos["profit_lvl"] = 1
                        action = -third
                    elif pos["profit_lvl"] == 1 and gain >= PROFIT_T2 and pos["shares"] >= third:
                        proceeds = third * price * 0.999
                        pos["cash"] += proceeds
                        pos["shares"] -= third
                        pos["profit_lvl"] = 2
                        action = -third

            if buy and pos["shares"] == 0 and pos["cash"] > price:
                # Conviction-based sizing: deeper RSI dip → larger fraction of allocation
                if "Dynamic" in mode:
                    conviction = np.clip((40 - rsi_val) / 20, 0.5, 1.0)
                    day_sent = sent_lookup.get((dt.strftime("%Y-%m-%d"), tic), 0)
                    sent_boost = np.clip(1.0 + day_sent * 2, 0.6, 1.5)
                    size_frac = conviction * sent_boost
                    buyable_cash = pos["cash"] * size_frac
                else:
                    buyable_cash = pos["cash"]

                shares_to_buy = int(buyable_cash // (price * 1.001))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * 1.001
                    pos["shares"] = shares_to_buy
                    pos["cash"] -= cost
                    pos["peak_price"] = price
                    pos["entry_price"] = price
                    pos["entry_shares"] = shares_to_buy
                    pos["profit_lvl"] = 0
                    action = shares_to_buy
            elif sell and pos["shares"] > 0:
                # ATR stop or trend reversal: exit the remaining position fully.
                # Dynamic-mode partial sell only applies to soft (non-stop) signals.
                if (
                    "Dynamic" in mode
                    and not stop_triggered
                    and rsi_val < 80
                    and price > pos["peak_price"] * 0.95
                ):
                    shares_to_sell = max(1, pos["shares"] // 2)
                else:
                    shares_to_sell = pos["shares"]

                proceeds = shares_to_sell * price * 0.999
                action = -shares_to_sell
                pos["cash"] += proceeds
                pos["shares"] -= shares_to_sell
                if pos["shares"] == 0:
                    pos["peak_price"] = 0.0
                    pos["entry_price"] = 0.0
                    pos["entry_shares"] = 0
                    pos["profit_lvl"] = 0

            total_value += pos["cash"] + pos["shares"] * price
            day_actions[tic] = action

        account_history.append({"date": dt.strftime("%Y-%m-%d"), "account_value": total_value})
        actions_history.append(day_actions)

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name

    return {"account": df_account, "actions": df_actions}


# ---------------------------------------------------------------------------
# STEP 4b2 — Regime-adaptive strategy (target-weight execution, long + short)
# ---------------------------------------------------------------------------
def run_regime_adaptive_backtest(df_market, tickers):
    """
    Regime-adaptive strategy: picks target exposure per-ticker per-day based on
    (SMA50 vs SMA200) + (SMA20 vs SMA50) + (vol60 vs 252d median vol60).

    Regime signal → target weight mapping:
        1 Mom_LowVol (bull + low vol + ST up)   →  +1.0  (full long)
        2 HalfHedge  (bull + low vol + ST down) →  +0.5  (half long)
        3 Short_Only (bear + ST down)           →  -1.0  (full short)
        4 Flat       (chop or bear bounce)      →   0.0  (cash)

    Execution is target-weight rebalancing — each day we move position toward
    the regime target, paying 0.1% transaction cost on notional traded.
    """
    agent_name = "RuleBased (RegimeAdaptive)"
    print(f"  Running backtest: {agent_name}...")

    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start_dt = pd.Timestamp(TEST_START)

    # ---- Feature computation on full history, slice to test window ----
    per_ticker: dict[str, pd.DataFrame] = {}
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").copy()
        if len(tdf) < 260:
            continue
        tdf["SMA20"] = tdf["close"].rolling(20).mean()
        tdf["SMA50"] = tdf["close"].rolling(50).mean()
        tdf["SMA200"] = tdf["close"].rolling(200).mean()
        daily_ret = tdf["close"].pct_change()
        tdf["VOL60"] = daily_ret.rolling(60).std()
        tdf["VOL60_MED252"] = tdf["VOL60"].rolling(252).median()
        tdf.dropna(subset=["SMA20", "SMA50", "SMA200", "VOL60", "VOL60_MED252"], inplace=True)
        tdf = tdf[tdf["date"] >= test_start_dt].copy()
        if tdf.empty:
            continue
        per_ticker[tic] = tdf.reset_index(drop=True)

    if not per_ticker:
        print(f"    WARNING: No tickers had enough data for {agent_name}")
        return None

    # ---- Market-trend gate: only allow shorts when SPY itself is in a downtrend ----
    # Pulls ^GSPC back to START_DATE so SMAs are causal on the test window.
    try:
        import yfinance as yf
        spy_raw = yf.download("^GSPC", start=START_DATE, end=END_DATE,
                              progress=False, auto_adjust=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = [c[0] for c in spy_raw.columns]
        spy = spy_raw.reset_index()[["Date", "Close"]].rename(
            columns={"Date": "date", "Close": "close"}
        )
        spy["date"] = pd.to_datetime(spy["date"])
        spy["SMA50"] = spy["close"].rolling(50).mean()
        spy["SMA200"] = spy["close"].rolling(200).mean()
        spy["spy_bear"] = (spy["SMA50"] < spy["SMA200"])
        spy_bear_by_date = dict(zip(spy["date"], spy["spy_bear"].fillna(False)))
        print(f"    SPY trend gate: {int(sum(spy_bear_by_date.values()))} of "
              f"{len(spy_bear_by_date)} SPY days flagged bear (shorts allowed only on those)")
    except Exception as e:
        print(f"    WARNING: SPY gate unavailable ({e}); falling back to no gate")
        spy_bear_by_date = {}

    n = len(per_ticker)
    alloc_per_ticker = INITIAL_AMOUNT / n

    # Position state: cash + shares (shares can be negative for shorts)
    positions = {tic: {"shares": 0, "cash": alloc_per_ticker, "short_proceeds": 0.0}
                 for tic in per_ticker}

    def regime_target(row, spy_bear: bool) -> float:
        s20, s50, s200 = row["SMA20"], row["SMA50"], row["SMA200"]
        v60, vm = row["VOL60"], row["VOL60_MED252"]
        vol_scale = max(0.5, min(1.5, vm / v60)) if v60 and v60 > 0 else 1.0
        if s50 > s200:                        # ticker bull — always participate, sized by vol
            base = 1.0 if s20 > s50 else 0.5
            return base * vol_scale
        if s50 < s200 and s20 < s50:          # ticker bear + ST down
            # Only allow shorts when SPY is also bear; else flat.
            return (-1.0 * vol_scale) if spy_bear else 0.0
        return 0.0                            # chop / bear bounce

    all_dates = sorted(set().union(*(sig["date"].tolist() for sig in per_ticker.values())))
    account_history: list[dict] = []
    actions_history: list[dict] = []
    regime_counts = {1.0: 0, 0.5: 0, -1.0: 0, 0.0: 0}

    TX_COST = 0.001

    for dt in all_dates:
        total_value = 0.0
        day_actions = {"date": dt.strftime("%Y-%m-%d")}

        for tic, tdf in per_ticker.items():
            pos = positions[tic]
            # Idle cash earns the daily risk-free rate.
            pos["cash"] *= (1 + DAILY_CASH_YIELD)
            row_slice = tdf[tdf["date"] == dt]
            if row_slice.empty:
                total_value += pos["cash"] + pos["shares"] * 0  # stale: hold cash only
                day_actions[tic] = 0
                continue
            row = row_slice.iloc[0]
            price = float(row["close"])
            spy_bear = bool(spy_bear_by_date.get(dt, False))
            target = regime_target(row, spy_bear)
            if target > 0.75:
                bucket = 1.0
            elif target > 0.25:
                bucket = 0.5
            elif target >= -0.25:
                bucket = 0.0
            else:
                bucket = -1.0
            regime_counts[bucket] = regime_counts.get(bucket, 0) + 1

            # Equity currently available to this ticker: cash side + MTM of position
            equity = pos["cash"] + pos["shares"] * price
            target_shares = 0 if price <= 0 else int((target * equity) // price)

            delta = target_shares - pos["shares"]
            action = 0
            if delta > 0:  # buying (covering short or adding long)
                cost = delta * price * (1 + TX_COST)
                if pos["cash"] >= cost:
                    pos["cash"] -= cost
                    pos["shares"] += delta
                    action = delta
            elif delta < 0:  # selling (exiting long or opening short)
                proceeds = (-delta) * price * (1 - TX_COST)
                pos["cash"] += proceeds
                pos["shares"] += delta
                action = delta

            total_value += pos["cash"] + pos["shares"] * price
            day_actions[tic] = action

        account_history.append({"date": dt.strftime("%Y-%m-%d"), "account_value": total_value})
        actions_history.append(day_actions)

    total_regimes = sum(regime_counts.values()) or 1
    regime_pct = {k: f"{100 * v / total_regimes:.1f}%" for k, v in regime_counts.items()}
    print(f"    Regime distribution — full-long={regime_pct.get(1.0,'0%')}, "
          f"half={regime_pct.get(0.5,'0%')}, short={regime_pct.get(-1.0,'0%')}, "
          f"flat={regime_pct.get(0.0,'0%')}")

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name
    return {"account": df_account, "actions": df_actions}


# ---------------------------------------------------------------------------
# STEP 4b3 — Cross-sectional momentum (rank-based, not per-ticker buckets)
# ---------------------------------------------------------------------------
def run_cross_momentum_backtest(df_market, tickers, top_n=10, lookback=63, rebalance_days=21):
    """
    Cross-sectional momentum: every `rebalance_days` trading days, rank all
    tickers by their `lookback`-day return and hold the top `top_n` equal-weight.
    Liquidate any name that drops out of the top set on rebalance day.
    Per-position ATR(14) trailing stop (3.0× — wider since we're trend-following)
    + cash earns the daily risk-free rate while between rebalances.
    """
    agent_name = "RuleBased (CrossMomentum)"
    print(f"  Running backtest: {agent_name}...")

    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start_dt = pd.Timestamp(TEST_START)

    per_ticker: dict[str, pd.DataFrame] = {}
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").copy()
        if len(tdf) < lookback + 30:
            continue
        tdf["MOM"] = tdf["close"].pct_change(lookback)
        prev_close = tdf["close"].shift()
        tr = pd.concat([
            tdf["high"] - tdf["low"],
            (tdf["high"] - prev_close).abs(),
            (tdf["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        tdf["ATR14"] = tr.rolling(14).mean()
        tdf.dropna(subset=["MOM", "ATR14"], inplace=True)
        tdf = tdf[tdf["date"] >= test_start_dt].reset_index(drop=True)
        if tdf.empty:
            continue
        per_ticker[tic] = tdf

    if not per_ticker:
        print(f"    WARNING: No tickers had enough data for {agent_name}")
        return None

    all_dates = sorted(set().union(*(t["date"].tolist() for t in per_ticker.values())))
    cash = INITIAL_AMOUNT
    holdings: dict[str, dict] = {}  # tic -> {shares, peak_price, entry_price}
    account_history = []
    actions_history = []
    days_since_rebal = rebalance_days  # force rebalance on day 0

    for dt in all_dates:
        # Lookups for the day
        prices = {}
        atrs = {}
        moms = {}
        for tic, tdf in per_ticker.items():
            row = tdf[tdf["date"] == dt]
            if row.empty:
                continue
            prices[tic] = float(row["close"].iloc[0])
            atrs[tic] = float(row["ATR14"].iloc[0])
            moms[tic] = float(row["MOM"].iloc[0])

        cash *= (1 + DAILY_CASH_YIELD)

        # ATR trailing stops on currently-held positions
        for tic in list(holdings.keys()):
            if tic not in prices:
                continue
            h = holdings[tic]
            price = prices[tic]
            h["peak_price"] = max(h["peak_price"], price)
            atr = atrs[tic] if np.isfinite(atrs[tic]) and atrs[tic] > 0 else price * 0.02
            stop = h["peak_price"] - TRAILING_ATR_MULT_TREND * atr
            if price < stop:
                cash += h["shares"] * price * 0.999
                del holdings[tic]

        # Rebalance every rebalance_days trading days
        day_actions = {"date": dt.strftime("%Y-%m-%d")}
        days_since_rebal += 1
        if days_since_rebal >= rebalance_days and prices:
            days_since_rebal = 0
            ranked = sorted(moms.items(), key=lambda x: -x[1])
            # Use a list (deterministic order: highest momentum first) so cash
            # constraints are resolved reproducibly when capital is tight.
            target_list = [t for t, _ in ranked[:top_n] if moms[t] > 0]
            target_set = set(target_list)

            # Sell anything not in target set
            for tic in sorted(holdings.keys()):
                if tic not in target_set and tic in prices:
                    h = holdings[tic]
                    cash += h["shares"] * prices[tic] * 0.999
                    day_actions[tic] = -h["shares"]
                    del holdings[tic]

            # Total equity right now
            total_equity = cash + sum(
                h["shares"] * prices.get(t, h["entry_price"]) for t, h in holdings.items()
            )
            target_alloc = total_equity / max(len(target_list), 1)

            # Buy/scale-in for target tickers — deterministic order (highest mom first)
            for tic in target_list:
                if tic not in prices:
                    continue
                price = prices[tic]
                current_value = holdings[tic]["shares"] * price if tic in holdings else 0
                want_value = target_alloc
                delta_value = want_value - current_value
                if delta_value > price:  # buy more
                    shares_to_buy = int(delta_value // (price * 1.001))
                    if shares_to_buy > 0 and cash >= shares_to_buy * price * 1.001:
                        cost = shares_to_buy * price * 1.001
                        cash -= cost
                        if tic in holdings:
                            holdings[tic]["shares"] += shares_to_buy
                        else:
                            holdings[tic] = {
                                "shares": shares_to_buy,
                                "peak_price": price,
                                "entry_price": price,
                            }
                        day_actions[tic] = shares_to_buy
                elif delta_value < -price and tic in holdings:  # trim
                    shares_to_sell = int(-delta_value // price)
                    shares_to_sell = min(shares_to_sell, holdings[tic]["shares"])
                    if shares_to_sell > 0:
                        cash += shares_to_sell * price * 0.999
                        holdings[tic]["shares"] -= shares_to_sell
                        if holdings[tic]["shares"] == 0:
                            del holdings[tic]
                        day_actions[tic] = -shares_to_sell

        # Mark to market
        equity = cash + sum(h["shares"] * prices.get(t, 0) for t, h in holdings.items())
        account_history.append({"date": dt.strftime("%Y-%m-%d"), "account_value": equity})
        for tic in tickers:
            day_actions.setdefault(tic, 0)
        actions_history.append(day_actions)

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name
    return {"account": df_account, "actions": df_actions}


# ---------------------------------------------------------------------------
# STEP 4b4 — Cross-sectional sentiment ranking
# ---------------------------------------------------------------------------
def run_sentiment_rank_backtest(df_market, tickers, df_sentiment, top_n=10,
                                short_window=5, long_window=30, rebalance_days=21):
    """
    Cross-sectional sentiment: each rebalance day, compute per-ticker sentiment
    surprise = mean(last short_window days) - mean(last long_window days), then
    Z-score the surprises ACROSS the universe. Long the top top_n names by Z.
    The relative-rank framing avoids absolute-threshold problems and bypasses
    the "news is priced in" issue that breaks gate-style sentiment strategies.
    """
    agent_name = "RuleBased (SentimentRank)"
    print(f"  Running backtest: {agent_name}...")

    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start_dt = pd.Timestamp(TEST_START)

    # Build per-ticker price + ATR + forward-filled daily sentiment series.
    sf = df_sentiment.copy() if df_sentiment is not None else None
    if sf is None or sf.empty:
        print(f"    WARNING: No sentiment data — {agent_name} skipped")
        return None
    sf["date"] = pd.to_datetime(sf["date"])
    sf = sf.rename(columns={"ticker": "tic"})
    sent_pivot = sf.pivot_table(
        index="date", columns="tic", values="weighted_avg_sentiment", aggfunc="mean"
    )

    per_ticker: dict[str, pd.DataFrame] = {}
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").copy()
        if len(tdf) < long_window + 30:
            continue
        prev_close = tdf["close"].shift()
        tr = pd.concat([
            tdf["high"] - tdf["low"],
            (tdf["high"] - prev_close).abs(),
            (tdf["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        tdf["ATR14"] = tr.rolling(14).mean()
        # Align sentiment onto the trading-day index, ffill (sticky news effect),
        # then 0-fill the cold start.
        if tic in sent_pivot.columns:
            tdf = tdf.merge(
                sent_pivot[tic].rename("sent_raw"),
                left_on="date", right_index=True, how="left",
            )
            tdf["sent"] = tdf["sent_raw"].ffill().fillna(0)
        else:
            tdf["sent"] = 0.0
        tdf["sent_short"] = tdf["sent"].rolling(short_window, min_periods=1).mean()
        tdf["sent_long"] = tdf["sent"].rolling(long_window, min_periods=5).mean()
        tdf["sent_surprise"] = tdf["sent_short"] - tdf["sent_long"]
        tdf.dropna(subset=["ATR14"], inplace=True)
        tdf = tdf[tdf["date"] >= test_start_dt].reset_index(drop=True)
        if tdf.empty:
            continue
        per_ticker[tic] = tdf

    if not per_ticker:
        print(f"    WARNING: No tickers had enough data for {agent_name}")
        return None

    # Coverage gate: only keep tickers that have any non-zero sentiment in the
    # test window (avoids 0-sentiment tickers tying for top rank in early days).
    covered = {t for t, td in per_ticker.items() if (td["sent"].abs() > 0).any()}
    print(f"    sentiment coverage: {len(covered)}/{len(per_ticker)} tickers")

    all_dates = sorted(set().union(*(t["date"].tolist() for t in per_ticker.values())))
    cash = INITIAL_AMOUNT
    holdings: dict[str, dict] = {}
    account_history = []
    actions_history = []
    days_since_rebal = rebalance_days

    for dt in all_dates:
        prices, atrs, surprises = {}, {}, {}
        for tic, tdf in per_ticker.items():
            row = tdf[tdf["date"] == dt]
            if row.empty:
                continue
            prices[tic] = float(row["close"].iloc[0])
            atrs[tic] = float(row["ATR14"].iloc[0])
            if tic in covered:
                surprises[tic] = float(row["sent_surprise"].iloc[0])

        cash *= (1 + DAILY_CASH_YIELD)

        # ATR trailing stops on holdings
        for tic in list(holdings.keys()):
            if tic not in prices:
                continue
            h = holdings[tic]
            price = prices[tic]
            h["peak_price"] = max(h["peak_price"], price)
            atr = atrs[tic] if np.isfinite(atrs[tic]) and atrs[tic] > 0 else price * 0.02
            stop = h["peak_price"] - TRAILING_ATR_MULT_TREND * atr
            if price < stop:
                cash += h["shares"] * price * 0.999
                del holdings[tic]

        day_actions = {"date": dt.strftime("%Y-%m-%d")}
        days_since_rebal += 1
        if days_since_rebal >= rebalance_days and len(surprises) >= top_n:
            days_since_rebal = 0
            # Z-score across the cross-section, then pick top_n with positive Z
            vals = np.array(list(surprises.values()))
            mu, sd = vals.mean(), vals.std(ddof=0)
            if sd <= 0:
                ranked_list = []
            else:
                z = {t: (s - mu) / sd for t, s in surprises.items()}
                ranked = sorted(z.items(), key=lambda x: -x[1])
                ranked_list = [t for t, zv in ranked[:top_n] if zv > 0]
            target_set = set(ranked_list)

            # Sell anything not in target
            for tic in sorted(holdings.keys()):
                if tic not in target_set and tic in prices:
                    h = holdings[tic]
                    cash += h["shares"] * prices[tic] * 0.999
                    day_actions[tic] = -h["shares"]
                    del holdings[tic]

            total_equity = cash + sum(
                h["shares"] * prices.get(t, h["entry_price"]) for t, h in holdings.items()
            )
            target_alloc = total_equity / max(len(ranked_list), 1)

            for tic in ranked_list:
                if tic not in prices:
                    continue
                price = prices[tic]
                current_value = holdings[tic]["shares"] * price if tic in holdings else 0
                delta_value = target_alloc - current_value
                if delta_value > price:
                    shares_to_buy = int(delta_value // (price * 1.001))
                    if shares_to_buy > 0 and cash >= shares_to_buy * price * 1.001:
                        cost = shares_to_buy * price * 1.001
                        cash -= cost
                        if tic in holdings:
                            holdings[tic]["shares"] += shares_to_buy
                        else:
                            holdings[tic] = {"shares": shares_to_buy, "peak_price": price, "entry_price": price}
                        day_actions[tic] = shares_to_buy
                elif delta_value < -price and tic in holdings:
                    shares_to_sell = min(int(-delta_value // price), holdings[tic]["shares"])
                    if shares_to_sell > 0:
                        cash += shares_to_sell * price * 0.999
                        holdings[tic]["shares"] -= shares_to_sell
                        if holdings[tic]["shares"] == 0:
                            del holdings[tic]
                        day_actions[tic] = -shares_to_sell

        equity = cash + sum(h["shares"] * prices.get(t, 0) for t, h in holdings.items())
        account_history.append({"date": dt.strftime("%Y-%m-%d"), "account_value": equity})
        for tic in tickers:
            day_actions.setdefault(tic, 0)
        actions_history.append(day_actions)

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name
    return {"account": df_account, "actions": df_actions}


# ---------------------------------------------------------------------------
# STEP 4b5 — External analyst data: cross-sectional ranking by upgrade flow
# ---------------------------------------------------------------------------
def fetch_analyst_actions(tickers, cache_path):
    """
    Fetch yfinance.Ticker.upgrades_downgrades for each ticker, stash a single
    long DataFrame keyed by (tic, GradeDate). Cached because yfinance is slow
    and rate-limit-prone — refetch only when cache file is absent.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["GradeDate"])
        return df
    import yfinance as yf
    rows = []
    print(f"    fetching analyst actions for {len(tickers)} tickers...")
    for tic in tickers:
        try:
            ud = yf.Ticker(tic).upgrades_downgrades
            if ud is None or ud.empty:
                continue
            ud = ud.reset_index().copy()
            ud["tic"] = tic
            rows.append(ud)
        except Exception as e:
            print(f"      {tic}: skipped ({e})")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(cache_path, index=False)
    print(f"    cached {len(df)} analyst events to {os.path.basename(cache_path)}")
    return df


def run_analyst_rank_backtest(df_market, tickers, top_n=10, lookback=60, rebalance_days=21):
    """
    Cross-sectional analyst-momentum: each rebalance day, score each ticker by
        net_grade_changes(60d) + net_target_changes(60d)
    where each up/Raises = +1, down/Lowers = -1. Long top top_n by score.
    Uses yfinance Wall Street analyst feed (cached) — completely external to
    the news-sentiment pipeline, and the academic literature consistently
    shows analyst revisions lead price by 1-3 days.
    """
    agent_name = "RuleBased (AnalystRank)"
    print(f"  Running backtest: {agent_name}...")

    cache_path = os.path.join(OUTPUT_DIR, "analyst_actions.csv")
    actions_df = fetch_analyst_actions(tickers, cache_path)
    if actions_df.empty:
        print(f"    WARNING: No analyst data — {agent_name} skipped")
        return None
    actions_df["GradeDate"] = pd.to_datetime(actions_df["GradeDate"]).dt.tz_localize(None)
    actions_df["GradeDate"] = actions_df["GradeDate"].dt.normalize()

    # +1 / -1 / 0 score per row
    def score_row(r):
        a = str(r.get("Action", "")).lower()
        pa = str(r.get("priceTargetAction", "")).lower()
        s = 0
        if a == "up":
            s += 1
        elif a == "down":
            s -= 1
        elif a == "init":
            grade = str(r.get("ToGrade", "")).lower()
            if any(w in grade for w in ("buy", "outperform", "overweight", "positive")):
                s += 0.5
            elif any(w in grade for w in ("sell", "underperform", "underweight", "negative")):
                s -= 0.5
        if "raises" in pa or "raise" in pa:
            s += 0.5
        elif "lowers" in pa or "lower" in pa:
            s -= 0.5
        return s
    actions_df["score"] = actions_df.apply(score_row, axis=1)
    actions_df = actions_df[actions_df["score"] != 0].copy()
    print(f"    {len(actions_df)} non-neutral analyst events across {actions_df['tic'].nunique()} tickers")

    # Per-ticker price + ATR
    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start_dt = pd.Timestamp(TEST_START)
    per_ticker: dict[str, pd.DataFrame] = {}
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").copy()
        if len(tdf) < 30:
            continue
        prev_close = tdf["close"].shift()
        tr = pd.concat([
            tdf["high"] - tdf["low"],
            (tdf["high"] - prev_close).abs(),
            (tdf["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        tdf["ATR14"] = tr.rolling(14).mean()
        tdf.dropna(subset=["ATR14"], inplace=True)
        tdf = tdf[tdf["date"] >= test_start_dt].reset_index(drop=True)
        if tdf.empty:
            continue
        per_ticker[tic] = tdf

    if not per_ticker:
        return None

    # Sort actions by date for fast windowed lookup
    actions_df.sort_values("GradeDate", inplace=True)
    actions_by_tic = {tic: sub for tic, sub in actions_df.groupby("tic", sort=False)}

    all_dates = sorted(set().union(*(t["date"].tolist() for t in per_ticker.values())))
    cash = INITIAL_AMOUNT
    holdings: dict[str, dict] = {}
    account_history = []
    actions_history = []
    days_since_rebal = rebalance_days
    lookback_td = pd.Timedelta(days=lookback)

    for dt in all_dates:
        prices, atrs = {}, {}
        for tic, tdf in per_ticker.items():
            row = tdf[tdf["date"] == dt]
            if row.empty:
                continue
            prices[tic] = float(row["close"].iloc[0])
            atrs[tic] = float(row["ATR14"].iloc[0])

        cash *= (1 + DAILY_CASH_YIELD)

        for tic in list(holdings.keys()):
            if tic not in prices:
                continue
            h = holdings[tic]
            price = prices[tic]
            h["peak_price"] = max(h["peak_price"], price)
            atr = atrs[tic] if np.isfinite(atrs[tic]) and atrs[tic] > 0 else price * 0.02
            stop = h["peak_price"] - TRAILING_ATR_MULT_TREND * atr
            if price < stop:
                cash += h["shares"] * price * 0.999
                del holdings[tic]

        day_actions = {"date": dt.strftime("%Y-%m-%d")}
        days_since_rebal += 1
        if days_since_rebal >= rebalance_days and prices:
            days_since_rebal = 0
            window_start = dt - lookback_td
            scores = {}
            for tic in prices:
                sub = actions_by_tic.get(tic)
                if sub is None:
                    continue
                window = sub[(sub["GradeDate"] >= window_start) & (sub["GradeDate"] <= dt)]
                if not window.empty:
                    scores[tic] = float(window["score"].sum())

            if not scores:
                ranked_list = []
            else:
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                ranked_list = [t for t, sc in ranked[:top_n] if sc > 0]
            target_set = set(ranked_list)

            for tic in sorted(holdings.keys()):
                if tic not in target_set and tic in prices:
                    h = holdings[tic]
                    cash += h["shares"] * prices[tic] * 0.999
                    day_actions[tic] = -h["shares"]
                    del holdings[tic]

            total_equity = cash + sum(
                h["shares"] * prices.get(t, h["entry_price"]) for t, h in holdings.items()
            )
            target_alloc = total_equity / max(len(ranked_list), 1)

            for tic in ranked_list:
                if tic not in prices:
                    continue
                price = prices[tic]
                current_value = holdings[tic]["shares"] * price if tic in holdings else 0
                delta_value = target_alloc - current_value
                if delta_value > price:
                    shares_to_buy = int(delta_value // (price * 1.001))
                    if shares_to_buy > 0 and cash >= shares_to_buy * price * 1.001:
                        cost = shares_to_buy * price * 1.001
                        cash -= cost
                        if tic in holdings:
                            holdings[tic]["shares"] += shares_to_buy
                        else:
                            holdings[tic] = {"shares": shares_to_buy, "peak_price": price, "entry_price": price}
                        day_actions[tic] = shares_to_buy

        equity = cash + sum(h["shares"] * prices.get(t, 0) for t, h in holdings.items())
        account_history.append({"date": dt.strftime("%Y-%m-%d"), "account_value": equity})
        for tic in tickers:
            day_actions.setdefault(tic, 0)
        actions_history.append(day_actions)

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name
    return {"account": df_account, "actions": df_actions}


# ---------------------------------------------------------------------------
# STEP 4b6 — Insider trading flow (SEC Form 4s, via yfinance)
# ---------------------------------------------------------------------------
def fetch_insider_transactions(tickers, cache_path):
    """Cached pull of yfinance.Ticker.insider_transactions per ticker."""
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Start Date"])
        return df
    import yfinance as yf
    rows = []
    print(f"    fetching insider transactions for {len(tickers)} tickers...")
    for tic in tickers:
        try:
            ins = yf.Ticker(tic).insider_transactions
            if ins is None or ins.empty:
                continue
            ins = ins.copy()
            ins["tic"] = tic
            rows.append(ins)
        except Exception as e:
            print(f"      {tic}: skipped ({e})")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(cache_path, index=False)
    print(f"    cached {len(df)} insider events to {os.path.basename(cache_path)}")
    return df


def run_insider_rank_backtest(df_market, tickers, top_n=10, lookback=60, rebalance_days=21):
    """
    Cross-sectional insider-flow ranking. Each rebalance day, score each ticker
    by net DISCRETIONARY insider dollar flow over `lookback` days:
        +Value for "Purchase at price..."
        -Value for "Sale at price..."
        zero for grants/awards/option exercises/tax withholdings
    Long top-N by score. The literature (Lakonishok-Lee 2001; Cohen-Malloy-Pomorski
    2012) consistently shows discretionary insider buying leads price by weeks.
    For large-caps where buys are rare, the signal degrades to "least-negative"
    selling — still mildly predictive per Jeng-Metrick-Zeckhauser 2003.
    """
    agent_name = "RuleBased (InsiderRank)"
    print(f"  Running backtest: {agent_name}...")

    cache_path = os.path.join(OUTPUT_DIR, "insider_transactions.csv")
    ins_df = fetch_insider_transactions(tickers, cache_path)
    if ins_df.empty:
        print(f"    WARNING: No insider data — {agent_name} skipped")
        return None
    ins_df["Start Date"] = pd.to_datetime(ins_df["Start Date"], errors="coerce")
    ins_df = ins_df.dropna(subset=["Start Date"]).copy()

    # Classify each row by parsing Text
    def discretionary_value(row):
        txt = str(row.get("Text", "")).lower()
        val = row.get("Value", 0) or 0
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = 0
        if not np.isfinite(val):
            val = 0
        if "purchase" in txt:
            return val  # buy → positive
        if "sale" in txt and "tax" not in txt:
            return -val  # discretionary sell → negative
        # Award/grant/exercise/tax → neutral, not a discretionary signal
        return 0.0
    ins_df["signed_value"] = ins_df.apply(discretionary_value, axis=1)
    nz = ins_df[ins_df["signed_value"] != 0]
    n_buys = (nz["signed_value"] > 0).sum()
    n_sells = (nz["signed_value"] < 0).sum()
    print(f"    {len(nz)} discretionary insider events ({n_buys} buys, {n_sells} sells) "
          f"across {nz['tic'].nunique()} tickers")

    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start_dt = pd.Timestamp(TEST_START)
    per_ticker: dict[str, pd.DataFrame] = {}
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").copy()
        if len(tdf) < 30:
            continue
        prev_close = tdf["close"].shift()
        tr = pd.concat([
            tdf["high"] - tdf["low"],
            (tdf["high"] - prev_close).abs(),
            (tdf["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        tdf["ATR14"] = tr.rolling(14).mean()
        tdf.dropna(subset=["ATR14"], inplace=True)
        tdf = tdf[tdf["date"] >= test_start_dt].reset_index(drop=True)
        if tdf.empty:
            continue
        per_ticker[tic] = tdf
    if not per_ticker:
        return None

    ins_df.sort_values("Start Date", inplace=True)
    by_tic = {tic: sub for tic, sub in ins_df.groupby("tic", sort=False)}
    all_dates = sorted(set().union(*(t["date"].tolist() for t in per_ticker.values())))
    cash = INITIAL_AMOUNT
    holdings: dict[str, dict] = {}
    account_history = []
    actions_history = []
    days_since_rebal = rebalance_days
    lookback_td = pd.Timedelta(days=lookback)

    for dt in all_dates:
        prices, atrs = {}, {}
        for tic, tdf in per_ticker.items():
            row = tdf[tdf["date"] == dt]
            if row.empty:
                continue
            prices[tic] = float(row["close"].iloc[0])
            atrs[tic] = float(row["ATR14"].iloc[0])

        cash *= (1 + DAILY_CASH_YIELD)

        for tic in list(holdings.keys()):
            if tic not in prices:
                continue
            h = holdings[tic]
            price = prices[tic]
            h["peak_price"] = max(h["peak_price"], price)
            atr = atrs[tic] if np.isfinite(atrs[tic]) and atrs[tic] > 0 else price * 0.02
            stop = h["peak_price"] - TRAILING_ATR_MULT_TREND * atr
            if price < stop:
                cash += h["shares"] * price * 0.999
                del holdings[tic]

        day_actions = {"date": dt.strftime("%Y-%m-%d")}
        days_since_rebal += 1
        if days_since_rebal >= rebalance_days and prices:
            days_since_rebal = 0
            window_start = dt - lookback_td
            scores = {}
            for tic in prices:
                sub = by_tic.get(tic)
                if sub is None:
                    continue
                window = sub[(sub["Start Date"] >= window_start) & (sub["Start Date"] <= dt)]
                if not window.empty and (window["signed_value"] != 0).any():
                    scores[tic] = float(window["signed_value"].sum())

            # Cross-sectional ranking — for large-caps signed_value is mostly
            # negative, so we pick the least-negative (top by raw value).
            ranked_list = []
            if scores:
                ranked = sorted(scores.items(), key=lambda x: -x[1])
                ranked_list = [t for t, _ in ranked[:top_n]]
            target_set = set(ranked_list)

            for tic in sorted(holdings.keys()):
                if tic not in target_set and tic in prices:
                    h = holdings[tic]
                    cash += h["shares"] * prices[tic] * 0.999
                    day_actions[tic] = -h["shares"]
                    del holdings[tic]

            total_equity = cash + sum(
                h["shares"] * prices.get(t, h["entry_price"]) for t, h in holdings.items()
            )
            target_alloc = total_equity / max(len(ranked_list), 1)

            for tic in ranked_list:
                if tic not in prices:
                    continue
                price = prices[tic]
                current_value = holdings[tic]["shares"] * price if tic in holdings else 0
                delta_value = target_alloc - current_value
                if delta_value > price:
                    shares_to_buy = int(delta_value // (price * 1.001))
                    if shares_to_buy > 0 and cash >= shares_to_buy * price * 1.001:
                        cost = shares_to_buy * price * 1.001
                        cash -= cost
                        if tic in holdings:
                            holdings[tic]["shares"] += shares_to_buy
                        else:
                            holdings[tic] = {"shares": shares_to_buy, "peak_price": price, "entry_price": price}
                        day_actions[tic] = shares_to_buy

        equity = cash + sum(h["shares"] * prices.get(t, 0) for t, h in holdings.items())
        account_history.append({"date": dt.strftime("%Y-%m-%d"), "account_value": equity})
        for tic in tickers:
            day_actions.setdefault(tic, 0)
        actions_history.append(day_actions)

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name
    return {"account": df_account, "actions": df_actions}


# ---------------------------------------------------------------------------
# STEP 4b7 — Supervised meta-model: GradientBoostingClassifier over
# price + analyst + earnings + sentiment features, used as a ranking signal.
# ---------------------------------------------------------------------------
def _build_meta_features(df_market, tickers, analyst_actions_df,
                         earnings_by_tic, df_sentiment):
    """Compute one feature row per (date, tic) across the FULL history.
    Includes a forward 21-day return for label construction. Forward returns
    near the end of the series will be NaN — those rows are dropped before
    training/scoring."""
    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Pre-shape analyst data: per-ticker, sorted by date, with already-scored events.
    if analyst_actions_df is not None and not analyst_actions_df.empty:
        a = analyst_actions_df.copy()
        a["GradeDate"] = pd.to_datetime(a["GradeDate"]).dt.tz_localize(None).dt.normalize()
        if "score" not in a.columns:
            # Reuse the AnalystRank scoring logic
            def score_row(r):
                act = str(r.get("Action", "")).lower()
                pa = str(r.get("priceTargetAction", "")).lower()
                s = 0.0
                if act == "up":
                    s += 1.0
                elif act == "down":
                    s -= 1.0
                elif act == "init":
                    g = str(r.get("ToGrade", "")).lower()
                    if any(w in g for w in ("buy", "outperform", "overweight", "positive")):
                        s += 0.5
                    elif any(w in g for w in ("sell", "underperform", "underweight", "negative")):
                        s -= 0.5
                if "raises" in pa or "raise" in pa:
                    s += 0.5
                elif "lowers" in pa or "lower" in pa:
                    s -= 0.5
                return s
            a["score"] = a.apply(score_row, axis=1)
        analyst_by_tic = {tic: sub.sort_values("GradeDate") for tic, sub in a.groupby("tic", sort=False)}
    else:
        analyst_by_tic = {}

    # Sentiment pivot (date × ticker)
    sent_pivot = None
    if df_sentiment is not None and not df_sentiment.empty:
        sf = df_sentiment.copy()
        sf["date"] = pd.to_datetime(sf["date"])
        sent_pivot = sf.pivot_table(
            index="date", columns="ticker",
            values="weighted_avg_sentiment", aggfunc="mean",
        ).sort_index()

    rows = []
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").reset_index(drop=True)
        if len(tdf) < 100:
            continue
        # Price-side features
        tdf["mom_5d"] = tdf["close"].pct_change(5)
        tdf["mom_21d"] = tdf["close"].pct_change(21)
        tdf["mom_63d"] = tdf["close"].pct_change(63)
        delta = tdf["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        tdf["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        tdf["vol_60d"] = tdf["close"].pct_change().rolling(60).std()
        # Forward 21-day return — the basis for the cross-sectional label
        tdf["fwd_21d_ret"] = tdf["close"].shift(-21) / tdf["close"] - 1

        a_sub = analyst_by_tic.get(tic)
        earn_dates = earnings_by_tic.get(tic, []) if earnings_by_tic else []

        # Per-ticker sentiment series (forward-filled to align with trading days)
        if sent_pivot is not None and tic in sent_pivot.columns:
            sent_series = sent_pivot[tic].reindex(tdf["date"]).ffill().fillna(0).values
        else:
            sent_series = np.zeros(len(tdf))

        for i, r in tdf.iterrows():
            if not np.isfinite(r.get("mom_63d", np.nan)) or not np.isfinite(r.get("rsi14", np.nan)):
                continue
            dt = r["date"]
            close = float(r["close"])

            # Analyst features (windowed)
            net_60d = 0.0
            n_events = 0
            days_since_up = 365
            days_since_down = 365
            target_upside = 0.0
            if a_sub is not None and not a_sub.empty:
                window60 = a_sub[(a_sub["GradeDate"] >= dt - pd.Timedelta(days=60))
                                  & (a_sub["GradeDate"] <= dt)]
                if not window60.empty:
                    net_60d = float(window60["score"].sum())
                    n_events = int((window60["score"] != 0).sum())
                ups = a_sub[(a_sub["score"] > 0) & (a_sub["GradeDate"] <= dt)]
                downs = a_sub[(a_sub["score"] < 0) & (a_sub["GradeDate"] <= dt)]
                if not ups.empty:
                    days_since_up = min((dt - ups["GradeDate"].max()).days, 365)
                if not downs.empty:
                    days_since_down = min((dt - downs["GradeDate"].max()).days, 365)
                tgt = a_sub[(a_sub["GradeDate"] >= dt - pd.Timedelta(days=90))
                             & (a_sub["GradeDate"] <= dt)]
                if "currentPriceTarget" in tgt.columns:
                    tgts = tgt["currentPriceTarget"].dropna()
                    tgts = tgts[tgts > 0]
                    if not tgts.empty and close > 0:
                        target_upside = float(tgts.median() / close - 1)

            near_earn = 0
            last_eps_surprise_pct = 0.0
            days_since_earnings = 365
            for ed_entry in earn_dates:
                # earn_dates may be list of Timestamps OR list of (Timestamp, surprise) tuples
                if isinstance(ed_entry, tuple):
                    ed, sp = ed_entry
                else:
                    ed, sp = ed_entry, 0.0
                if abs((dt - ed).days) <= 5:
                    near_earn = 1
                # Track most recent past earnings (for PEAD)
                if ed <= dt:
                    delta_days = (dt - ed).days
                    if delta_days < days_since_earnings:
                        days_since_earnings = delta_days
                        last_eps_surprise_pct = float(sp) if np.isfinite(sp) else 0.0
            # Exponential post-earnings drift decay: stronger right after the
            # release, fades over ~30 trading days (Bernard-Thomas 1989; PEAD).
            pead_decay = float(last_eps_surprise_pct * np.exp(-days_since_earnings / 30.0))

            # Sentiment 3-day MA at this row
            i_lo = max(0, i - 2)
            sent_3d = float(np.mean(sent_series[i_lo:i + 1]))

            rows.append({
                "date": dt, "tic": tic, "close": close,
                "mom_5d": r["mom_5d"], "mom_21d": r["mom_21d"], "mom_63d": r["mom_63d"],
                "rsi14": r["rsi14"], "vol_60d": r["vol_60d"],
                "net_60d_analyst": net_60d, "n_analyst_events_60d": n_events,
                "days_since_up": days_since_up, "days_since_down": days_since_down,
                "target_upside": target_upside,
                "near_earnings": near_earn, "sent_3d": sent_3d,
                "sent_x_near_earn": sent_3d * near_earn,
                # PEAD features (Bernard-Thomas 1989): EPS surprise lingers for
                # ~30 trading days post-release. Both raw last-surprise and an
                # exp(-days/30) decayed version so the GBM can pick the form.
                "last_eps_surprise_pct": last_eps_surprise_pct,
                "pead_decay": pead_decay,
                "fwd_21d_ret": r.get("fwd_21d_ret", np.nan),
            })

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        return feat_df
    # Cross-sectional binary label: above the universe median next-21d return.
    feat_df["label"] = feat_df.groupby("date")["fwd_21d_ret"].transform(
        lambda x: (x > x.median()).astype(int) if x.notna().sum() >= 5 else np.nan
    )
    return feat_df


def run_meta_model_backtest(df_market, tickers, df_sentiment, top_n=10, rebalance_days=21,
                             feature_cols=None, agent_name="RuleBased (MetaModel)",
                             prob_weighted=False):
    """
    Supervised meta-model. Trains a sklearn GradientBoostingClassifier on
    rows from BEFORE the test window, using cross-sectional next-21d return
    rank as the label. Default features blend price, analyst, earnings, and
    sentiment data. Pass `feature_cols` / `agent_name` to deploy variants
    side-by-side (e.g. the 2-feature SentimentMeta variant).
    """
    print(f"  Running backtest: {agent_name}...")

    from sklearn.ensemble import GradientBoostingClassifier

    analyst_cache = os.path.join(OUTPUT_DIR, "analyst_actions.csv")
    if not os.path.exists(analyst_cache):
        print(f"    WARNING: {analyst_cache} missing — run AnalystRank first; skipping {agent_name}")
        return None
    analyst_df = pd.read_csv(analyst_cache, parse_dates=["GradeDate"])

    earnings_cache = os.path.join(OUTPUT_DIR, "earnings_dates.csv")
    # Use the shared fetch helper so the (date, surprise_pct) format propagates
    # consistently — including auto-refresh of legacy date-only caches.
    earnings_by_tic = fetch_earnings_dates(tickers, earnings_cache)

    feats = _build_meta_features(df_market, tickers, analyst_df, earnings_by_tic, df_sentiment)
    if feats.empty:
        print(f"    WARNING: empty feature matrix — {agent_name} skipped")
        return None

    if feature_cols is None:
        feature_cols = [
            "mom_5d", "mom_21d", "mom_63d", "rsi14", "vol_60d",
            "net_60d_analyst", "n_analyst_events_60d",
            "days_since_up", "days_since_down", "target_upside",
            "near_earnings", "sent_3d", "sent_x_near_earn",
            # PEAD features (Ablation F)
            "last_eps_surprise_pct", "pead_decay",
        ]

    test_start = pd.Timestamp(TEST_START)
    train = feats[(feats["date"] < test_start) & feats["label"].notna()].dropna(subset=feature_cols).copy()
    if len(train) < 1000:
        print(f"    WARNING: only {len(train)} training rows — {agent_name} skipped")
        return None
    X_train, y_train = train[feature_cols], train["label"].astype(int)
    print(f"    training GBM on {len(train)} rows, {len(feature_cols)} features...")
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"    GBM train accuracy: {train_acc:.3f}")

    # Test-window scoring frame
    test = feats[(feats["date"] >= test_start)].dropna(subset=feature_cols).copy()
    test["prob"] = clf.predict_proba(test[feature_cols])[:, 1]

    # Trade simulation — same skeleton as AnalystRank
    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])
    per_ticker = {}
    for tic in tickers:
        tdf = df[df["tic"] == tic].sort_values("date").copy()
        if len(tdf) < 30:
            continue
        prev_close = tdf["close"].shift()
        tr = pd.concat([
            tdf["high"] - tdf["low"],
            (tdf["high"] - prev_close).abs(),
            (tdf["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        tdf["ATR14"] = tr.rolling(14).mean()
        tdf.dropna(subset=["ATR14"], inplace=True)
        tdf = tdf[tdf["date"] >= test_start].reset_index(drop=True)
        if tdf.empty:
            continue
        per_ticker[tic] = tdf
    if not per_ticker:
        return None

    probs_by_date = {dt: g.set_index("tic")["prob"].to_dict()
                     for dt, g in test.groupby("date")}

    all_dates = sorted(set().union(*(t["date"].tolist() for t in per_ticker.values())))
    cash = INITIAL_AMOUNT
    holdings = {}
    account_history = []
    actions_history = []
    days_since_rebal = rebalance_days

    for dt in all_dates:
        prices, atrs = {}, {}
        for tic, tdf in per_ticker.items():
            row = tdf[tdf["date"] == dt]
            if row.empty:
                continue
            prices[tic] = float(row["close"].iloc[0])
            atrs[tic] = float(row["ATR14"].iloc[0])

        cash *= (1 + DAILY_CASH_YIELD)

        # ATR stops on existing holdings
        for tic in list(holdings.keys()):
            if tic not in prices:
                continue
            h = holdings[tic]
            price = prices[tic]
            h["peak_price"] = max(h["peak_price"], price)
            atr = atrs[tic] if np.isfinite(atrs[tic]) and atrs[tic] > 0 else price * 0.02
            stop = h["peak_price"] - TRAILING_ATR_MULT_TREND * atr
            if price < stop:
                cash += h["shares"] * price * 0.999
                del holdings[tic]

        day_actions = {"date": dt.strftime("%Y-%m-%d")}
        days_since_rebal += 1
        if days_since_rebal >= rebalance_days and prices:
            days_since_rebal = 0
            today_probs = probs_by_date.get(dt, {})
            scored = [(tic, today_probs[tic]) for tic in prices if tic in today_probs]
            scored.sort(key=lambda x: -x[1])
            top_picks = [(tic, p) for tic, p in scored[:top_n] if p > 0.5]
            target_list = [tic for tic, _ in top_picks]
            target_set = set(target_list)

            for tic in sorted(holdings.keys()):
                if tic not in target_set and tic in prices:
                    h = holdings[tic]
                    cash += h["shares"] * prices[tic] * 0.999
                    day_actions[tic] = -h["shares"]
                    del holdings[tic]

            total_equity = cash + sum(
                h["shares"] * prices.get(t, h["entry_price"]) for t, h in holdings.items()
            )

            # Build per-ticker target allocation. Equal-weight (default) or
            # weighted by (P - 0.5) so higher-conviction picks get more capital.
            if prob_weighted and top_picks:
                excess = [(tic, max(p - 0.5, 0.0)) for tic, p in top_picks]
                total_excess = sum(w for _, w in excess)
                if total_excess > 0:
                    weights = {tic: w / total_excess for tic, w in excess}
                else:
                    weights = {tic: 1.0 / len(top_picks) for tic, _ in top_picks}
            else:
                weights = {tic: 1.0 / max(len(target_list), 1) for tic in target_list}

            for tic in target_list:
                if tic not in prices:
                    continue
                price = prices[tic]
                target_alloc = total_equity * weights.get(tic, 0.0)
                cur_val = holdings[tic]["shares"] * price if tic in holdings else 0
                delta = target_alloc - cur_val
                if delta > price:
                    n = int(delta // (price * 1.001))
                    if n > 0 and cash >= n * price * 1.001:
                        cash -= n * price * 1.001
                        if tic in holdings:
                            holdings[tic]["shares"] += n
                        else:
                            holdings[tic] = {"shares": n, "peak_price": price, "entry_price": price}
                        day_actions[tic] = n

        equity = cash + sum(h["shares"] * prices.get(t, 0) for t, h in holdings.items())
        account_history.append({"date": dt.strftime("%Y-%m-%d"), "account_value": equity})
        for tic in tickers:
            day_actions.setdefault(tic, 0)
        actions_history.append(day_actions)

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name
    return {"account": df_account, "actions": df_actions}


def run_meta_model_pweighted_backtest(df_market, tickers, df_sentiment,
                                       top_n=10, rebalance_days=21):
    """
    MetaModel_PWeighted — the 13-feature MetaModel with conviction-weighted
    position sizing. Each held name gets capital proportional to (P - 0.5)
    instead of equal weight. Higher-confidence picks dominate the book.
    """
    return run_meta_model_backtest(
        df_market=df_market,
        tickers=tickers,
        df_sentiment=df_sentiment,
        top_n=top_n,
        rebalance_days=rebalance_days,
        agent_name="RuleBased (MetaModel_PWeighted)",
        prob_weighted=True,
    )


def run_sentiment_meta_pweighted_backtest(df_market, tickers, df_sentiment,
                                            top_n=10, rebalance_days=21):
    """
    SentimentMeta_PWeighted — the 2-feature SentimentMeta variant with
    conviction-weighted position sizing.
    """
    return run_meta_model_backtest(
        df_market=df_market,
        tickers=tickers,
        df_sentiment=df_sentiment,
        top_n=top_n,
        rebalance_days=rebalance_days,
        feature_cols=["sent_3d", "near_earnings"],
        agent_name="RuleBased (SentimentMeta_PWeighted)",
        prob_weighted=True,
    )


def run_sentiment_meta_backtest(df_market, tickers, df_sentiment, top_n=10, rebalance_days=21):
    """
    SentimentMeta — the 2-feature variant of MetaModel.
    Trained on the same labels as MetaModel but with only `sent_3d` and
    `near_earnings` as inputs. Walk-forward CV (results/ablation/
    meta_walkforward.csv) showed this variant has the highest mean fold
    Sharpe (0.920) of the three configurations tested, at the cost of higher
    fold-to-fold variance. We deploy it standalone alongside the 13-feature
    MetaModel rather than replacing it, so the two variants can be compared
    directly under any future regime.
    """
    return run_meta_model_backtest(
        df_market=df_market,
        tickers=tickers,
        df_sentiment=df_sentiment,
        top_n=top_n,
        rebalance_days=rebalance_days,
        feature_cols=["sent_3d", "near_earnings"],
        agent_name="RuleBased (SentimentMeta)",
    )


# ---------------------------------------------------------------------------
# STEP 4c — Ensemble: combine DRL + rule-based agent signals
# ---------------------------------------------------------------------------
def run_ensemble_backtest(results, df_market, tickers, df_sentiment):
    """
    Ensemble meta-strategy that combines signals from DRL and rule-based agents.

    For each (date, ticker):
    - DRL agents vote via position change direction (buy/sell/hold)
    - Rule-based agents vote via their action columns
    - Votes are weighted by each agent's rolling 30-day Sharpe ratio
    - Majority weighted vote determines the ensemble action
    - Position sizing uses conviction (vote margin) + volatility weighting
    """
    agent_name = "Ensemble"
    print(f"  Building {agent_name} from {len(results)} sub-agents...")

    # We need agents that trade the DRL tickers (original 27)
    # Best 4-voter ensemble: RSI + RegimeAdaptive + CrossMomentum + AnalystRank
    # (Sharpe 1.368). MetaModel was tested both as a swap-in (1.214) and as a
    # 5th voter (1.251) — both regressed. The GBM ingests price/momentum/
    # sentiment/analyst features so its votes correlate with the existing
    # voters, paying a soft-Sharpe weight-dilution tax with no diversification
    # benefit. MetaModel stays as a documented standalone strategy.
    drl_agents = ["PPO", "A2C", "DDPG"]
    rb_agents = [
        "RuleBased (RSI)",
        "RuleBased (RegimeAdaptive)",
        "RuleBased (CrossMomentum)",
        "RuleBased (AnalystRank)",
    ]
    ensemble_agents = [a for a in drl_agents + rb_agents if a in results]

    if len(ensemble_agents) < 2:
        print("    WARNING: Need at least 2 agents for ensemble")
        return None

    print(f"    Using agents: {ensemble_agents}")

    # Build per-agent action DataFrames: date → ticker → action (shares delta)
    # DRL actions are current holdings; we need deltas
    agent_actions = {}
    agent_accounts = {}
    for ag in ensemble_agents:
        data = results[ag]
        acct = data["account"].copy()
        acct["date"] = pd.to_datetime(acct["date"]).dt.strftime("%Y-%m-%d")
        agent_accounts[ag] = acct.set_index("date")["account_value"].to_dict()

        acts = data["actions"].copy()
        if "date" in acts.columns:
            acts["date"] = pd.to_datetime(acts["date"]).dt.strftime("%Y-%m-%d")
            acts = acts.set_index("date")
        acts = acts.drop(columns=["agent"], errors="ignore")

        if ag in drl_agents:
            # DRL actions are holdings → compute deltas
            acts = acts.apply(pd.to_numeric, errors="coerce").fillna(0)
            deltas = acts.diff().fillna(0)
            agent_actions[ag] = deltas
        else:
            # Rule-based actions are already deltas (shares bought/sold)
            acts = acts.apply(pd.to_numeric, errors="coerce").fillna(0)
            agent_actions[ag] = acts

    # Get common dates and tickers
    all_dates_sets = [set(agent_actions[ag].index) for ag in ensemble_agents]
    common_dates = sorted(set.intersection(*all_dates_sets))

    # Get tickers that at least some agents trade
    all_tics = set()
    for ag in ensemble_agents:
        all_tics.update(agent_actions[ag].columns)
    trade_tickers = sorted(all_tics)

    if not common_dates or not trade_tickers:
        print("    WARNING: No common dates/tickers for ensemble")
        return None

    print(f"    {len(common_dates)} common dates, {len(trade_tickers)} tickers")

    # Build price + ATR lookup from market data
    df_m = df_market.copy()
    df_m["date"] = pd.to_datetime(df_m["date"]).dt.strftime("%Y-%m-%d")
    price_lookup = {}
    atr_lookup = {}
    for tic, sub in df_m.groupby("tic", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        prev_close = sub["close"].shift()
        tr = pd.concat([
            sub["high"] - sub["low"],
            (sub["high"] - prev_close).abs(),
            (sub["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        sub["ATR14"] = tr.rolling(14).mean()
        for _, r in sub.iterrows():
            key = (r["date"], tic)
            price_lookup[key] = r["close"]
            atr_lookup[key] = r["ATR14"]

    # Compute rolling 30-day Sharpe for each agent → adaptive weights
    def rolling_sharpe(account_dict, dates, window=30):
        """Compute rolling sharpe for weight calculation."""
        values = [account_dict.get(d, np.nan) for d in dates]
        sharpes = {}
        for i, d in enumerate(dates):
            if i < window:
                sharpes[d] = 0.0
                continue
            window_vals = [v for v in values[max(0, i - window):i + 1] if not np.isnan(v)]
            if len(window_vals) < 10:
                sharpes[d] = 0.0
                continue
            rets = np.diff(window_vals) / np.array(window_vals[:-1])
            if np.std(rets) > 0:
                sharpes[d] = np.mean(rets) / np.std(rets)
            else:
                sharpes[d] = 0.0
        return sharpes

    agent_sharpes = {}
    for ag in ensemble_agents:
        agent_sharpes[ag] = rolling_sharpe(agent_accounts[ag], common_dates)

    # Simulate portfolio
    n_tics = len(trade_tickers)
    cash_per_stock = INITIAL_AMOUNT / n_tics
    positions = {
        tic: {
            "shares": 0,
            "cash": cash_per_stock,
            "peak_price": 0.0,
            "entry_price": 0.0,
            "entry_shares": 0,
            "profit_lvl": 0,
        }
        for tic in trade_tickers
    }

    account_history = []
    actions_history = []

    for dt in common_dates:
        total_value = 0
        day_actions = {"date": dt}

        for tic in trade_tickers:
            pos = positions[tic]
            # Daily cash yield
            pos["cash"] *= (1 + DAILY_CASH_YIELD)
            price = price_lookup.get((dt, tic))
            if price is None or np.isnan(price):
                total_value += pos["cash"]
                day_actions[tic] = 0
                continue

            price = float(price)
            atr_val = atr_lookup.get((dt, tic), price * 0.02)
            if not np.isfinite(atr_val) or atr_val <= 0:
                atr_val = price * 0.02

            # Gather weighted votes — only count agents with actual opinions
            buy_weight = 0.0
            sell_weight = 0.0
            n_voters = 0

            for ag in ensemble_agents:
                acts_df = agent_actions[ag]
                if dt in acts_df.index and tic in acts_df.columns:
                    action_val = acts_df.loc[dt, tic]
                else:
                    continue  # Agent doesn't trade this ticker — skip, don't count as hold

                if np.isnan(action_val) or action_val == 0:
                    continue  # No opinion — skip

                # Agent weight: softmax of rolling sharpe
                raw_sharpe = agent_sharpes[ag].get(dt, 0.0)
                weight = np.exp(raw_sharpe * 5)

                n_voters += 1
                if action_val > 0:
                    buy_weight += weight
                else:
                    sell_weight += weight

            # Need at least 1 voter with an opinion
            total_weight = buy_weight + sell_weight
            if total_weight == 0 or n_voters == 0:
                total_value += pos["cash"] + pos["shares"] * price
                day_actions[tic] = 0
                continue

            # Decision: whichever side has more weight wins
            # Conviction = margin (0 = split, 1 = unanimous)
            if buy_weight >= sell_weight:
                decision = "buy"
                conviction = (buy_weight - sell_weight) / total_weight
            else:
                decision = "sell"
                conviction = (sell_weight - buy_weight) / total_weight

            action = 0
            stop_triggered = False

            # ATR-based trailing stop check
            if pos["shares"] > 0:
                pos["peak_price"] = max(pos["peak_price"], price)
                stop_level = pos["peak_price"] - TRAILING_ATR_MULT * atr_val
                if price < stop_level:
                    decision = "sell"
                    conviction = 1.0
                    stop_triggered = True

                # Profit ladder: 1/3 at +T1, another 1/3 at +T2
                if not stop_triggered and pos["entry_price"] > 0:
                    gain = (price - pos["entry_price"]) / pos["entry_price"]
                    third = max(1, pos["entry_shares"] // 3)
                    if pos["profit_lvl"] == 0 and gain >= PROFIT_T1 and pos["shares"] >= third:
                        proceeds = third * price * 0.999
                        pos["cash"] += proceeds
                        pos["shares"] -= third
                        pos["profit_lvl"] = 1
                        action = -third
                    elif pos["profit_lvl"] == 1 and gain >= PROFIT_T2 and pos["shares"] >= third:
                        proceeds = third * price * 0.999
                        pos["cash"] += proceeds
                        pos["shares"] -= third
                        pos["profit_lvl"] = 2
                        action = -third

            if decision == "buy" and pos["shares"] == 0 and pos["cash"] > price:
                # Size by conviction: higher margin → larger position
                size_frac = np.clip(0.5 + conviction, 0.3, 1.0)
                buyable = pos["cash"] * size_frac
                shares_to_buy = int(buyable // (price * 1.001))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * 1.001
                    pos["shares"] = shares_to_buy
                    pos["cash"] -= cost
                    pos["peak_price"] = price
                    pos["entry_price"] = price
                    pos["entry_shares"] = shares_to_buy
                    pos["profit_lvl"] = 0
                    action = shares_to_buy

            elif decision == "sell" and pos["shares"] > 0:
                # Stop / strong-conviction signal exits the rest of the position fully.
                if stop_triggered or conviction >= 0.3:
                    shares_to_sell = pos["shares"]
                else:
                    shares_to_sell = max(1, pos["shares"] // 2)
                proceeds = shares_to_sell * price * 0.999
                action = -shares_to_sell
                pos["cash"] += proceeds
                pos["shares"] -= shares_to_sell
                if pos["shares"] == 0:
                    pos["peak_price"] = 0.0
                    pos["entry_price"] = 0.0
                    pos["entry_shares"] = 0
                    pos["profit_lvl"] = 0

            total_value += pos["cash"] + pos["shares"] * price
            day_actions[tic] = action

        account_history.append({"date": dt, "account_value": total_value})
        actions_history.append(day_actions)

    df_account = pd.DataFrame(account_history)
    df_account["agent"] = agent_name
    df_actions = pd.DataFrame(actions_history)
    df_actions["agent"] = agent_name

    return {"account": df_account, "actions": df_actions}


# ---------------------------------------------------------------------------
# STEP 5 — Get baseline (S&P 500)
# ---------------------------------------------------------------------------
def get_snp_baseline():
    print("  Downloading S&P 500 baseline...")
    import yfinance as yf
    raw = yf.download("^GSPC", start=TEST_START, end=TEST_END, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df_snp = raw[["Close"]].copy()
    df_snp.columns = ["close"]
    df_snp["date"] = df_snp.index
    df_snp = df_snp.reset_index(drop=True)
    df_snp["date"] = pd.to_datetime(df_snp["date"]).dt.strftime("%Y-%m-%d")
    first_close = df_snp["close"].iloc[0]
    df_snp["account_value"] = df_snp["close"] / first_close * INITIAL_AMOUNT
    df_snp["agent"] = "S&P 500 Baseline"
    return df_snp[["date", "account_value", "agent"]]


# ---------------------------------------------------------------------------
# STEP 6 — Compute performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(df_account):
    """Compute Sharpe, return, drawdown, etc. from an account value series."""
    vals = df_account["account_value"].values.astype(float)
    dates = pd.to_datetime(df_account["date"])

    total_return = (vals[-1] - vals[0]) / vals[0]
    n_days = (dates.iloc[-1] - dates.iloc[0]).days
    annual_return = (1 + total_return) ** (365.0 / max(n_days, 1)) - 1

    daily_returns = np.diff(vals) / vals[:-1]
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0  # 4% risk-free

    # Max drawdown
    peak = np.maximum.accumulate(vals)
    drawdown = (peak - vals) / peak
    max_dd = np.max(drawdown)

    # Win rate
    win_rate = np.mean(daily_returns > 0) if len(daily_returns) > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1
    sortino = (annual_return - 0.04) / downside_std if downside_std > 0 else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "annual_volatility_pct": round(annual_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "final_value": round(vals[-1], 2),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("INTEGRATED PIPELINE: Sentiment + DRL Trading")
    print("=" * 60)

    # 1. Sentiment (Alpha Vantage news) — covers original 27 tickers
    print("\n[1/9] Preparing Alpha Vantage sentiment data...")
    df_sentiment = prepare_sentiment_data()
    drl_tickers = sorted(df_sentiment["ticker"].unique().tolist())

    # 2. Collect social media sentiment for EXPANDED ticker universe
    print(f"\n[2/9] Collecting social media sentiment for {len(EXPANDED_TICKERS)} tickers...")
    df_sm_sentiment = collect_telegram_sentiment(EXPANDED_TICKERS)

    # 2b. Alpha Vantage live NEWS_SENTIMENT (opt-in, USE_LIVE_ALPHAVANTAGE=1)
    print("\n[2b/9] Alpha Vantage live sentiment...")
    df_av_live = fetch_alphavantage_live_sentiment(EXPANDED_TICKERS)

    # 2c. Apify multi-platform scrape (opt-in, USE_APIFY=1)
    print("\n[2c/9] Apify social scrape...")
    df_apify = fetch_apify_social_sentiment(EXPANDED_TICKERS)

    # Merge every available source into Alpha Vantage cached baseline
    frames = [df_sentiment]
    for extra in (df_sm_sentiment, df_av_live, df_apify):
        if extra is not None and not extra.empty:
            extra = extra.copy()
            extra["date"] = pd.to_datetime(extra["date"]).dt.date
            frames.append(extra)

    if len(frames) > 1:
        df_combined = pd.concat(frames, ignore_index=True)
        df_sentiment = (
            df_combined.groupby(["date", "ticker"])
            .agg(
                raw_avg_sentiment=("raw_avg_sentiment", "mean"),
                total_relevance=("total_relevance", "sum"),
                weighted_sum=("weighted_sum", "sum"),
                article_count=("article_count", "sum"),
                weighted_avg_sentiment=("weighted_avg_sentiment", "mean"),
            )
            .reset_index()
        )
        df_sentiment.sort_values(["date", "ticker"], inplace=True)
        print(f"  Combined sentiment: {len(df_sentiment)} rows, {df_sentiment['ticker'].nunique()} tickers "
              f"(sources: HF+{sum(1 for f in frames[1:])})")
    else:
        print("  No extra sentiment sources active — using Alpha Vantage cached only")

    # 3. Market data for original 27 (DRL agents)
    print(f"\n[3/9] Preparing market data for {len(drl_tickers)} DRL tickers...")
    df_market = prepare_market_data(drl_tickers)

    # 4. Merge for DRL
    print("\n[4/9] Merging market + sentiment for DRL agents...")
    df_merged = merge_data(df_market, df_sentiment)
    df_merged.to_csv(os.path.join(OUTPUT_DIR, "merged_data.csv"), index=False)
    trade = data_split(df_merged, TEST_START, TEST_END)
    print(f"  Test set: {len(trade)} rows ({TEST_START} → {TEST_END})")

    # 5. Backtest DRL agents (on original 27)
    print("\n[5/9] Loading DRL agents and running backtests...")
    env_kwargs = build_env_kwargs(df_merged)
    results = load_and_backtest_agents(trade, env_kwargs)

    # 5b. LangGraph-gated backtest — analyst/manager decisions modify DRL actions
    print("\n[5b/9] Applying LangGraph signal gate to DRL agents...")
    from langgraph_signals import build_signals_cache, load_signals
    signals_df = build_signals_cache()  # rebuilds from Capstone/reports/ each run
    print(f"  Loaded {len(signals_df)} LangGraph signal(s) from Capstone/reports/")

    if signals_df.empty:
        print("  No LangGraph signals found — gated backtest skipped. "
              "Run graph/trade_generation_pipeline.py to generate reports, then re-run.")
    else:
        model_dir = os.path.join(DRL_DIR, "trained_models")
        agent_map = {"PPO": ("agent_ppo", PPO), "A2C": ("agent_a2c", A2C), "DDPG": ("agent_ddpg", DDPG)}
        all_overrides = []
        for name, (filename, ModelClass) in agent_map.items():
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path + ".zip"):
                continue
            try:
                model = ModelClass.load(path, device="cpu")
                g_acct, g_acts, overrides = run_agent_backtest_gated(
                    model, trade, env_kwargs, signals_df, name
                )
                results[f"{name}_GATED"] = {"account": g_acct, "actions": g_acts}
                for row in overrides:
                    row["agent"] = name
                all_overrides.extend(overrides)
                print(f"  {name}_GATED: {len(overrides)} action(s) overridden by LangGraph signals")
            except Exception as e:
                print(f"  ERROR running gated {name}: {e}")

        if all_overrides:
            pd.DataFrame(all_overrides).to_csv(
                os.path.join(OUTPUT_DIR, "langgraph_overrides.csv"), index=False
            )
            print(f"  Saved override log to dashboard_data/langgraph_overrides.csv")
        else:
            # Touch an empty file so dashboard can read it consistently
            pd.DataFrame(columns=[
                "date", "ticker", "original_action", "gated_action", "recommendation",
                "conviction", "sentiment_score", "reason", "signal_date", "agent",
            ]).to_csv(os.path.join(OUTPUT_DIR, "langgraph_overrides.csv"), index=False)

    # 6. Download expanded market data for rule-based strategies
    print(f"\n[6/9] Preparing expanded market data for {len(EXPANDED_TICKERS)} tickers...")
    expanded_cache = os.path.join(OUTPUT_DIR, "market_data_expanded.csv")
    if os.path.exists(expanded_cache):
        print("  Loading cached expanded market data...")
        df_market_exp = pd.read_csv(expanded_cache)
    else:
        import yfinance as yf
        print("  Downloading expanded OHLCV from Yahoo Finance...")
        all_dfs = []
        for tic in EXPANDED_TICKERS:
            try:
                raw = yf.download(tic, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
                if raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                cols = raw.columns.tolist()
                if "Adj Close" in cols:
                    tmp = raw[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
                else:
                    tmp = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                    tmp["Adj Close"] = tmp["Close"]
                tmp.columns = ["open", "high", "low", "close", "adjcp", "volume"]
                tmp["tic"] = tic
                tmp["date"] = tmp.index
                tmp = tmp.reset_index(drop=True)
                tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
                all_dfs.append(tmp)
            except Exception as e:
                print(f"    WARNING: {tic} download failed: {e}")
        df_market_exp = pd.concat(all_dfs, ignore_index=True)
        df_market_exp.sort_values(["date", "tic"], inplace=True)
        df_market_exp.to_csv(expanded_cache, index=False)
    exp_tickers = sorted(df_market_exp["tic"].unique().tolist())
    print(f"  Expanded market data: {len(df_market_exp)} rows, {len(exp_tickers)} tickers")

    # 7. Rule-based strategies on EXPANDED universe
    print(f"\n[7/9] Running rule-based strategies on {len(exp_tickers)} tickers...")
    earnings_cache = os.path.join(OUTPUT_DIR, "earnings_dates.csv")
    earnings_by_tic = fetch_earnings_dates(exp_tickers, earnings_cache)
    print(f"  Earnings dates: {len(earnings_by_tic)} tickers covered")
    for mode in ["SMA", "RSI", "SMA_RSI", "SMA_RSI_Sentiment", "Dynamic",
                 "SentimentMomentum", "EarningsSentiment"]:
        rb = run_multistrategy_backtest(
            df_market_exp, exp_tickers, mode=mode,
            df_sentiment=df_sentiment, earnings_by_tic=earnings_by_tic,
        )
        if rb is not None:
            results[f"RuleBased ({mode})"] = rb

    regime = run_regime_adaptive_backtest(df_market_exp, exp_tickers)
    if regime is not None:
        results["RuleBased (RegimeAdaptive)"] = regime

    cross_mom = run_cross_momentum_backtest(df_market_exp, exp_tickers)
    if cross_mom is not None:
        results["RuleBased (CrossMomentum)"] = cross_mom

    sent_rank = run_sentiment_rank_backtest(df_market_exp, exp_tickers, df_sentiment)
    if sent_rank is not None:
        results["RuleBased (SentimentRank)"] = sent_rank

    analyst_rank = run_analyst_rank_backtest(df_market_exp, exp_tickers)
    if analyst_rank is not None:
        results["RuleBased (AnalystRank)"] = analyst_rank

    insider_rank = run_insider_rank_backtest(df_market_exp, exp_tickers)
    if insider_rank is not None:
        results["RuleBased (InsiderRank)"] = insider_rank

    meta_model = run_meta_model_backtest(df_market_exp, exp_tickers, df_sentiment)
    if meta_model is not None:
        results["RuleBased (MetaModel)"] = meta_model

    sentiment_meta = run_sentiment_meta_backtest(df_market_exp, exp_tickers, df_sentiment)
    if sentiment_meta is not None:
        results["RuleBased (SentimentMeta)"] = sentiment_meta

    meta_pw = run_meta_model_pweighted_backtest(df_market_exp, exp_tickers, df_sentiment)
    if meta_pw is not None:
        results["RuleBased (MetaModel_PWeighted)"] = meta_pw

    sent_meta_pw = run_sentiment_meta_pweighted_backtest(df_market_exp, exp_tickers, df_sentiment)
    if sent_meta_pw is not None:
        results["RuleBased (SentimentMeta_PWeighted)"] = sent_meta_pw

    # 8. Ensemble strategy — combine DRL + rule-based signals
    print("\n[8/10] Building ensemble strategy...")
    ensemble_result = run_ensemble_backtest(results, df_market_exp, exp_tickers, df_sentiment)
    if ensemble_result is not None:
        results["Ensemble"] = ensemble_result

    # 9. Baseline
    print("\n[9/10] Getting S&P 500 baseline...")
    snp_baseline = get_snp_baseline()

    # 10. Save everything
    print("\n[10/10] Saving results...")

    # Combine all account values
    all_accounts = [snp_baseline]
    all_actions = []
    metrics_rows = []

    for agent_name, data in results.items():
        all_accounts.append(data["account"][["date", "account_value", "agent"]])
        all_actions.append(data["actions"])

        m = compute_metrics(data["account"])
        m["agent"] = agent_name
        metrics_rows.append(m)

    # S&P 500 metrics
    m = compute_metrics(snp_baseline)
    m["agent"] = "S&P 500 Baseline"
    metrics_rows.append(m)

    df_all_accounts = pd.concat(all_accounts, ignore_index=True)
    df_all_accounts.to_csv(os.path.join(OUTPUT_DIR, "account_values.csv"), index=False)

    if all_actions:
        df_all_actions = pd.concat(all_actions, ignore_index=True)
        df_all_actions.to_csv(os.path.join(OUTPUT_DIR, "actions.csv"), index=False)

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    # Save sentiment for dashboard
    df_sentiment.to_csv(os.path.join(OUTPUT_DIR, "sentiment.csv"), index=False)

    print("\n" + "=" * 60)
    print("DONE! Results saved to:", OUTPUT_DIR)
    print("=" * 60)
    print("\nPerformance Summary:")
    print(df_metrics.to_string(index=False))
    print(f"\nRun the dashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
