"""
Sentiment Ablation Study
========================
Systematically tests whether sentiment helps a rule-based trading strategy
by sweeping: {source} x {lookback window} x {signal mode}.

Base strategy: SMA(50/200) crossover + RSI(14), trailing stop 8%, 0.1% txn cost,
equal-weight across the 27 core tickers. Test window = TEST_START..TEST_END
from pipeline.py. Indicators are computed on full history to avoid warmup bias.

Outputs:
  results/ablation/ablation_results.csv   — one row per config w/ full metrics
  results/ablation/ablation_equity.csv    — equity curves, one column per config
  console pivot tables (Sharpe and total_return by source x mode x lookback)
"""
import os, sys, warnings, itertools, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")
OUT_DIR = os.path.join(BASE_DIR, "results", "ablation")
os.makedirs(OUT_DIR, exist_ok=True)

# Match pipeline.py
TEST_START = "2025-01-01"
TEST_END = "2026-04-21"
INITIAL_AMOUNT = 1_000_000
TRAILING_STOP_PCT = 0.08
TXN_COST = 0.001

CORE_TICKERS = sorted([
    'AAPL','ABBV','ACN','ADBE','AMZN','AVGO','BAC','COST','CRM','CVX','GOOG','HD',
    'KO','LLY','MA','META','MSFT','NVDA','ORCL','PEP','PG','TMO','TSLA','UNH','V','WMT','XOM'
])


# -------------------- Load sentiment sources --------------------

def load_av_sentiment():
    """Alpha Vantage news-based sentiment (Dow30), from HF-hosted CSV."""
    path = os.path.join(BASE_DIR, "sentiment-drl-trading-main", "datasets",
                        "dow30_monthly_news_sentiment.csv")
    if not os.path.exists(path):
        print(f"  [av] missing: {path}")
        return pd.DataFrame(columns=["date", "ticker", "weighted_avg_sentiment"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["published_time"]).dt.date
    df["weighted_score"] = df["ticker_sentiment_score"] * df["ticker_relevance_score"]
    agg = (df.groupby(["date", "ticker"])
             .agg(total_relevance=("ticker_relevance_score", "sum"),
                  weighted_sum=("weighted_score", "sum"))
             .reset_index())
    agg["weighted_avg_sentiment"] = (agg["weighted_sum"] / agg["total_relevance"]).fillna(0)
    agg["date"] = pd.to_datetime(agg["date"])
    return agg[["date", "ticker", "weighted_avg_sentiment"]]


def load_social_sentiment():
    """Social+news mix from collect_telegram_sentiment (HF news + yfinance +
    Telegram + NewsCollector Yahoo/Google/Finnhub)."""
    path = os.path.join(DATA_DIR, "telegram_sentiment.csv")
    if not os.path.exists(path):
        print(f"  [social] missing: {path}")
        return pd.DataFrame(columns=["date", "ticker", "weighted_avg_sentiment"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "ticker", "weighted_avg_sentiment"]]


def combine_sentiment(df_av, df_social):
    """Average the two sources on overlapping (date, ticker); take either where only one exists."""
    if df_av.empty and df_social.empty:
        return pd.DataFrame(columns=["date", "ticker", "weighted_avg_sentiment"])
    if df_av.empty:
        return df_social.copy()
    if df_social.empty:
        return df_av.copy()
    merged = pd.merge(
        df_av.rename(columns={"weighted_avg_sentiment": "s_av"}),
        df_social.rename(columns={"weighted_avg_sentiment": "s_social"}),
        on=["date", "ticker"], how="outer",
    )
    merged["weighted_avg_sentiment"] = merged[["s_av", "s_social"]].mean(axis=1, skipna=True)
    return merged[["date", "ticker", "weighted_avg_sentiment"]]


def apply_lookback(df_sent, days):
    """Rolling mean over `days` calendar days per ticker."""
    if df_sent.empty or days <= 1:
        return df_sent.copy()
    parts = []
    for tic, grp in df_sent.groupby("ticker"):
        g = grp.sort_values("date").set_index("date")
        g["weighted_avg_sentiment"] = (
            g["weighted_avg_sentiment"].rolling(f"{days}D").mean()
        )
        g = g.reset_index()
        g["ticker"] = tic
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# -------------------- Market + indicators --------------------

def load_market():
    df = pd.read_csv(os.path.join(DATA_DIR, "market_data_expanded.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["tic"].isin(CORE_TICKERS)].copy()
    return df


def precompute_signals(df_market):
    """Compute SMA50/SMA200/RSI on full history for each ticker; cache per ticker."""
    per_tic = {}
    for tic, g in df_market.groupby("tic"):
        g = g.sort_values("date").copy()
        if len(g) < 250:
            continue
        g["SMA50"] = g["close"].rolling(50).mean()
        g["SMA200"] = g["close"].rolling(200).mean()
        delta = g["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g["RSI"] = 100 - (100 / (1 + rs))
        g.dropna(inplace=True)
        g = g[g["date"] >= pd.Timestamp(TEST_START)].copy()
        if g.empty:
            continue
        # Base SMA_RSI entry/exit
        uptrend = g["SMA50"] > g["SMA200"]
        g["Buy_base"] = uptrend & (g["RSI"] < 40) & (g["RSI"].shift(1) >= 40)
        g["Sell_base"] = (~uptrend) & (g["SMA50"].shift(1) > g["SMA200"].shift(1))
        per_tic[tic] = g[["date", "close", "RSI", "Buy_base", "Sell_base"]].reset_index(drop=True)
    return per_tic


# -------------------- Backtest engine --------------------

def run_backtest(per_tic, sent_by_ticker, mode):
    """Simulate equal-weight portfolio across tickers with sentiment-modulated rules.

    mode:
      - 'none'         → ignore sentiment (baseline)
      - 'gate_nonneg'  → buy only if sent >= 0
      - 'gate_pos'     → buy only if sent >  +0.1
      - 'veto_neg'     → exit early if sent < -0.2
      - 'tilt_size'    → size position by clip(1 + 2*sent, 0.3, 1.7) of allocation
    """
    tickers = list(per_tic.keys())
    n = len(tickers)
    alloc = INITIAL_AMOUNT / n
    positions = {t: {"shares": 0, "cash": alloc, "peak": 0.0} for t in tickers}

    all_dates = sorted(set(itertools.chain.from_iterable(
        df["date"].tolist() for df in per_tic.values())))
    equity = []
    trade_count = 0
    wins = 0
    losses = 0
    active_entry_price = {t: None for t in tickers}
    sentiment_active_days = 0
    total_sig_days = 0

    for dt in all_dates:
        total = 0.0
        for t in tickers:
            df = per_tic[t]
            row = df[df["date"] == dt]
            pos = positions[t]
            if row.empty:
                total += pos["cash"] + pos["shares"] * 0  # no price update
                continue
            price = float(row["close"].iloc[0])
            buy = bool(row["Buy_base"].iloc[0])
            sell = bool(row["Sell_base"].iloc[0])
            sent = sent_by_ticker.get(t, {}).get(dt, 0.0)
            if pd.isna(sent):
                sent = 0.0

            total_sig_days += 1
            if abs(sent) > 1e-9:
                sentiment_active_days += 1

            # Apply sentiment rules to base signals
            size_mult = 1.0
            if mode == "gate_nonneg":
                if buy and sent < 0: buy = False
            elif mode == "gate_pos":
                if buy and sent <= 0.1: buy = False
            elif mode == "veto_neg":
                if pos["shares"] > 0 and sent < -0.2:
                    sell = True
            elif mode == "tilt_size":
                size_mult = float(np.clip(1.0 + 2.0 * sent, 0.3, 1.7))

            # Trailing stop
            if pos["shares"] > 0:
                pos["peak"] = max(pos["peak"], price)
                if price < pos["peak"] * (1 - TRAILING_STOP_PCT):
                    sell = True

            if buy and pos["shares"] == 0 and pos["cash"] > price:
                buy_cash = pos["cash"] * size_mult if mode == "tilt_size" else pos["cash"]
                buy_cash = min(buy_cash, pos["cash"])
                shares = int(buy_cash // (price * (1 + TXN_COST)))
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    pos["shares"] = shares
                    pos["cash"] -= cost
                    pos["peak"] = price
                    active_entry_price[t] = price
                    trade_count += 1
            elif sell and pos["shares"] > 0:
                proceeds = pos["shares"] * price * (1 - TXN_COST)
                entry = active_entry_price[t] or price
                if price > entry: wins += 1
                else: losses += 1
                pos["cash"] += proceeds
                pos["shares"] = 0
                pos["peak"] = 0.0
                active_entry_price[t] = None
                trade_count += 1

            total += pos["cash"] + pos["shares"] * price
        equity.append({"date": dt, "value": total})

    eq = pd.DataFrame(equity)
    return eq, {
        "trade_count": trade_count,
        "win_rate_pct": 100 * wins / max(1, wins + losses),
        "sentiment_coverage_pct": 100 * sentiment_active_days / max(1, total_sig_days),
    }


# -------------------- Metrics --------------------

def compute_metrics(eq):
    if eq.empty:
        return {}
    vals = eq["value"].values
    rets = pd.Series(vals).pct_change().dropna()
    total_return = (vals[-1] / vals[0] - 1) * 100
    ann_return = ((vals[-1] / vals[0]) ** (252 / max(1, len(vals))) - 1) * 100
    ann_vol = rets.std() * np.sqrt(252) * 100
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252) + 1e-12)
    downside = rets[rets < 0]
    sortino = (rets.mean() * 252) / (downside.std() * np.sqrt(252) + 1e-12) if len(downside) else np.nan
    cummax = np.maximum.accumulate(vals)
    mdd = ((vals - cummax) / cummax).min() * 100
    return {
        "total_return_pct": round(total_return, 3),
        "annual_return_pct": round(ann_return, 3),
        "annual_vol_pct": round(ann_vol, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd, 3),
        "final_value": round(vals[-1], 2),
    }


# -------------------- Ablation loop --------------------

def build_sent_lookup(df_sent):
    """Return dict[ticker] -> dict[date(Timestamp)] -> sentiment."""
    d = {}
    if df_sent is None or df_sent.empty:
        return d
    for tic, grp in df_sent.groupby("ticker"):
        d[tic] = dict(zip(pd.to_datetime(grp["date"]), grp["weighted_avg_sentiment"].astype(float)))
    return d


def main():
    t0 = time.time()
    print("=" * 70)
    print("  SENTIMENT ABLATION STUDY")
    print("=" * 70)

    print("\n[1/4] Loading market data + precomputing indicators...")
    df_market = load_market()
    per_tic = precompute_signals(df_market)
    print(f"  {len(per_tic)} tickers ready; test window {TEST_START}..{TEST_END}")

    print("\n[2/4] Loading sentiment sources...")
    df_av = load_av_sentiment()
    df_social = load_social_sentiment()
    df_combined = combine_sentiment(df_av, df_social)
    print(f"  AV:       {len(df_av):>6} rows, {df_av['ticker'].nunique() if not df_av.empty else 0} tickers")
    print(f"  Social:   {len(df_social):>6} rows, {df_social['ticker'].nunique() if not df_social.empty else 0} tickers")
    print(f"  Combined: {len(df_combined):>6} rows")

    sources = {"av_only": df_av, "social_only": df_social, "combined": df_combined}
    lookbacks = [1, 3, 7]
    modes = ["gate_nonneg", "gate_pos", "veto_neg", "tilt_size"]

    results = []
    equity_curves = {}

    print("\n[3/4] Running baseline (no sentiment)...")
    eq, extra = run_backtest(per_tic, {}, mode="none")
    m = compute_metrics(eq)
    m.update(extra)
    m.update({"source": "none", "lookback_days": 0, "mode": "none", "config": "baseline"})
    results.append(m)
    equity_curves["baseline"] = eq.set_index("date")["value"]
    print(f"  baseline: return={m['total_return_pct']}%  sharpe={m['sharpe']}  trades={m['trade_count']}")

    n_cfg = len(sources) * len(lookbacks) * len(modes)
    print(f"\n[4/4] Sweeping source x lookback x mode = {n_cfg} configs...")
    for src_name, df_src in sources.items():
        for lb in lookbacks:
            df_lb = apply_lookback(df_src, lb)
            sent_lookup = build_sent_lookup(df_lb)
            for mode in modes:
                cfg = f"{src_name}__lb{lb}__{mode}"
                eq, extra = run_backtest(per_tic, sent_lookup, mode=mode)
                m = compute_metrics(eq)
                m.update(extra)
                m.update({"source": src_name, "lookback_days": lb, "mode": mode, "config": cfg})
                results.append(m)
                equity_curves[cfg] = eq.set_index("date")["value"]
                print(f"  {cfg:<42}  ret={m['total_return_pct']:>6.2f}%  "
                      f"sharpe={m['sharpe']:>6.3f}  trades={m['trade_count']:>3}  "
                      f"cov={m['sentiment_coverage_pct']:>5.1f}%")

    # Save
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "ablation_results.csv")
    df_res.to_csv(out_csv, index=False)

    df_eq = pd.concat(equity_curves, axis=1)
    df_eq.columns = list(equity_curves.keys())
    eq_csv = os.path.join(OUT_DIR, "ablation_equity.csv")
    df_eq.to_csv(eq_csv)

    # Pivots
    print("\n" + "=" * 70)
    print("  SHARPE — pivot (rows: mode, cols: source) averaged across lookbacks")
    print("=" * 70)
    pivot_sharpe = df_res[df_res["source"] != "none"].pivot_table(
        index="mode", columns="source", values="sharpe", aggfunc="mean"
    )
    baseline_sharpe = df_res[df_res["source"] == "none"]["sharpe"].iloc[0]
    print(pivot_sharpe.round(3).to_string())
    print(f"  baseline (no sentiment): sharpe = {baseline_sharpe:.3f}")

    print("\n" + "=" * 70)
    print("  TOTAL RETURN % — pivot (rows: mode, cols: source)")
    print("=" * 70)
    pivot_ret = df_res[df_res["source"] != "none"].pivot_table(
        index="mode", columns="source", values="total_return_pct", aggfunc="mean"
    )
    baseline_ret = df_res[df_res["source"] == "none"]["total_return_pct"].iloc[0]
    print(pivot_ret.round(2).to_string())
    print(f"  baseline (no sentiment): total_return = {baseline_ret:.2f}%")

    # Top/bottom
    print("\n" + "=" * 70)
    print("  TOP 5 CONFIGS by Sharpe")
    print("=" * 70)
    top = df_res.sort_values("sharpe", ascending=False).head(5)
    print(top[["config", "total_return_pct", "sharpe", "max_drawdown_pct",
               "trade_count", "sentiment_coverage_pct"]].to_string(index=False))

    print("\n  BOTTOM 5 CONFIGS by Sharpe")
    bot = df_res.sort_values("sharpe", ascending=True).head(5)
    print(bot[["config", "total_return_pct", "sharpe", "max_drawdown_pct",
               "trade_count", "sentiment_coverage_pct"]].to_string(index=False))

    # Improvement vs baseline
    df_diff = df_res[df_res["source"] != "none"].copy()
    df_diff["sharpe_vs_baseline"] = df_diff["sharpe"] - baseline_sharpe
    df_diff["return_vs_baseline_pct"] = df_diff["total_return_pct"] - baseline_ret
    n_better_sharpe = (df_diff["sharpe_vs_baseline"] > 0).sum()
    n_better_ret = (df_diff["return_vs_baseline_pct"] > 0).sum()
    n_total = len(df_diff)
    print("\n" + "=" * 70)
    print("  VERDICT vs baseline (no sentiment)")
    print("=" * 70)
    print(f"  configs beating baseline on Sharpe: {n_better_sharpe}/{n_total}")
    print(f"  configs beating baseline on Total Return: {n_better_ret}/{n_total}")
    print(f"  mean Δ Sharpe: {df_diff['sharpe_vs_baseline'].mean():+.4f}")
    print(f"  mean Δ Return: {df_diff['return_vs_baseline_pct'].mean():+.3f}%")

    print(f"\n  Results saved:")
    print(f"    {out_csv}")
    print(f"    {eq_csv}")
    print(f"\n  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
