"""
Sentiment-feature ablation study for the MetaModel strategy.

For each variant, we:
  1. Rebuild the feature matrix (reusing pipeline._build_meta_features)
  2. Optionally add / drop / replace sentiment-related columns
  3. Refit the GradientBoostingClassifier on training rows (date < TEST_START)
  4. Run the same rebalance backtest used by run_meta_model_backtest
  5. Compute Sharpe / return / MDD / final value on the held-out test window
  6. Log feature importances

Output: results/ablation/meta_sentiment_ablation.csv
"""
from __future__ import annotations

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Reuse the helpers from the main pipeline so the ablation matches the
# production code path exactly (no drift in feature generation).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (  # type: ignore
    _build_meta_features, prepare_sentiment_data, EXPANDED_TICKERS,
    INITIAL_AMOUNT, DAILY_CASH_YIELD, TRAILING_ATR_MULT_TREND,
    TEST_START, TEST_END, OUTPUT_DIR,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "meta_sentiment_ablation.csv")

BASE_FEATURES = [
    "mom_5d", "mom_21d", "mom_63d", "rsi14", "vol_60d",
    "net_60d_analyst", "n_analyst_events_60d",
    "days_since_up", "days_since_down", "target_upside",
    "near_earnings", "sent_3d",
]


def add_derived_sentiment_columns(feats: pd.DataFrame) -> pd.DataFrame:
    """Append sentiment-derivative features the baseline doesn't have."""
    feats = feats.copy()
    feats["sent_x_near_earn"] = feats["sent_3d"] * feats["near_earnings"]
    feats["sent_change_3d"] = feats.groupby("tic")["sent_3d"].diff(3).fillna(0)
    # 30-day sentiment using a longer window via cumulative-mean-style proxy:
    # rolling mean of sent_3d itself over 30 trading days as a slow-moving baseline.
    feats["sent_30d"] = (
        feats.sort_values(["tic", "date"]).groupby("tic")["sent_3d"]
        .transform(lambda s: s.rolling(30, min_periods=5).mean().fillna(0))
    )
    feats["sent_3d_minus_30d"] = feats["sent_3d"] - feats["sent_30d"]
    return feats


def simulate(test_with_probs: pd.DataFrame, df_market: pd.DataFrame,
             tickers: list, top_n: int = 10, rebalance_days: int = 21):
    """Replay the same monthly-rebalance + ATR-stop loop as run_meta_model_backtest."""
    df = df_market.copy()
    df["date"] = pd.to_datetime(df["date"])
    test_start_dt = pd.Timestamp(TEST_START)

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
        tdf = tdf[tdf["date"] >= test_start_dt].reset_index(drop=True)
        if tdf.empty:
            continue
        per_ticker[tic] = tdf

    probs_by_date = {dt: g.set_index("tic")["prob"].to_dict()
                     for dt, g in test_with_probs.groupby("date")}

    all_dates = sorted(set().union(*(t["date"].tolist() for t in per_ticker.values())))
    cash = INITIAL_AMOUNT
    holdings = {}
    history = []
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

        # ATR stops
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

        days_since_rebal += 1
        if days_since_rebal >= rebalance_days and prices:
            days_since_rebal = 0
            today_probs = probs_by_date.get(dt, {})
            scored = [(tic, today_probs[tic]) for tic in prices if tic in today_probs]
            scored.sort(key=lambda x: -x[1])
            target_list = [tic for tic, p in scored[:top_n] if p > 0.5]
            target_set = set(target_list)
            for tic in sorted(holdings.keys()):
                if tic not in target_set and tic in prices:
                    h = holdings[tic]
                    cash += h["shares"] * prices[tic] * 0.999
                    del holdings[tic]
            total_equity = cash + sum(
                h["shares"] * prices.get(t, h["entry_price"]) for t, h in holdings.items()
            )
            target_alloc = total_equity / max(len(target_list), 1)
            for tic in target_list:
                if tic not in prices:
                    continue
                price = prices[tic]
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

        equity = cash + sum(h["shares"] * prices.get(t, 0) for t, h in holdings.items())
        history.append({"date": dt, "equity": equity})

    df_acct = pd.DataFrame(history)
    return df_acct


def metrics(df_acct: pd.DataFrame) -> dict:
    vals = df_acct["equity"].astype(float).values
    dates = pd.to_datetime(df_acct["date"])
    total_ret = (vals[-1] - vals[0]) / vals[0]
    n_days = (dates.iloc[-1] - dates.iloc[0]).days
    ann_ret = (1 + total_ret) ** (365.0 / max(n_days, 1)) - 1
    daily = np.diff(vals) / vals[:-1]
    ann_vol = np.std(daily) * np.sqrt(252)
    sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(np.max((peak - vals) / peak)) if len(vals) else 0.0
    return {
        "total_return_pct": round(total_ret * 100, 2),
        "annual_return_pct": round(ann_ret * 100, 2),
        "annual_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "final_value": round(float(vals[-1]), 2),
    }


def fit_and_simulate(feats_full: pd.DataFrame, feature_cols: list,
                     df_market: pd.DataFrame, tickers: list, label: str):
    """Train GBM on the listed features, run backtest, return metrics + importances."""
    test_start = pd.Timestamp(TEST_START)
    train = feats_full[(feats_full["date"] < test_start)
                       & feats_full["label"].notna()].dropna(subset=feature_cols).copy()
    if len(train) < 1000:
        return None
    X_train = train[feature_cols]
    y_train = train["label"].astype(int)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    test = feats_full[feats_full["date"] >= test_start].dropna(subset=feature_cols).copy()
    test["prob"] = clf.predict_proba(test[feature_cols])[:, 1]
    df_acct = simulate(test, df_market, tickers)
    m = metrics(df_acct)
    m["variant"] = label
    m["train_acc"] = round(train_acc, 3)
    m["n_features"] = len(feature_cols)
    importances = dict(zip(feature_cols, clf.feature_importances_.round(4).tolist()))
    return m, importances


def main():
    print("[1/3] Loading market + sentiment + analyst + earnings...")
    expanded_cache = os.path.join(OUTPUT_DIR, "market_data_expanded.csv")
    df_market = pd.read_csv(expanded_cache)
    df_market["date"] = pd.to_datetime(df_market["date"])
    tickers = sorted(df_market["tic"].unique().tolist())

    sentiment_csv = os.path.join(OUTPUT_DIR, "sentiment.csv")
    df_sentiment = pd.read_csv(sentiment_csv) if os.path.exists(sentiment_csv) else None
    if df_sentiment is not None and not df_sentiment.empty:
        df_sentiment["date"] = pd.to_datetime(df_sentiment["date"])

    analyst_csv = os.path.join(OUTPUT_DIR, "analyst_actions.csv")
    analyst_df = pd.read_csv(analyst_csv, parse_dates=["GradeDate"])
    earnings_csv = os.path.join(OUTPUT_DIR, "earnings_dates.csv")
    ed_df = pd.read_csv(earnings_csv, parse_dates=["earnings_date"])
    earnings_by_tic = {tic: sorted(sub["earnings_date"].dt.normalize().tolist())
                       for tic, sub in ed_df.groupby("tic")}

    print("[2/3] Building feature matrix...")
    feats = _build_meta_features(df_market, tickers, analyst_df, earnings_by_tic, df_sentiment)
    feats = add_derived_sentiment_columns(feats)
    print(f"    matrix: {len(feats):,} rows × {feats.shape[1]} cols")

    print("[3/3] Running variants...")
    variants = [
        ("baseline (12 features)",                BASE_FEATURES),
        ("drop sent_3d",                          [c for c in BASE_FEATURES if c != "sent_3d"]),
        ("drop near_earnings",                    [c for c in BASE_FEATURES if c != "near_earnings"]),
        ("drop sent_3d + near_earnings",          [c for c in BASE_FEATURES if c not in {"sent_3d", "near_earnings"}]),
        ("baseline + sent_x_near_earn",           BASE_FEATURES + ["sent_x_near_earn"]),
        ("baseline + sent_change_3d",             BASE_FEATURES + ["sent_change_3d"]),
        ("baseline + sent_30d + sent_3d_minus_30d", BASE_FEATURES + ["sent_30d", "sent_3d_minus_30d"]),
        ("price-only (no sentiment, no analyst)", ["mom_5d", "mom_21d", "mom_63d", "rsi14", "vol_60d"]),
        ("analyst-only (no price, no sentiment)", ["net_60d_analyst", "n_analyst_events_60d",
                                                    "days_since_up", "days_since_down", "target_upside"]),
        ("sentiment-only",                         ["sent_3d", "near_earnings"]),
    ]

    rows = []
    importances_log = []
    for name, cols in variants:
        print(f"  - {name} (n={len(cols)})...", end=" ", flush=True)
        result = fit_and_simulate(feats, cols, df_market, tickers, name)
        if result is None:
            print("FAILED")
            continue
        m, imps = result
        print(f"Sharpe {m['sharpe']:+.3f}  Ret {m['total_return_pct']:+.2f}%  MDD {m['max_drawdown_pct']:.2f}%")
        rows.append(m)
        for feat, imp in imps.items():
            importances_log.append({"variant": name, "feature": feat, "importance": imp})

    df_results = pd.DataFrame(rows)
    df_results.to_csv(OUT_CSV, index=False)
    df_importances = pd.DataFrame(importances_log)
    df_importances.to_csv(os.path.join(RESULTS_DIR, "meta_sentiment_importances.csv"), index=False)

    print("\n=== Variant comparison ===")
    print(df_results[["variant", "n_features", "train_acc", "total_return_pct", "sharpe",
                      "max_drawdown_pct", "annual_vol_pct"]].to_string(index=False))

    print("\n=== Baseline feature importances (top 10) ===")
    base_imps = (df_importances[df_importances["variant"] == "baseline (12 features)"]
                  .sort_values("importance", ascending=False).head(10))
    print(base_imps.to_string(index=False))

    print(f"\nResults written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
