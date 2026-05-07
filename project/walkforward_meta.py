"""
Walk-forward cross-validation for the sentiment-only MetaModel variant.

Procedure (per quarterly fold):
  - Train rows: every (date, tic) row with date < fold_start
  - Test rows : every (date, tic) row with fold_start <= date < fold_end
  - Refit GradientBoostingClassifier on train rows for each variant
  - Score test rows, run the same monthly-rebalance + ATR-stop backtest
    on that quarter only (capital reset to $1M each quarter for fair fold-Sharpe
    comparison; portfolio-of-quarters Sharpe also computed)
  - Compare three variants per fold:
       * baseline (12 features)
       * deployed (13 features = baseline + sent_x_near_earn)
       * sentiment-only (2 features: sent_3d + near_earnings)

We then check: across folds, is sentiment-only's Sharpe consistently above
the deployed variant's, or was Ablation C's 1.454 a single-fold artifact?
"""
from __future__ import annotations

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (  # type: ignore
    _build_meta_features, INITIAL_AMOUNT, DAILY_CASH_YIELD,
    TRAILING_ATR_MULT_TREND, OUTPUT_DIR,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "meta_walkforward.csv")

BASE_FEATURES = [
    "mom_5d", "mom_21d", "mom_63d", "rsi14", "vol_60d",
    "net_60d_analyst", "n_analyst_events_60d",
    "days_since_up", "days_since_down", "target_upside",
    "near_earnings", "sent_3d",
]
DEPLOYED_FEATURES = BASE_FEATURES + ["sent_x_near_earn"]
SENTIMENT_ONLY = ["sent_3d", "near_earnings"]

VARIANTS = {
    "baseline (12)":      BASE_FEATURES,
    "deployed (13+inter)": DEPLOYED_FEATURES,
    "sentiment-only (2)": SENTIMENT_ONLY,
}


def add_interaction(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    feats["sent_x_near_earn"] = feats["sent_3d"] * feats["near_earnings"]
    return feats


def quarterly_folds(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Generate non-overlapping quarterly test folds. Each fold has at least
    250 training-day buffer and a clean test window of one quarter."""
    folds = []
    d = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    while d < end_ts:
        nxt = d + pd.DateOffset(months=3)
        if nxt > end_ts:
            nxt = end_ts
        folds.append((d, nxt, f"{d:%Y-Q}{(d.month-1)//3+1}"))
        d = nxt
    return folds


def simulate_fold(test_with_probs: pd.DataFrame, df_market: pd.DataFrame,
                  fold_start: pd.Timestamp, fold_end: pd.Timestamp,
                  tickers: list, top_n: int = 10, rebalance_days: int = 21):
    """Run the same monthly rebalance + ATR stop loop, but on a single fold.
    Capital is reset to $1M at the start of every fold."""
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
        tdf = tdf[(tdf["date"] >= fold_start) & (tdf["date"] < fold_end)].reset_index(drop=True)
        if tdf.empty:
            continue
        per_ticker[tic] = tdf
    if not per_ticker:
        return None

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

    return pd.DataFrame(history)


def fold_sharpe(df_acct: pd.DataFrame) -> dict:
    if df_acct is None or len(df_acct) < 5:
        return {"sharpe": np.nan, "total_return_pct": np.nan, "max_dd_pct": np.nan,
                "n_days": 0, "final_value": np.nan}
    vals = df_acct["equity"].astype(float).values
    daily = np.diff(vals) / vals[:-1]
    if len(daily) == 0 or np.std(daily) == 0:
        return {"sharpe": 0.0, "total_return_pct": float((vals[-1]/vals[0] - 1) * 100),
                "max_dd_pct": 0.0, "n_days": len(vals), "final_value": float(vals[-1])}
    ann_ret = np.mean(daily) * 252
    ann_vol = np.std(daily) * np.sqrt(252)
    sharpe = (ann_ret - 0.04) / ann_vol if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(np.max((peak - vals) / peak))
    return {"sharpe": round(float(sharpe), 3),
            "total_return_pct": round(float((vals[-1]/vals[0] - 1) * 100), 2),
            "max_dd_pct": round(mdd * 100, 2),
            "n_days": len(vals),
            "final_value": round(float(vals[-1]), 2)}


def main():
    print("[1/4] Loading data...")
    df_market = pd.read_csv(os.path.join(OUTPUT_DIR, "market_data_expanded.csv"))
    df_market["date"] = pd.to_datetime(df_market["date"])
    tickers = sorted(df_market["tic"].unique().tolist())

    sent_csv = os.path.join(OUTPUT_DIR, "sentiment.csv")
    df_sent = pd.read_csv(sent_csv) if os.path.exists(sent_csv) else None
    if df_sent is not None and not df_sent.empty:
        df_sent["date"] = pd.to_datetime(df_sent["date"])

    analyst_df = pd.read_csv(os.path.join(OUTPUT_DIR, "analyst_actions.csv"),
                              parse_dates=["GradeDate"])
    ed_df = pd.read_csv(os.path.join(OUTPUT_DIR, "earnings_dates.csv"),
                        parse_dates=["earnings_date"])
    earnings_by_tic = {tic: sorted(sub["earnings_date"].dt.normalize().tolist())
                       for tic, sub in ed_df.groupby("tic")}

    print("[2/4] Building feature matrix...")
    feats = _build_meta_features(df_market, tickers, analyst_df, earnings_by_tic, df_sent)
    feats = add_interaction(feats)
    print(f"    matrix: {len(feats):,} rows × {feats.shape[1]} cols")

    # Quarterly folds spanning the full test window. We require at least 1 year
    # of training data before the first fold, so folds start 2024-Q1 (since the
    # feature matrix begins ~2022-Q3 after 252-day rolling indicators warm up).
    folds = []
    d = pd.Timestamp("2024-01-01")
    while d < pd.Timestamp("2026-04-01"):
        nxt = d + pd.DateOffset(months=3)
        folds.append((d, nxt, f"{d.year}-Q{(d.month - 1)//3 + 1}"))
        d = nxt

    print(f"[3/4] Walk-forward folds: {len(folds)}")
    for f in folds:
        print(f"    {f[2]:>8s}  {f[0].date()} → {f[1].date()}")

    rows = []
    print("\n[4/4] Running fold × variant matrix...")
    for fold_start, fold_end, fold_name in folds:
        train_mask = (feats["date"] < fold_start) & feats["label"].notna()
        train = feats[train_mask].copy()
        if len(train) < 1000:
            print(f"  skip {fold_name}: only {len(train)} train rows")
            continue
        for vname, fcols in VARIANTS.items():
            train_v = train.dropna(subset=fcols)
            if len(train_v) < 1000:
                continue
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            clf.fit(train_v[fcols], train_v["label"].astype(int))
            test_mask = (feats["date"] >= fold_start) & (feats["date"] < fold_end)
            test = feats[test_mask].dropna(subset=fcols).copy()
            if test.empty:
                continue
            test["prob"] = clf.predict_proba(test[fcols])[:, 1]
            df_acct = simulate_fold(test, df_market, fold_start, fold_end, tickers)
            m = fold_sharpe(df_acct)
            m["fold"] = fold_name
            m["variant"] = vname
            m["n_train"] = len(train_v)
            m["n_features"] = len(fcols)
            rows.append(m)
            print(f"  {fold_name}  {vname:<22s}  Sharpe {m['sharpe']:+.3f}  "
                  f"Ret {m['total_return_pct']:+6.2f}%  MDD {m['max_dd_pct']:5.2f}%")

    df_out = pd.DataFrame(rows)
    df_out = df_out[["fold", "variant", "n_features", "n_train",
                      "n_days", "total_return_pct", "sharpe", "max_dd_pct",
                      "final_value"]]
    df_out.to_csv(OUT_CSV, index=False)

    print("\n=== Per-variant aggregate (mean of fold Sharpes) ===")
    agg = (df_out.groupby("variant")
           .agg(mean_sharpe=("sharpe", "mean"),
                median_sharpe=("sharpe", "median"),
                std_sharpe=("sharpe", "std"),
                mean_return=("total_return_pct", "mean"),
                folds=("fold", "count"))
           .round(3).sort_values("mean_sharpe", ascending=False))
    print(agg.to_string())

    print("\n=== Side-by-side fold table ===")
    pivot = df_out.pivot_table(index="fold", columns="variant",
                                values="sharpe", aggfunc="first")
    print(pivot.round(3).to_string())

    print(f"\nResults written to {OUT_CSV}")


if __name__ == "__main__":
    main()
