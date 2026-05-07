"""
Window-size ablation: SMA/RSI lookback sweep on RSI + SMA_RSI rules.
Answers: does making the rule more responsive help returns in 2025?
"""
import os, sys, warnings, time, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")
OUT_DIR = os.path.join(BASE_DIR, "results", "ablation")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_START = "2025-01-01"
INITIAL_AMOUNT = 1_000_000
TRAILING_STOP_PCT = 0.08
TXN_COST = 0.001


def load_market():
    df = pd.read_csv(os.path.join(DATA_DIR, "market_data_expanded.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_signals(df_market, sma_fast, sma_slow, rsi_n, mode):
    per_tic = {}
    for tic, g in df_market.groupby("tic"):
        g = g.sort_values("date").copy()
        min_len = max(sma_slow, rsi_n) + 50
        if len(g) < min_len:
            continue
        g["SMA_F"] = g["close"].rolling(sma_fast).mean()
        g["SMA_S"] = g["close"].rolling(sma_slow).mean()
        delta = g["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_n).mean()
        rs = gain / loss.replace(0, np.nan)
        g["RSI"] = 100 - (100 / (1 + rs))
        g.dropna(inplace=True)
        g = g[g["date"] >= pd.Timestamp(TEST_START)].copy()
        if g.empty:
            continue

        if mode == "RSI":
            g["Buy"] = g["RSI"] < 35
            g["Sell"] = g["RSI"] > 75
        elif mode == "SMA_RSI":
            up = g["SMA_F"] > g["SMA_S"]
            g["Buy"] = up & (g["RSI"] < 40) & (g["RSI"].shift(1) >= 40)
            g["Sell"] = (~up) & (g["SMA_F"].shift(1) > g["SMA_S"].shift(1))
        per_tic[tic] = g[["date", "close", "Buy", "Sell"]].reset_index(drop=True)
    return per_tic


def backtest(per_tic):
    tickers = list(per_tic.keys())
    if not tickers:
        return None, None
    alloc = INITIAL_AMOUNT / len(tickers)
    pos = {t: {"shares": 0, "cash": alloc, "peak": 0.0} for t in tickers}
    dates = sorted(set(itertools.chain.from_iterable(df["date"].tolist() for df in per_tic.values())))
    eq, trades = [], 0
    for dt in dates:
        tot = 0.0
        for t in tickers:
            row = per_tic[t][per_tic[t]["date"] == dt]
            p = pos[t]
            if row.empty:
                tot += p["cash"]; continue
            price = float(row["close"].iloc[0])
            buy = bool(row["Buy"].iloc[0]); sell = bool(row["Sell"].iloc[0])
            if p["shares"] > 0:
                p["peak"] = max(p["peak"], price)
                if price < p["peak"] * (1 - TRAILING_STOP_PCT):
                    sell = True
            if buy and p["shares"] == 0 and p["cash"] > price:
                sh = int(p["cash"] // (price * (1 + TXN_COST)))
                if sh > 0:
                    p["cash"] -= sh * price * (1 + TXN_COST)
                    p["shares"] = sh; p["peak"] = price; trades += 1
            elif sell and p["shares"] > 0:
                p["cash"] += p["shares"] * price * (1 - TXN_COST)
                p["shares"] = 0; p["peak"] = 0.0; trades += 1
            tot += p["cash"] + p["shares"] * price
        eq.append(tot)
    vals = np.array(eq)
    rets = pd.Series(vals).pct_change().dropna()
    total_ret = (vals[-1]/vals[0] - 1) * 100
    ann_ret = ((vals[-1]/vals[0]) ** (252/max(1,len(vals))) - 1) * 100
    sharpe = (rets.mean()*252) / (rets.std()*np.sqrt(252) + 1e-12)
    mdd = ((vals - np.maximum.accumulate(vals)) / np.maximum.accumulate(vals)).min() * 100
    return {"total_return_pct": round(total_ret,2), "annual_return_pct": round(ann_ret,2),
            "sharpe": round(sharpe,3), "max_dd_pct": round(mdd,2), "trades": trades}, vals


def main():
    t0 = time.time()
    df = load_market()
    print(f"Universe: {df['tic'].nunique()} tickers  |  test window: {TEST_START} → end\n")

    # SMA window pairs (fast, slow)
    sma_pairs = [(10,30), (20,50), (20,100), (50,200), (100,200)]
    rsi_lens  = [5, 7, 9, 14, 21]

    rows = []
    print("=== RSI-only mode: sweep RSI length (SMAs unused) ===")
    print(f"{'RSI(n)':>8} {'return':>10} {'ann':>8} {'sharpe':>8} {'mdd':>8} {'trades':>7}")
    for n in rsi_lens:
        per_tic = build_signals(df, 20, 50, n, "RSI")
        m, _ = backtest(per_tic)
        rows.append({"mode":"RSI","sma_fast":None,"sma_slow":None,"rsi_n":n, **m})
        print(f"  RSI({n:>2}) {m['total_return_pct']:>9.2f}% {m['annual_return_pct']:>7.2f}%"
              f" {m['sharpe']:>8.3f} {m['max_dd_pct']:>7.2f}% {m['trades']:>7}")

    print("\n=== SMA_RSI mode: sweep SMA(fast, slow) with RSI(14) ===")
    print(f"{'(f,s)':>10} {'return':>10} {'ann':>8} {'sharpe':>8} {'mdd':>8} {'trades':>7}")
    for f, s in sma_pairs:
        per_tic = build_signals(df, f, s, 14, "SMA_RSI")
        m, _ = backtest(per_tic)
        rows.append({"mode":"SMA_RSI","sma_fast":f,"sma_slow":s,"rsi_n":14, **m})
        print(f"  ({f:>3},{s:>3}) {m['total_return_pct']:>9.2f}% {m['annual_return_pct']:>7.2f}%"
              f" {m['sharpe']:>8.3f} {m['max_dd_pct']:>7.2f}% {m['trades']:>7}")

    print("\n=== Combined: best-looking configs ===")
    combos = [(10,30,7), (20,50,7), (20,100,9), (50,200,14), (100,200,21)]
    print(f"{'(f,s,rsi)':>14} {'return':>10} {'ann':>8} {'sharpe':>8} {'mdd':>8} {'trades':>7}")
    for f, s, n in combos:
        per_tic = build_signals(df, f, s, n, "SMA_RSI")
        m, _ = backtest(per_tic)
        rows.append({"mode":"SMA_RSI","sma_fast":f,"sma_slow":s,"rsi_n":n, **m})
        print(f"  ({f:>3},{s:>3},{n:>2}) {m['total_return_pct']:>9.2f}% {m['annual_return_pct']:>7.2f}%"
              f" {m['sharpe']:>8.3f} {m['max_dd_pct']:>7.2f}% {m['trades']:>7}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "window_ablation.csv"), index=False)
    print(f"\nSaved → results/ablation/window_ablation.csv  |  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
