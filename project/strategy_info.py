"""
Per-strategy metadata used by the dashboard's Strategy Detail page.
- description: paragraph explaining how the strategy works
- signal_type: one-line classifier for the leaderboard
- data_sources: list of inputs the strategy consumes
- parameters: key tunable constants
- architecture: graphviz DOT source describing data flow
"""
from __future__ import annotations


def _dot(nodes: list[str], edges: list[tuple[str, str]], label: str) -> str:
    """Compose a horizontal DOT diagram sized so it fits a Streamlit container
    column without forcing labels to shrink:
    - rankdir=LR (horizontal data flow)
    - bgcolor=transparent so the chart blends with the dashboard theme
    - size="11,4.5" caps the bounding box at 11×4.5 in (~1056×432 px @ 96dpi)
      which is close to a normal Streamlit column width — graphviz scales the
      layout down if needed but text stays readable
    - tight ranksep/nodesep so multi-stage flows pack horizontally instead of
      spreading and triggering Streamlit's container scale-down
    """
    body = []
    body.append('digraph G {')
    body.append('  rankdir=LR;')
    body.append('  bgcolor="transparent";')
    body.append('  size="11,4.5";')
    body.append('  ratio=compress;')
    body.append('  pad="0.15";')
    body.append('  ranksep="0.35"; nodesep="0.18";')
    # Node text inside boxes stays dark (boxes have light pastel fills);
    # graph label + edge labels + arrow color are theme-neutral light gray
    # so they're readable on Streamlit's default dark background.
    body.append('  node [shape=box, style="rounded,filled",')
    body.append('        fontname="Helvetica", fontsize=10,')
    body.append('        margin="0.15,0.08",')
    body.append('        color="#3a3a3a", penwidth=1.0,')
    body.append('        height=0.5,')
    body.append('        fontcolor="#1a1a1a"];')
    body.append('  edge [fontname="Helvetica", fontsize=9,')
    body.append('        color="#9a9a9a", fontcolor="#dddddd",')
    body.append('        penwidth=1.2, arrowsize=0.7];')
    for n in nodes:
        body.append(f"  {n}")
    for a, b in edges:
        body.append(f"  {a} -> {b}")
    body.append(f'  label="{label}";')
    body.append('  labelloc="t";')
    body.append('  fontname="Helvetica"; fontsize=12;')
    body.append('  fontcolor="#dddddd";')
    body.append('}')
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Architecture DOT per strategy
# ---------------------------------------------------------------------------
ARCH = {
    "PPO": _dot(
        ['ohlcv [label="Yahoo OHLCV\\n+ MACD/RSI/CCI/BB", fillcolor="#cfe8ff"];',
         'sent [label="Alpha Vantage\\nNews Sentiment", fillcolor="#fad2cf"];',
         'env [label="StockTradingEnv\\n(FinRL)", fillcolor="#ffe9b3"];',
         'ppo [label="PPO Policy\\n(Stable-Baselines3)", fillcolor="#d6c5e8"];',
         'pos [label="Long-only positions\\n|action| ≤ 100 sh", fillcolor="#d4f0d2"];'],
        [('ohlcv', 'env'), ('sent', 'env'), ('env', 'ppo'), ('ppo', 'pos')],
        "PPO — on-policy DRL with clipped surrogate objective",
    ),
    "A2C": _dot(
        ['ohlcv [label="OHLCV + indicators", fillcolor="#cfe8ff"];',
         'sent [label="AV Sentiment", fillcolor="#fad2cf"];',
         'env [label="StockTradingEnv", fillcolor="#ffe9b3"];',
         'a2c [label="A2C Policy\\n(synchronous A3C)", fillcolor="#d6c5e8"];',
         'pos [label="Discrete actions\\n|a| ≤ 100", fillcolor="#d4f0d2"];'],
        [('ohlcv', 'env'), ('sent', 'env'), ('env', 'a2c'), ('a2c', 'pos')],
        "A2C — synchronous actor-critic",
    ),
    "DDPG": _dot(
        ['ohlcv [label="OHLCV + indicators", fillcolor="#cfe8ff"];',
         'sent [label="AV Sentiment", fillcolor="#fad2cf"];',
         'env [label="StockTradingEnv", fillcolor="#ffe9b3"];',
         'ddpg [label="DDPG\\n(deterministic + critic)", fillcolor="#d6c5e8"];',
         'pos [label="Continuous actions", fillcolor="#d4f0d2"];'],
        [('ohlcv', 'env'), ('sent', 'env'), ('env', 'ddpg'), ('ddpg', 'pos')],
        "DDPG — off-policy actor-critic",
    ),
    "PPO_GATED": _dot(
        ['ppo [label="PPO base action", fillcolor="#d6c5e8"];',
         'lg [label="LangGraph debate\\n(bull/bear/executor)", fillcolor="#ffd6a5"];',
         'gate [label="Gate layer\\nveto / dampen / tilt", fillcolor="#e8c5c5"];',
         'pos [label="Final position", fillcolor="#d4f0d2"];'],
        [('ppo', 'gate'), ('lg', 'gate'), ('gate', 'pos')],
        "PPO_GATED — PPO with LangGraph signal override",
    ),
    "A2C_GATED": _dot(
        ['a2c [label="A2C base action", fillcolor="#d6c5e8"];',
         'lg [label="LangGraph debate", fillcolor="#ffd6a5"];',
         'gate [label="Gate layer", fillcolor="#e8c5c5"];',
         'pos [label="Final position", fillcolor="#d4f0d2"];'],
        [('a2c', 'gate'), ('lg', 'gate'), ('gate', 'pos')],
        "A2C_GATED — A2C with LangGraph signal override",
    ),
    "DDPG_GATED": _dot(
        ['ddpg [label="DDPG base action", fillcolor="#d6c5e8"];',
         'lg [label="LangGraph debate", fillcolor="#ffd6a5"];',
         'gate [label="Gate layer", fillcolor="#e8c5c5"];',
         'pos [label="Final position", fillcolor="#d4f0d2"];'],
        [('ddpg', 'gate'), ('lg', 'gate'), ('gate', 'pos')],
        "DDPG_GATED — DDPG with LangGraph signal override",
    ),
    "RuleBased (SMA)": _dot(
        ['ohlcv [label="OHLCV", fillcolor="#cfe8ff"];',
         'sma [label="SMA50 vs SMA200\\n(golden / death cross)", fillcolor="#ffe9b3"];',
         'sig [label="Buy on golden cross\\nSell on death cross", fillcolor="#d4f0d2"];',
         'stop [label="ATR(14) × 3.0\\ntrailing stop", fillcolor="#fad2cf"];',
         'pos [label="Position", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'sma'), ('sma', 'sig'), ('sig', 'stop'), ('stop', 'pos')],
        "SMA — pure trend-following crossover",
    ),
    "RuleBased (RSI)": _dot(
        ['ohlcv [label="OHLCV", fillcolor="#cfe8ff"];',
         'rsi [label="RSI(14)", fillcolor="#ffe9b3"];',
         'sig [label="Buy RSI<35\\nSell RSI>75", fillcolor="#d4f0d2"];',
         'mom [label="63d momentum filter\\n(skip falling knives)", fillcolor="#fad2cf"];',
         'stop [label="ATR(14) × 2.5\\ntrailing stop", fillcolor="#fad2cf"];',
         'pos [label="Position\\n+ profit ladder 15/30%", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'rsi'), ('rsi', 'sig'), ('sig', 'mom'), ('mom', 'stop'), ('stop', 'pos')],
        "RSI — mean-reversion dip-buy with momentum filter",
    ),
    "RuleBased (SMA_RSI)": _dot(
        ['ohlcv [label="OHLCV", fillcolor="#cfe8ff"];',
         'sma [label="SMA50 > SMA200\\n(uptrend confirmed)", fillcolor="#ffe9b3"];',
         'rsi [label="RSI<40 dip", fillcolor="#ffe9b3"];',
         'mom [label="63d momentum ≥ 0", fillcolor="#fad2cf"];',
         'sig [label="Buy = uptrend ∧ RSI dip ∧ momentum", fillcolor="#d4f0d2"];',
         'stop [label="ATR × 3.0 stop", fillcolor="#fad2cf"];',
         'pos [label="Position", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'sma'), ('ohlcv', 'rsi'), ('sma', 'sig'), ('rsi', 'sig'), ('mom', 'sig'),
         ('sig', 'stop'), ('stop', 'pos')],
        "SMA_RSI — trend-confirmed dip-buy",
    ),
    "RuleBased (SMA_RSI_Sentiment)": _dot(
        ['ohlcv [label="OHLCV", fillcolor="#cfe8ff"];',
         'sent [label="3-day MA sentiment\\n(combined: AV/RSS/Telegram/Apify)", fillcolor="#fad2cf"];',
         'sma [label="SMA50 > SMA200", fillcolor="#ffe9b3"];',
         'rsi [label="RSI<40 dip", fillcolor="#ffe9b3"];',
         'gate [label="sentiment > +0.1\\n(buy gate)", fillcolor="#ffd6a5"];',
         'sig [label="Buy if all gates pass", fillcolor="#d4f0d2"];',
         'pos [label="Position\\nATR×3 stop + ladder", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'sma'), ('ohlcv', 'rsi'), ('sent', 'gate'),
         ('sma', 'sig'), ('rsi', 'sig'), ('gate', 'sig'), ('sig', 'pos')],
        "SMA_RSI_Sentiment — trend + RSI dip + sentiment gate (everywhere)",
    ),
    "RuleBased (Dynamic)": _dot(
        ['ohlcv [label="OHLCV", fillcolor="#cfe8ff"];',
         'sent [label="Sentiment\\n(combined sources)", fillcolor="#fad2cf"];',
         'sig [label="SMA_RSI signal", fillcolor="#ffe9b3"];',
         'vol [label="60d realized vol\\n→ inverse-vol weight", fillcolor="#ffd6a5"];',
         'sw [label="Sentiment weight\\n×0.5 to ×2.0", fillcolor="#ffd6a5"];',
         'alloc [label="Allocation = vol × sent", fillcolor="#d4f0d2"];',
         'conv [label="RSI conviction sizing\\n+ partial sells", fillcolor="#d4f0d2"];',
         'pos [label="Position", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'sig'), ('sent', 'sw'), ('ohlcv', 'vol'),
         ('vol', 'alloc'), ('sw', 'alloc'), ('sig', 'conv'), ('alloc', 'conv'), ('conv', 'pos')],
        "Dynamic — sentiment- and vol-weighted SMA_RSI",
    ),
    "RuleBased (RegimeAdaptive)": _dot(
        ['ohlcv [label="OHLCV", fillcolor="#cfe8ff"];',
         'sma [label="SMA20 / SMA50 / SMA200", fillcolor="#ffe9b3"];',
         'vol [label="60d vol vs 252d median", fillcolor="#ffe9b3"];',
         'spy [label="SPY trend gate\\n(bear: SMA50 < SMA200)", fillcolor="#fad2cf"];',
         'reg [label="Regime → target weight\\n+1 / +0.5 / 0 / -1", fillcolor="#ffd6a5"];',
         'pos [label="Target-weight rebalance\\n(long + short)", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'sma'), ('ohlcv', 'vol'), ('sma', 'reg'), ('vol', 'reg'),
         ('spy', 'reg'), ('reg', 'pos')],
        "RegimeAdaptive — multi-MA regime with SPY-gated shorts",
    ),
    "RuleBased (SentimentMomentum)": _dot(
        ['sent [label="Per-ticker sentiment\\n(forward-filled)", fillcolor="#fad2cf"];',
         'short [label="5-day sentiment MA", fillcolor="#ffe9b3"];',
         'long [label="30-day sentiment MA", fillcolor="#ffe9b3"];',
         'surp [label="Surprise = short - long", fillcolor="#ffd6a5"];',
         'sma [label="SMA50 > SMA200\\nuptrend confirmation", fillcolor="#ffe9b3"];',
         'sig [label="Buy if surprise > +0.05\\n∧ uptrend", fillcolor="#d4f0d2"];',
         'pos [label="Position\\nATR×3 stop", fillcolor="#d6c5e8"];'],
        [('sent', 'short'), ('sent', 'long'), ('short', 'surp'), ('long', 'surp'),
         ('surp', 'sig'), ('sma', 'sig'), ('sig', 'pos')],
        "SentimentMomentum — buy on positive sentiment surprise",
    ),
    "RuleBased (CrossMomentum)": _dot(
        ['ohlcv [label="OHLCV (65 tickers)", fillcolor="#cfe8ff"];',
         'mom [label="63-day return\\nper ticker", fillcolor="#ffe9b3"];',
         'rank [label="Cross-sectional sort\\nby 63d return", fillcolor="#ffd6a5"];',
         'top [label="Top 10 (positive only)\\nequal-weight target", fillcolor="#d4f0d2"];',
         'rebal [label="Monthly rebalance\\n(21 trading days)", fillcolor="#fad2cf"];',
         'stop [label="ATR(14) × 3.0\\ntrailing stop", fillcolor="#fad2cf"];',
         'pos [label="Portfolio", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'mom'), ('mom', 'rank'), ('rank', 'top'),
         ('top', 'rebal'), ('rebal', 'stop'), ('stop', 'pos')],
        "CrossMomentum — top-10 by 63-day return, monthly rebal",
    ),
    "RuleBased (SentimentRank)": _dot(
        ['sent [label="Combined sentiment\\n(AV/RSS/Telegram/Apify)", fillcolor="#fad2cf"];',
         'surp [label="5d MA - 30d MA\\nper ticker", fillcolor="#ffe9b3"];',
         'z [label="Z-score across\\ntickers (cross-section)", fillcolor="#ffd6a5"];',
         'top [label="Top 10 by Z\\n(z > 0)", fillcolor="#d4f0d2"];',
         'rebal [label="Monthly rebalance", fillcolor="#fad2cf"];',
         'pos [label="Portfolio\\nATR×3 stop", fillcolor="#d6c5e8"];'],
        [('sent', 'surp'), ('surp', 'z'), ('z', 'top'),
         ('top', 'rebal'), ('rebal', 'pos')],
        "SentimentRank — cross-sectional sentiment Z-score",
    ),
    "RuleBased (AnalystRank)": _dot(
        ['yf [label="yfinance.upgrades_downgrades\\n(33,605 events cached)", fillcolor="#cfe8ff"];',
         'score [label="Per-event score:\\n+1 up / -1 down\\n+0.5 raise / -0.5 lower\\n±0.5 init bullish/bearish", fillcolor="#ffe9b3"];',
         'roll [label="Sum scores in last 60d\\nper ticker", fillcolor="#ffd6a5"];',
         'top [label="Top 10 by net score\\n(score > 0)", fillcolor="#d4f0d2"];',
         'rebal [label="Monthly rebalance", fillcolor="#fad2cf"];',
         'pos [label="Portfolio\\nATR×3 stop", fillcolor="#d6c5e8"];'],
        [('yf', 'score'), ('score', 'roll'), ('roll', 'top'),
         ('top', 'rebal'), ('rebal', 'pos')],
        "AnalystRank — Wall Street analyst flow (external)",
    ),
    "RuleBased (InsiderRank)": _dot(
        ['yf [label="yfinance.insider_transactions\\n(7,637 events, 2,864 discretionary)", fillcolor="#cfe8ff"];',
         'cls [label="Classify by Text:\\n+Value if Purchase\\n-Value if Sale\\n0 if Award/Grant/Tax", fillcolor="#ffe9b3"];',
         'roll [label="Sum signed $ in last 60d\\nper ticker", fillcolor="#ffd6a5"];',
         'top [label="Top 10 by net $\\n(least-negative ok)", fillcolor="#d4f0d2"];',
         'rebal [label="Monthly rebalance", fillcolor="#fad2cf"];',
         'pos [label="Portfolio\\nATR×3 stop", fillcolor="#d6c5e8"];'],
        [('yf', 'cls'), ('cls', 'roll'), ('roll', 'top'),
         ('top', 'rebal'), ('rebal', 'pos')],
        "InsiderRank — corporate insider $ flow (external)",
    ),
    "RuleBased (EarningsSentiment)": _dot(
        ['ohlcv [label="OHLCV", fillcolor="#cfe8ff"];',
         'earn [label="yfinance.earnings_dates\\n(1,588 dates cached)", fillcolor="#cfe8ff"];',
         'sent [label="Combined sentiment\\n(3d MA)", fillcolor="#fad2cf"];',
         'sma [label="SMA50 > SMA200", fillcolor="#ffe9b3"];',
         'rsi [label="RSI<40 dip", fillcolor="#ffe9b3"];',
         'win [label="±5 trading days\\nof earnings?", fillcolor="#ffd6a5"];',
         'gate [label="If in window:\\nrequire sent > +0.1\\nElse: pass through", fillcolor="#ffd6a5"];',
         'sig [label="Buy = SMA_RSI ∧ gate", fillcolor="#d4f0d2"];',
         'pos [label="Position\\nATR×3 stop + ladder", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'sma'), ('ohlcv', 'rsi'), ('earn', 'win'), ('sent', 'gate'), ('win', 'gate'),
         ('sma', 'sig'), ('rsi', 'sig'), ('gate', 'sig'), ('sig', 'pos')],
        "EarningsSentiment — sentiment gate active only near earnings",
    ),
    "RuleBased (MetaModel)": _dot(
        ['ohlcv [label="OHLCV (full history)", fillcolor="#cfe8ff"];',
         'analyst [label="analyst_actions.csv\\n(33,605 events)", fillcolor="#cfe8ff"];',
         'earn [label="earnings_dates.csv\\n(1,588 dates)", fillcolor="#cfe8ff"];',
         'sent [label="Combined sentiment", fillcolor="#fad2cf"];',
         'inter [label="Interaction term\\nsent_3d × near_earnings\\n(Ablation C: +0.146 Sharpe)", fillcolor="#ffd6a5"];',
         'feat [label="13 features per (date, tic):\\nmom_5/21/63d, RSI, vol_60d,\\nnet_60d_analyst, n_events,\\ndays_since_up/down,\\ntarget_upside, near_earnings,\\nsent_3d, sent_x_near_earn", fillcolor="#ffe9b3"];',
         'lab [label="Label = next_21d_ret\\n> universe median\\n(cross-sectional binary)", fillcolor="#ffd6a5"];',
         'gbm [label="GradientBoostingClassifier\\n200 trees, depth 3,\\ntrained 2022-03 → 2025-01\\n(41,864 rows)", fillcolor="#d6c5e8"];',
         'prob [label="P(above-median next-21d)\\nper (date, tic)", fillcolor="#ffd6a5"];',
         'top [label="Top 10 by P(>0.5)\\nmonthly rebalance", fillcolor="#d4f0d2"];',
         'pos [label="Portfolio\\nATR×3 stop", fillcolor="#d6c5e8"];'],
        [('ohlcv', 'feat'), ('analyst', 'feat'), ('earn', 'feat'), ('sent', 'feat'),
         ('sent', 'inter'), ('earn', 'inter'), ('inter', 'feat'),
         ('ohlcv', 'lab'), ('feat', 'gbm'), ('lab', 'gbm'), ('gbm', 'prob'),
         ('prob', 'top'), ('top', 'pos')],
        "MetaModel — supervised GBM with sent_3d × near_earnings interaction",
    ),
    "RuleBased (SentimentMeta)": _dot(
        ['sent [label="Combined sentiment\\n(FinBERT-scored, ffilled)", fillcolor="#fad2cf"];',
         'earn [label="earnings_dates.csv\\n(1,588 dates)", fillcolor="#cfe8ff"];',
         'sent3d [label="sent_3d\\n3-day rolling MA", fillcolor="#ffe9b3"];',
         'near [label="near_earnings\\n(±5 trading days flag)", fillcolor="#ffe9b3"];',
         'lab [label="Label = next_21d_ret\\n> universe median", fillcolor="#ffd6a5"];',
         'gbm [label="GradientBoostingClassifier\\nSAME hyperparams as MetaModel\\nbut ONLY 2 input features\\n(walk-forward winner)", fillcolor="#d6c5e8"];',
         'prob [label="P(above-median next-21d)", fillcolor="#ffd6a5"];',
         'top [label="Top 10 by P(>0.5)\\nmonthly rebalance", fillcolor="#d4f0d2"];',
         'pos [label="Portfolio\\nATR×3 stop", fillcolor="#d6c5e8"];'],
        [('sent', 'sent3d'), ('earn', 'near'), ('sent3d', 'gbm'), ('near', 'gbm'),
         ('lab', 'gbm'), ('gbm', 'prob'), ('prob', 'top'), ('top', 'pos')],
        "SentimentMeta — 2-feature variant of MetaModel (walk-forward winner)",
    ),
    "RuleBased (MetaModel_PWeighted)": _dot(
        ['gbm [label="Trained 13-feature\\nGradientBoostingClassifier\\n(same as MetaModel)", fillcolor="#d6c5e8"];',
         'prob [label="P(above-median next-21d)\\nper (date, tic)", fillcolor="#ffd6a5"];',
         'top [label="Top 10 by P(>0.5)", fillcolor="#d4f0d2"];',
         'wts [label="Weight = (P − 0.5)\\nnormalized across top-10\\n(conviction-weighted)", fillcolor="#ffd6a5"];',
         'pos [label="Allocate per weight\\nMonthly rebalance + ATR×3 stop", fillcolor="#d6c5e8"];'],
        [('gbm', 'prob'), ('prob', 'top'), ('top', 'wts'), ('wts', 'pos')],
        "MetaModel_PWeighted — 13-feature MetaModel with conviction-weighted sizing",
    ),
    "RuleBased (SentimentMeta_PWeighted)": _dot(
        ['gbm [label="Trained 2-feature\\nGradientBoostingClassifier\\n(same as SentimentMeta)", fillcolor="#d6c5e8"];',
         'prob [label="P(above-median next-21d)\\nfrom sent_3d + near_earnings", fillcolor="#ffd6a5"];',
         'top [label="Top 10 by P(>0.5)", fillcolor="#d4f0d2"];',
         'wts [label="Weight = (P − 0.5)\\nKelly-style conviction tilt", fillcolor="#ffd6a5"];',
         'pos [label="Allocate per weight\\nMonthly rebalance + ATR×3 stop", fillcolor="#d6c5e8"];'],
        [('gbm', 'prob'), ('prob', 'top'), ('top', 'wts'), ('wts', 'pos')],
        "SentimentMeta_PWeighted — sentiment GBM + conviction-weighted sizing (project-best)",
    ),
    "Ensemble": _dot(
        ['rsi [label="RuleBased (RSI)", fillcolor="#cfe8ff"];',
         'reg [label="RuleBased (RegimeAdaptive)", fillcolor="#cfe8ff"];',
         'cmom [label="RuleBased (CrossMomentum)", fillcolor="#cfe8ff"];',
         'arank [label="RuleBased (AnalystRank)", fillcolor="#cfe8ff"];',
         'sharpe [label="Rolling 30-day Sharpe\\nper voter", fillcolor="#ffe9b3"];',
         'soft [label="exp(Sharpe × 5)\\nsoft-Sharpe weights", fillcolor="#ffd6a5"];',
         'vote [label="Weighted vote per\\n(date, ticker)", fillcolor="#fad2cf"];',
         'conv [label="Conviction-sized buy\\n(0.3 to 1.0 of cash)", fillcolor="#d4f0d2"];',
         'pos [label="Portfolio\\nATR×3 stop + ladder", fillcolor="#d6c5e8"];'],
        [('rsi', 'sharpe'), ('reg', 'sharpe'), ('cmom', 'sharpe'), ('arank', 'sharpe'),
         ('sharpe', 'soft'), ('soft', 'vote'), ('vote', 'conv'), ('conv', 'pos')],
        "Ensemble — soft-Sharpe weighted vote of 4 rule strategies",
    ),
    "S&P 500 Baseline": _dot(
        ['idx [label="^GSPC index", fillcolor="#cfe8ff"];',
         'bnh [label="Buy & hold\\n(no rebalance)", fillcolor="#ffe9b3"];',
         'pos [label="Always 100% invested", fillcolor="#d6c5e8"];'],
        [('idx', 'bnh'), ('bnh', 'pos')],
        "S&P 500 — passive benchmark",
    ),
}

# ---------------------------------------------------------------------------
# Strategy explainers
# ---------------------------------------------------------------------------
INFO = {
    "PPO": {
        "signal_type": "Deep RL (on-policy)",
        "data_sources": ["Yahoo OHLCV", "Alpha Vantage news sentiment", "8 technical indicators"],
        "parameters": {"obs space": "298-dim (training) / 271 (current)",
                        "action": "discrete |a| ≤ 100 shares",
                        "tickers": "27 (DRL universe)"},
        "description": (
            "PPO (Proximal Policy Optimization) is an on-policy actor-critic algorithm trained on the "
            "FinRL StockTradingEnv. It learns a policy mapping (price + indicators + sentiment + holdings) "
            "to trade actions. Currently failing to load due to an observation-shape regression "
            "(298 → 271) introduced after Spring I; metrics are unavailable for the test window."
        ),
    },
    "A2C": {
        "signal_type": "Deep RL (synchronous actor-critic)",
        "data_sources": ["Yahoo OHLCV", "Alpha Vantage news sentiment", "8 technical indicators"],
        "parameters": {"obs space": "298-dim", "action": "discrete |a| ≤ 100 shares", "tickers": "27"},
        "description": (
            "A2C is the synchronous variant of A3C — same actor-critic structure as PPO but without the "
            "clipped surrogate. Same FinRL environment, same observation regression issue."
        ),
    },
    "DDPG": {
        "signal_type": "Deep RL (off-policy deterministic)",
        "data_sources": ["Yahoo OHLCV", "AV sentiment", "8 indicators"],
        "parameters": {"obs space": "298-dim", "action": "continuous", "tickers": "27"},
        "description": (
            "DDPG is an off-policy deterministic policy gradient with a critic network. Replaced TD3 "
            "in our pipeline due to FinRL stability. Same observation regression issue as PPO/A2C."
        ),
    },
    "PPO_GATED": {
        "signal_type": "DRL + LangGraph signal override",
        "data_sources": ["PPO base actions", "LangGraph bull/bear/executor debate"],
        "parameters": {"gate": "veto / dampen / tilt", "memory": "TF-IDF over prior reports"},
        "description": (
            "Same PPO policy, but each (date, ticker) action is post-processed by a LangGraph debate "
            "(bull analyst, bear analyst, executor). The gate vetoes wrong-side trades, dampens "
            "low-conviction sizes, or tilts toward the agreed direction."
        ),
    },
    "A2C_GATED": {
        "signal_type": "DRL + LangGraph signal override",
        "data_sources": ["A2C base actions", "LangGraph debate"],
        "parameters": {"gate": "veto / dampen / tilt"},
        "description": "A2C with the same LangGraph gate layer as PPO_GATED.",
    },
    "DDPG_GATED": {
        "signal_type": "DRL + LangGraph signal override",
        "data_sources": ["DDPG base actions", "LangGraph debate"],
        "parameters": {"gate": "veto / dampen / tilt"},
        "description": "DDPG with the same LangGraph gate layer.",
    },
    "RuleBased (SMA)": {
        "signal_type": "Trend-following (single-ticker)",
        "data_sources": ["Yahoo OHLCV"],
        "parameters": {"fast SMA": 50, "slow SMA": 200, "stop": "ATR×3.0", "tickers": "65"},
        "description": (
            "Pure golden-cross / death-cross trend-follower: buys when SMA50 crosses above SMA200, "
            "exits on the reverse cross or the ATR(14)×3.0 trailing stop. Rare trades (only on regime "
            "shifts) but very tight risk profile — best MDD of all strategies in our backtest."
        ),
    },
    "RuleBased (RSI)": {
        "signal_type": "Mean-reversion (single-ticker)",
        "data_sources": ["Yahoo OHLCV"],
        "parameters": {"RSI period": 14, "buy": "RSI<35", "sell": "RSI>75",
                        "momentum filter": "63d return ≥ 0",
                        "stop": "ATR×2.5", "ladder": "1/3 at +15%, 1/3 at +30%"},
        "description": (
            "RSI dip-buy with momentum filter. Buys when RSI<35 (oversold) AND the 63-day return is "
            "non-negative (the momentum filter blocks falling-knife trades). Exits on RSI>75 or the "
            "tight ATR×2.5 stop (mean-reversion theses are either right in days or wrong)."
        ),
    },
    "RuleBased (SMA_RSI)": {
        "signal_type": "Trend-confirmed mean-reversion",
        "data_sources": ["Yahoo OHLCV"],
        "parameters": {"trend": "SMA50>SMA200", "dip": "RSI<40 with prior bar ≥40",
                        "momentum filter": "63d return ≥ 0", "stop": "ATR×3.0"},
        "description": (
            "Combines SMA trend confirmation with RSI pullback timing — buys only on shallow dips "
            "INSIDE confirmed uptrends. Sells on trend reversal (death cross), not RSI overbought."
        ),
    },
    "RuleBased (SMA_RSI_Sentiment)": {
        "signal_type": "Sentiment-gated SMA_RSI",
        "data_sources": ["OHLCV", "Combined sentiment (AV+RSS+Telegram+Apify)"],
        "parameters": {"sentiment threshold": "> +0.1 on 3d MA",
                        "coverage fallback": "<10% coverage → no gate",
                        "stop": "ATR×3.0"},
        "description": (
            "SMA_RSI base signal plus a sentiment buy-gate: only fires when the 3-day-smoothed sentiment "
            "exceeds +0.1. Coverage-aware fallback: if a ticker has <10% sentiment coverage, the gate "
            "is disabled and the base signal passes through. Standalone Sharpe is below SMA_RSI in "
            "this regime — news sentiment turns out to be priced in for mega-caps."
        ),
    },
    "RuleBased (Dynamic)": {
        "signal_type": "Sentiment + vol-weighted SMA_RSI",
        "data_sources": ["OHLCV", "Combined sentiment"],
        "parameters": {"vol weight": "inverse 60d realized vol",
                        "sent weight": "[0.5, 2.0]", "size": "RSI conviction × sentiment boost"},
        "description": (
            "Allocates capital across tickers via inverse-vol × sentiment weighting (so calm + "
            "well-liked names get bigger buckets), then sizes each entry by RSI conviction depth × "
            "intraday sentiment. Soft-sells half on weak exit signals."
        ),
    },
    "RuleBased (RegimeAdaptive)": {
        "signal_type": "Regime-switching with longs + shorts",
        "data_sources": ["OHLCV", "SPY trend"],
        "parameters": {"regimes": "Mom_LowVol / HalfHedge / Short_Only / Flat",
                        "shorts": "only when SPY is in a death cross",
                        "tx cost": "0.1%"},
        "description": (
            "Picks a target exposure per ticker per day from {+1, +0.5, 0, -1} based on SMA50/200, "
            "SMA20/50, and 60-day vol vs. its 252-day median. Shorts are gated by an SPY trend filter — "
            "you can only short individual names when SPY itself is in a death cross."
        ),
    },
    "RuleBased (SentimentMomentum)": {
        "signal_type": "Sentiment surprise + trend",
        "data_sources": ["OHLCV", "Combined sentiment (forward-filled)"],
        "parameters": {"short MA": 5, "long MA": 30,
                        "buy threshold": "surprise > +0.05",
                        "sell threshold": "surprise < -0.10",
                        "trend": "SMA50>SMA200"},
        "description": (
            "Buys when the 5-day sentiment moving average exceeds the 30-day baseline by more than "
            "+0.05 (positive surprise) AND the trend is up. Sells on surprise reversal or trend break. "
            "Standalone Sharpe is negative — the sentiment-surprise signal lacks alpha for our universe."
        ),
    },
    "RuleBased (CrossMomentum)": {
        "signal_type": "Cross-sectional momentum",
        "data_sources": ["OHLCV (65 tickers)"],
        "parameters": {"lookback": "63 days", "top N": 10,
                        "rebalance": "every 21 trading days", "stop": "ATR×3.0"},
        "description": (
            "Each rebalance day, ranks all 65 tickers by trailing 63-day return and holds the top 10 "
            "(equal-weight, positive-momentum names only). Standalone delivers near-S&P returns; in "
            "the ensemble it acts as the high-return cross-sectional voter."
        ),
    },
    "RuleBased (SentimentRank)": {
        "signal_type": "Cross-sectional sentiment ranking",
        "data_sources": ["Combined sentiment"],
        "parameters": {"surprise window": "5d - 30d", "top N": 10,
                        "ranking": "Z-score across tickers", "rebalance": "21 days"},
        "description": (
            "Computes per-ticker sentiment surprise (5d - 30d MA), Z-scores across the universe, and "
            "longs the top 10. Standalone Sharpe is negative — even cross-sectionally, aggregated news "
            "sentiment lacks alpha here. Documented as a negative ablation result."
        ),
    },
    "RuleBased (AnalystRank)": {
        "signal_type": "External — Wall Street analyst flow",
        "data_sources": ["yfinance upgrades_downgrades (33,605 events cached)"],
        "parameters": {"window": "60 days", "top N": 10, "rebalance": "21 days",
                        "scoring": "+1 up / -1 down / ±0.5 init / ±0.5 target raise/lower"},
        "description": (
            "Cross-sectional ranking by net analyst conviction shifts. Each rebalance, sums signed "
            "upgrade/downgrade and price-target events from the last 60 days; longs the top 10 with "
            "positive net score. Beats S&P standalone (Sharpe 0.696 vs 0.669) and is the best new "
            "voter in the ensemble. The literature (Womack 1996, Jegadeesh-Kim 2010) consistently "
            "shows analyst revisions lead price by 1-3 days."
        ),
    },
    "RuleBased (InsiderRank)": {
        "signal_type": "External — corporate insider $ flow",
        "data_sources": ["yfinance insider_transactions (7,637 events, 2,864 discretionary)"],
        "parameters": {"window": "60 days", "top N": 10, "rebalance": "21 days",
                        "scoring": "+$ on Purchase / -$ on Sale / 0 on Award/Grant/Tax"},
        "description": (
            "Cross-sectional ranking by net discretionary insider dollar flow over the last 60 days. "
            "Sales outnumber buys ~18:1 in mega-caps (vested-stock liquidations are routine), so the "
            "signal works as 'least-negative selling' rather than 'highest buying'. Best MDD of the "
            "cross-sectional rules at 12.53%. Cohen-Malloy-Pomorski 2012 documents the academic effect."
        ),
    },
    "RuleBased (EarningsSentiment)": {
        "signal_type": "Sentiment-gated SMA_RSI, earnings-localized",
        "data_sources": ["OHLCV", "Combined sentiment", "yfinance earnings_dates (1,588 cached)"],
        "parameters": {"earnings window": "±5 trading days (~7 calendar)",
                        "in-window gate": "sentiment > +0.1",
                        "out-of-window": "pass through (no gate)", "stop": "ATR×3.0"},
        "description": (
            "Same trend + dip base as SMA_RSI, but the sentiment buy-gate is only active inside the "
            "±5 trading-day window of an earnings release. Outside earnings windows, sentiment is "
            "ignored and the base signal passes through. Result: Sharpe 0.670 vs SMA_RSI's 0.489 — the "
            "first time news sentiment added measurable alpha in our pipeline. Confirms the academic "
            "finding that sentiment alpha is concentrated in earnings windows."
        ),
    },
    "RuleBased (MetaModel)": {
        "signal_type": "Supervised GBM (analyst + price + sentiment + earnings + interaction)",
        "data_sources": [
            "OHLCV (price/momentum/vol)",
            "yfinance upgrades_downgrades (analyst features)",
            "yfinance earnings_dates (earnings proximity)",
            "Combined sentiment (FinBERT-scored, ffilled)",
        ],
        "parameters": {
            "model": "sklearn GradientBoostingClassifier",
            "trees": 200, "depth": 3, "lr": 0.05, "subsample": 0.8,
            "features": "15 (mom_5/21/63d, RSI, vol_60d, net_60d_analyst, "
                         "n_events, days_since_up/down, target_upside, near_earnings, "
                         "sent_3d, sent_x_near_earn, last_eps_surprise_pct, pead_decay)",
            "label": "next_21d_return > cross-sectional median (binary)",
            "training": "2022-03 → 2025-01 (~41,864 rows; train accuracy 0.629)",
            "rebalance": "every 21 trading days, top 10 by P(>0.5)",
            "stop": "ATR×3.0",
        },
        "description": (
            "Supervised meta-strategy: a GradientBoostingClassifier learns to predict whether each "
            "ticker will rank in the upper half of the universe by 21-day forward return. Inputs blend "
            "price (momentum, RSI, vol), analyst conviction (60-day net score, days-since-events, "
            "price-target upside), earnings proximity, 3-day sentiment MA, sent_3d × near_earnings "
            "interaction (Ablation C: +0.146 Sharpe), and PEAD features — last_eps_surprise_pct + "
            "an exp(-days/30)-decayed version (Ablation F: +0.089 Sharpe, -2.1pp MDD). Trained on "
            "41,864 rows from 2022-2025 (training-window only). Test window: **41.88% return, "
            "Sharpe 1.151, MDD 16.11%** — beats S&P 500 on every metric (+20.74pp return, +0.482 "
            "Sharpe, -2.79pp MDD). The 2-feature SentimentMeta sibling still posts a higher "
            "headline Sharpe (1.454); MetaModel is the more diversified configuration."
        ),
    },
    "RuleBased (SentimentMeta)": {
        "signal_type": "Supervised GBM on sentiment + earnings only (walk-forward winner)",
        "data_sources": [
            "yfinance earnings_dates (earnings proximity)",
            "Combined sentiment (FinBERT-scored, ffilled)",
        ],
        "parameters": {
            "model": "sklearn GradientBoostingClassifier (same as MetaModel)",
            "trees": 200, "depth": 3, "lr": 0.05, "subsample": 0.8,
            "features": "2 (sent_3d, near_earnings)",
            "label": "next_21d_return > cross-sectional median (binary)",
            "training": "2022-03 → 2025-01 (~41,864 rows)",
            "rebalance": "every 21 trading days, top 10 by P(>0.5)",
            "stop": "ATR×3.0",
        },
        "description": (
            "A separate strategy from MetaModel that uses the SAME training "
            "infrastructure but only TWO features: sent_3d (3-day MA of "
            "combined sentiment) and near_earnings (±5 trading-day flag). "
            "Discovered via Ablation C, then validated by 9-quarter walk-forward "
            "cross-validation (2024-Q1 to 2026-Q1, results in "
            "results/ablation/meta_walkforward.csv): mean fold Sharpe 0.920 vs "
            "the 13-feature MetaModel's 0.807. Wins 6 of 9 folds. The signal "
            "is asymmetric: when sentiment fires near earnings, it concentrates "
            "capital aggressively; when nothing fires, it sits in cash earning "
            "the risk-free rate, which gives it defensive behaviour in 2 of 3 "
            "bad regimes. Fragile in the third (Q1-2026 fold: -1.843 Sharpe vs "
            "MetaModel's -1.464). Deployed alongside MetaModel — not as a "
            "replacement but as a parallel strategy so the two configurations "
            "can be tracked and compared under any future regime."
        ),
    },
    "RuleBased (MetaModel_PWeighted)": {
        "signal_type": "13-feature MetaModel with conviction-weighted sizing",
        "data_sources": ["Same as MetaModel"],
        "parameters": {
            "model": "Same trained classifier as MetaModel",
            "sizing": "weight_i = (P_i − 0.5) / Σ(P_j − 0.5) across top-10",
            "rebalance": "every 21 trading days",
            "stop": "ATR×3.0",
        },
        "description": (
            "Same trained 13-feature GBM as MetaModel, but instead of equal-weighting "
            "the top-10 picks each rebalance, capital is allocated proportional to "
            "(P − 0.5) — higher-conviction picks get more capital. On the deployment "
            "window: 37.26% return, Sharpe 0.968, MDD 16.90%. The Sharpe regressed "
            "slightly from equal-weight (1.062 → 0.968) — suggests the 13-feature "
            "model's probabilities are noisy enough that tilting toward extremes "
            "amplifies error. MDD did improve by 1.33pp. Documented for the "
            "position-sizing comparison; not the recommended deployment for this "
            "feature set."
        ),
    },
    "RuleBased (SentimentMeta_PWeighted)": {
        "signal_type": "2-feature SentimentMeta with conviction-weighted sizing (project-best)",
        "data_sources": ["Same as SentimentMeta"],
        "parameters": {
            "model": "Same trained classifier as SentimentMeta",
            "sizing": "weight_i = (P_i − 0.5) / Σ(P_j − 0.5) across top-10",
            "rebalance": "every 21 trading days",
            "stop": "ATR×3.0",
        },
        "description": (
            "Same trained 2-feature GBM as SentimentMeta, with conviction-weighted "
            "position sizing. On the deployment window: **63.67% return, Sharpe "
            "1.604, MDD 16.31%** — the project's best standalone strategy on every "
            "metric, beating S&P 500 by +42.53pp return and 2.4× Sharpe. The "
            "Sharpe lift over equal-weight SentimentMeta (+0.150) is empirical "
            "evidence that the 2-feature model's probabilities are well-calibrated: "
            "P(0.55) really does map to better forward returns than P(0.51). The "
            "tilt is essentially a Kelly-style bet sizing on the model's confidence."
        ),
    },
    "Ensemble": {
        "signal_type": "Soft-Sharpe weighted vote of 4 rule strategies",
        "data_sources": ["RuleBased (RSI)", "RuleBased (RegimeAdaptive)",
                          "RuleBased (CrossMomentum)", "RuleBased (AnalystRank)"],
        "parameters": {"weight": "exp(rolling 30d Sharpe × 5)",
                        "vote": "weighted majority per (date, ticker)",
                        "size": "0.3 to 1.0 of cash by conviction",
                        "stop": "ATR×3.0 + profit ladder 15/30%"},
        "description": (
            "Headline strategy. Each voter's recent 30-day Sharpe drives a softmax weight; voters "
            "cast +1/-1 for each (date, ticker), and the weighted majority wins. Position size scales "
            "with conviction (vote margin). RegimeAdaptive is load-bearing despite negative standalone "
            "Sharpe — its decorrelated (and short-side) signals sharpen the weighter's discrimination. "
            "Achieves Sharpe 1.368 (~2× S&P) with one-third of S&P's drawdown."
        ),
    },
    "S&P 500 Baseline": {
        "signal_type": "Passive benchmark",
        "data_sources": ["^GSPC daily close"],
        "parameters": {"strategy": "buy & hold from test start"},
        "description": (
            "S&P 500 buy-and-hold baseline. All strategies are evaluated against this on Sharpe, "
            "return, and drawdown."
        ),
    },
}


def get_info(agent: str) -> dict:
    """Return a default-filled info dict for any agent."""
    return INFO.get(agent, {
        "signal_type": "Unknown",
        "data_sources": [],
        "parameters": {},
        "description": f"No detail metadata registered for '{agent}'.",
    })


def get_arch_dot(agent: str) -> str | None:
    """Return the per-strategy DOT diagram, or None if unregistered."""
    return ARCH.get(agent)
