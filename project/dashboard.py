"""
Interactive Trading Dashboard
==============================
Streamlit dashboard for the Sentiment-Enhanced DRL Trading Pipeline.
Run:  streamlit run dashboard.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import strategy_info as si

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")

st.set_page_config(
    page_title="DRL Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Strategy color palette — one authoritative map used across every chart so
# that strategies are visually distinct (no two strategies share a color).
# Families: DRL (primary), gated DRL (dark shade of same hue), rule-based
# (distinct hues), ensemble + baseline.
# ---------------------------------------------------------------------------
STRATEGY_COLORS = {
    # Base DRL
    "PPO":           "#1f77b4",   # steel blue
    "A2C":           "#ff7f0e",   # orange
    "DDPG":          "#2ca02c",   # green
    "TD3":           "#9467bd",   # purple
    # Gated DRL — dark shade of same hue
    "PPO_GATED":     "#08306b",   # navy
    "A2C_GATED":     "#7f2704",   # rust
    "DDPG_GATED":    "#00441b",   # forest
    "TD3_GATED":     "#3f007d",   # indigo
    # Rule-based strategies — each its own distinct hue
    "RuleBased (SMA)":                  "#d62728",   # crimson
    "RuleBased (RSI)":                  "#17becf",   # cyan
    "RuleBased (SMA_RSI)":              "#e377c2",   # magenta
    "RuleBased (SMA_RSI_Sentiment)":    "#bcbd22",   # olive
    "RuleBased (Dynamic)":              "#8c564b",   # brown
    "RuleBased (RegimeAdaptive)":       "#f4b400",   # golden yellow
    "RuleBased (SentimentMomentum)":    "#7e57c2",   # medium violet
    "RuleBased (CrossMomentum)":        "#00838f",   # deep teal
    "RuleBased (SentimentRank)":        "#c2185b",   # raspberry
    "RuleBased (AnalystRank)":          "#ef6c00",   # dark amber
    "RuleBased (InsiderRank)":          "#455a64",   # blue gray
    "RuleBased (EarningsSentiment)":    "#689f38",   # olive-lime
    "RuleBased (MetaModel)":            "#1565c0",   # deep blue (distinct from steel blue PPO)
    "RuleBased (SentimentMeta)":        "#7b1fa2",   # deep purple — sentiment-driven sibling of MetaModel
    "RuleBased (MetaModel_PWeighted)":  "#0277bd",   # cyan-blue (lighter shade of MetaModel)
    "RuleBased (SentimentMeta_PWeighted)": "#aa00ff",  # vivid violet (brighter sibling of SentimentMeta)
    # Aggregation & benchmark
    "Ensemble":          "#9e0142",   # deep wine
    "S&P 500 Baseline":  "#4d4d4d",   # dark gray
}
STRATEGY_FALLBACK = "#e15759"


def strategy_color(agent: str) -> str:
    return STRATEGY_COLORS.get(agent, STRATEGY_FALLBACK)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    data = {}

    acct_path = os.path.join(DATA_DIR, "account_values.csv")
    if os.path.exists(acct_path):
        data["accounts"] = pd.read_csv(acct_path, parse_dates=["date"])
    else:
        data["accounts"] = None

    actions_path = os.path.join(DATA_DIR, "actions.csv")
    if os.path.exists(actions_path):
        data["actions"] = pd.read_csv(actions_path)
    else:
        data["actions"] = None

    metrics_path = os.path.join(DATA_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        data["metrics"] = pd.read_csv(metrics_path)
    else:
        data["metrics"] = None

    sentiment_path = os.path.join(DATA_DIR, "sentiment.csv")
    if os.path.exists(sentiment_path):
        data["sentiment"] = pd.read_csv(sentiment_path, parse_dates=["date"])
    else:
        data["sentiment"] = None

    merged_path = os.path.join(DATA_DIR, "merged_data.csv")
    if os.path.exists(merged_path):
        data["merged"] = pd.read_csv(merged_path, parse_dates=["date"])
    else:
        data["merged"] = None

    lg_sig_path = os.path.join(DATA_DIR, "langgraph_signals.csv")
    if os.path.exists(lg_sig_path):
        try:
            data["langgraph_signals"] = pd.read_csv(lg_sig_path, parse_dates=["date"])
        except Exception:
            data["langgraph_signals"] = pd.read_csv(lg_sig_path)
    else:
        data["langgraph_signals"] = None

    lg_ovr_path = os.path.join(DATA_DIR, "langgraph_overrides.csv")
    if os.path.exists(lg_ovr_path):
        try:
            data["langgraph_overrides"] = pd.read_csv(lg_ovr_path, parse_dates=["date"])
        except Exception:
            data["langgraph_overrides"] = pd.read_csv(lg_ovr_path)
    else:
        data["langgraph_overrides"] = None

    return data


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar(data):
    st.sidebar.title("DRL Trading Dashboard")
    st.sidebar.markdown("---")

    pages = [
        "Portfolio Overview", "Agent Comparison", "Strategy Detail",
        "Trade Activity", "Sentiment Analysis", "Per-Ticker Drill Down",
        "LangGraph-Gated", "Pipeline Architecture",
    ]
    # If a click in another page asked us to navigate, set the radio's
    # session_state key BEFORE the widget renders. (Streamlit ignores `index=`
    # once a radio's `key` is in session_state, so we must overwrite the key
    # itself to programmatically navigate.)
    forced = st.session_state.pop("nav_to", None)
    if forced in pages:
        st.session_state["nav_radio"] = forced
    elif "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = "Portfolio Overview"
    page = st.sidebar.radio("Navigate", pages, key="nav_radio")

    st.sidebar.markdown("---")

    # Date range filter
    if data["accounts"] is not None:
        df = data["accounts"]
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_range) == 2:
            return page, date_range[0], date_range[1]
        return page, min_date, max_date

    return page, None, None


# ---------------------------------------------------------------------------
# PAGE: Portfolio Overview
# ---------------------------------------------------------------------------
def page_portfolio_overview(data, start_date, end_date):
    st.title("Portfolio Performance Overview")

    if data["accounts"] is None:
        st.warning("No account data found. Run `python pipeline.py` first.")
        return

    df = data["accounts"].copy()
    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

    # ---- KPI Cards (grouped by agent type, sorted by return desc) ----
    if data["metrics"] is not None:
        metrics = data["metrics"]

        def _group(agent: str) -> str:
            if agent == "S&P 500 Baseline":
                return "Benchmark"
            if agent == "Ensemble":
                return "Ensemble"
            if agent.endswith("_GATED"):
                return "DRL (LangGraph-Gated)"
            if agent in {"PPO", "A2C", "DDPG", "TD3"}:
                return "DRL Agents"
            if agent.startswith("RuleBased"):
                return "Rule-Based Strategies"
            return "Other"

        section_order = [
            "DRL Agents",
            "DRL (LangGraph-Gated)",
            "Rule-Based Strategies",
            "Ensemble",
            "Benchmark",
        ]
        section_blurb = {
            "DRL Agents":
                "Reinforcement-learning policies trained on price + Alpha Vantage news sentiment. "
                "Continuous-action trading on the original 27-ticker universe.",
            "DRL (LangGraph-Gated)":
                "Same DRL policies, but each trade is filtered by the LangGraph analyst/manager "
                "recommendation: hard-veto on opposite-sign calls, 0.5× dampen on HOLD, 0× on "
                "strong analyst sentiment disagreement.",
            "Rule-Based Strategies":
                "Deterministic technical strategies across the 65-ticker expanded universe. "
                "SMA/RSI crossovers, sentiment-weighted, Dynamic volatility-scaled allocation, "
                "and **RegimeAdaptive** — a long/short strategy that sizes by SMA20/50/200 + "
                "vol-regime vs 252d median, with an SPY trend gate that blocks shorts during "
                "broad uptrends.",
            "Ensemble":
                "Meta-strategy that blends signals from DRL and rule-based agents on dates where "
                "all sub-agents have data — diversification hedge against single-model drift. "
                "Current composition: PPO, A2C, DDPG, RuleBased (RSI), RuleBased (RegimeAdaptive); "
                "votes are weighted by each sub-agent's rolling 30-day Sharpe.",
            "Benchmark":
                "Buy-and-hold of the S&P 500 (^GSPC) over the test window — "
                "the return-to-beat for any active strategy.",
        }

        m = metrics.copy()
        m["_group"] = m["agent"].apply(_group)
        m = m.sort_values("total_return_pct", ascending=False)

        for section in section_order:
            sub = m[m["_group"] == section]
            if sub.empty:
                continue
            header_col, info_col = st.columns([6, 1])
            with header_col:
                st.markdown(f"**{section}**")
            if section in section_blurb:
                with info_col:
                    with st.popover("ⓘ What is this?", use_container_width=True):
                        st.markdown(f"**{section}**")
                        st.write(section_blurb[section])
            # wrap at 4 per row for readability
            per_row = 4
            rows_needed = (len(sub) + per_row - 1) // per_row
            for r in range(rows_needed):
                chunk = sub.iloc[r * per_row : (r + 1) * per_row]
                cols = st.columns(per_row)
                for i, (_, row) in enumerate(chunk.iterrows()):
                    with cols[i]:
                        agent_name = row["agent"]
                        # Strategy name IS the button — clicking it routes to detail.
                        clicked = st.button(
                            agent_name,
                            key=f"card_{section}_{agent_name}",
                            use_container_width=True,
                            type="secondary",
                        )
                        ret = row["total_return_pct"]
                        ret_color = "#2ca02c" if ret >= 0 else "#d62728"
                        st.markdown(
                            f"<div style='line-height:1.15;'>"
                            f"<div style='font-size:1.6rem; font-weight:700;'>"
                            f"${row['final_value']:,.0f}</div>"
                            f"<div style='color:{ret_color}; font-size:1.0rem;'>"
                            f"{ret:+.1f}%</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"Sharpe `{row['sharpe_ratio']:.2f}`  •  "
                            f"MDD `{row['max_drawdown_pct']:.1f}%`  •  "
                            f"Vol `{row['annual_volatility_pct']:.1f}%`"
                        )
                        if clicked:
                            st.session_state["selected_strategy"] = agent_name
                            st.session_state["nav_to"] = "Strategy Detail"
                            st.rerun()

        st.markdown("---")

    # ---- Portfolio Value Chart ----
    fig = go.Figure()
    for agent in df["agent"].unique():
        agent_df = df[df["agent"] == agent].sort_values("date")
        fig.add_trace(go.Scatter(
            x=agent_df["date"],
            y=agent_df["account_value"],
            name=agent,
            mode="lines",
            line=dict(
                color=strategy_color(agent),
                width=3 if agent != "S&P 500 Baseline" else 2,
                dash="dot" if agent == "S&P 500 Baseline" else "solid",
            ),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + agent + "</extra>",
        ))

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Drawdown Chart ----
    st.subheader("Drawdown Analysis")
    fig_dd = go.Figure()
    for agent in df["agent"].unique():
        agent_df = df[df["agent"] == agent].sort_values("date")
        vals = agent_df["account_value"].values
        peak = np.maximum.accumulate(vals)
        drawdown = (vals - peak) / peak * 100
        fig_dd.add_trace(go.Scatter(
            x=agent_df["date"],
            y=drawdown,
            name=agent,
            fill="tozeroy" if agent != "S&P 500 Baseline" else None,
            line=dict(color=strategy_color(agent), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>" + agent + "</extra>",
        ))

    fig_dd.update_layout(
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=300,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Agent Comparison
# ---------------------------------------------------------------------------
def page_agent_comparison(data, start_date, end_date):
    st.title("Agent Performance Comparison")

    if data["metrics"] is None:
        st.warning("No metrics data found. Run `python pipeline.py` first.")
        return

    metrics = data["metrics"]

    # ---- Metrics Table ----
    st.subheader("Key Performance Metrics")
    display_cols = ["agent", "total_return_pct", "annual_return_pct", "sharpe_ratio",
                    "sortino_ratio", "max_drawdown_pct", "annual_volatility_pct", "win_rate_pct", "final_value"]
    display_names = {
        "agent": "Agent",
        "total_return_pct": "Total Return (%)",
        "annual_return_pct": "Annual Return (%)",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "max_drawdown_pct": "Max Drawdown (%)",
        "annual_volatility_pct": "Ann. Volatility (%)",
        "win_rate_pct": "Win Rate (%)",
        "final_value": "Final Value ($)",
    }
    df_display = metrics[[c for c in display_cols if c in metrics.columns]].rename(columns=display_names)
    st.caption("Click a row to open the strategy's detail page.")
    selection = st.dataframe(
        df_display, use_container_width=True, hide_index=True,
        selection_mode="single-row", on_select="rerun", key="agent_table",
    )
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        chosen = df_display.iloc[idx]["Agent"]
        st.session_state["selected_strategy"] = chosen
        st.session_state["nav_to"] = "Strategy Detail"
        st.rerun()

    st.markdown("---")

    # ---- Bar charts ----
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            metrics, x="agent", y="total_return_pct",
            title="Total Return (%)",
            color="agent",
            color_discrete_map=STRATEGY_COLORS,
        )
        fig.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            metrics, x="agent", y="sharpe_ratio",
            title="Sharpe Ratio",
            color="agent",
            color_discrete_map=STRATEGY_COLORS,
        )
        fig.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(
            metrics, x="agent", y="max_drawdown_pct",
            title="Max Drawdown (%)",
            color="agent",
            color_discrete_map=STRATEGY_COLORS,
        )
        fig.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.bar(
            metrics, x="agent", y="annual_volatility_pct",
            title="Annual Volatility (%)",
            color="agent",
            color_discrete_map=STRATEGY_COLORS,
        )
        fig.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Cumulative return comparison ----
    if data["accounts"] is not None:
        st.subheader("Normalized Returns (Base = 1.0)")
        df = data["accounts"].copy()
        df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

        fig = go.Figure()
        for agent in df["agent"].unique():
            adf = df[df["agent"] == agent].sort_values("date")
            normalized = adf["account_value"] / adf["account_value"].iloc[0]
            fig.add_trace(go.Scatter(
                x=adf["date"], y=normalized, name=agent,
                line=dict(color=strategy_color(agent), width=2),
            ))
        fig.update_layout(
            yaxis_title="Normalized Return",
            template="plotly_white", height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Trade Activity
# ---------------------------------------------------------------------------
def page_trade_activity(data, start_date, end_date):
    st.title("Trade Activity Log")

    if data["actions"] is None:
        st.warning("No action data found. Run `python pipeline.py` first.")
        return

    df_actions = data["actions"].copy()

    # Apply date filter only if date column has any non-NaN values
    has_dates = "date" in df_actions.columns and df_actions["date"].notna().any()
    if has_dates:
        df_actions["date"] = pd.to_datetime(df_actions["date"], errors="coerce")
        mask = (
            df_actions["date"].isna()
            | (
                (df_actions["date"].dt.date >= start_date)
                & (df_actions["date"].dt.date <= end_date)
            )
        )
        df_actions = df_actions[mask]

    if df_actions.empty or "agent" not in df_actions.columns:
        st.info("No trade actions in the selected window.")
        return

    # Agent filter
    agents = sorted(df_actions["agent"].dropna().unique().tolist())
    selected_agent = st.selectbox("Select Agent", agents)
    agent_actions = df_actions[df_actions["agent"] == selected_agent].reset_index(drop=True)

    # Ticker columns = everything except bookkeeping cols
    meta_cols = {"agent", "date"}
    ticker_cols = [c for c in agent_actions.columns if c not in meta_cols]
    # Keep only numeric ticker columns that actually have values for this agent
    numeric_ticker_cols = []
    for c in ticker_cols:
        col = pd.to_numeric(agent_actions[c], errors="coerce")
        if col.notna().any():
            agent_actions[c] = col.fillna(0)
            numeric_ticker_cols.append(c)

    st.subheader(f"Actions by {selected_agent}")

    if not numeric_ticker_cols:
        st.info("No share-action columns available for this agent.")
        st.dataframe(agent_actions.head(200), use_container_width=True, hide_index=True)
        return

    # Long-format melt for summary stats
    long = agent_actions.melt(
        id_vars=[c for c in ("agent", "date") if c in agent_actions.columns],
        value_vars=numeric_ticker_cols,
        var_name="ticker",
        value_name="action_shares",
    )
    long["action_shares"] = pd.to_numeric(long["action_shares"], errors="coerce").fillna(0)
    long["action_type"] = long["action_shares"].apply(
        lambda x: "BUY" if x > 0 else ("SELL" if x < 0 else "HOLD")
    )

    # Summary metrics
    action_counts = long["action_type"].value_counts()
    total_shares_bought = long.loc[long["action_shares"] > 0, "action_shares"].sum()
    total_shares_sold = -long.loc[long["action_shares"] < 0, "action_shares"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trading Steps", len(agent_actions))
    c2.metric("Buy Actions", int(action_counts.get("BUY", 0)))
    c3.metric("Sell Actions", int(action_counts.get("SELL", 0)))
    c4.metric("Shares Bought", f"{int(total_shares_bought):,}")
    c5.metric("Shares Sold", f"{int(total_shares_sold):,}")

    # Per-ticker rollup
    st.subheader("Per-Ticker Activity")
    by_ticker = (
        long.groupby("ticker")["action_shares"]
        .agg(
            buy_steps=lambda s: int((s > 0).sum()),
            sell_steps=lambda s: int((s < 0).sum()),
            net_shares=lambda s: int(s.sum()),
            gross_volume=lambda s: int(s.abs().sum()),
        )
        .reset_index()
        .sort_values("gross_volume", ascending=False)
    )
    st.dataframe(by_ticker, use_container_width=True, hide_index=True)

    # Heatmap of actions over time
    st.subheader("Action Heatmap")
    heat_df = agent_actions[numeric_ticker_cols].copy()
    # Keep only tickers with any nonzero activity to avoid empty rows
    active_tickers = [c for c in numeric_ticker_cols if heat_df[c].abs().sum() > 0]
    if not active_tickers:
        st.info("Agent took no nonzero actions in the selected window.")
    else:
        heat_df = heat_df[active_tickers]
        fig = px.imshow(
            heat_df.T.values,
            x=list(range(len(heat_df))),
            y=active_tickers,
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(x="Trading Step", y="Ticker", color="Shares"),
            title=f"{selected_agent} — Trade Actions Over Time",
        )
        fig.update_layout(height=max(400, 18 * len(active_tickers)), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Raw data toggle
    with st.expander("Show Raw Action Data"):
        st.dataframe(agent_actions.head(200), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE: Sentiment Analysis
# ---------------------------------------------------------------------------
def page_sentiment(data, start_date, end_date):
    st.title("Sentiment Analysis")

    if data["sentiment"] is None:
        st.warning("No sentiment data found. Run `python pipeline.py` first.")
        return

    df = data["sentiment"].copy()
    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

    # Ticker selector
    tickers = sorted(df["ticker"].unique())
    selected_tickers = st.multiselect("Select Tickers", tickers, default=tickers[:5])

    if not selected_tickers:
        st.info("Select at least one ticker.")
        return

    df_filtered = df[df["ticker"].isin(selected_tickers)]

    # ---- Sentiment over time ----
    st.subheader("Weighted Average Sentiment Over Time")
    fig = go.Figure()
    for ticker in selected_tickers:
        tdf = df_filtered[df_filtered["ticker"] == ticker].sort_values("date")
        # Rolling average for smoothing
        tdf["sentiment_smooth"] = tdf["weighted_avg_sentiment"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=tdf["date"], y=tdf["sentiment_smooth"],
            name=ticker, mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>Sentiment: %{y:.3f}<extra>" + ticker + "</extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        yaxis_title="Sentiment Score",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Sentiment distribution ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        fig = px.histogram(
            df_filtered, x="weighted_avg_sentiment", color="ticker",
            nbins=50, barmode="overlay", opacity=0.6,
            title="Distribution of Daily Sentiment Scores",
        )
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Average Sentiment by Ticker")
        avg_sent = df_filtered.groupby("ticker")["weighted_avg_sentiment"].mean().reset_index()
        avg_sent.columns = ["ticker", "avg_sentiment"]
        avg_sent = avg_sent.sort_values("avg_sentiment", ascending=True)
        fig = px.bar(
            avg_sent, x="avg_sentiment", y="ticker", orientation="h",
            title="Mean Sentiment Score",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Article coverage ----
    st.subheader("Article Coverage Over Time")
    coverage = df_filtered.groupby(["date", "ticker"])["article_count"].sum().reset_index()
    fig = px.area(
        coverage, x="date", y="article_count", color="ticker",
        title="Number of Sentiment Articles per Day",
    )
    fig.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Per-Ticker Drill Down
# ---------------------------------------------------------------------------
def page_ticker_drilldown(data, start_date, end_date):
    st.title("Per-Ticker Drill Down")

    if data["merged"] is None:
        st.warning("No merged data found. Run `python pipeline.py` first.")
        return

    df = data["merged"].copy()
    df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

    ticker = st.selectbox("Select Ticker", sorted(df["tic"].unique()))
    tdf = df[df["tic"] == ticker].sort_values("date")

    if tdf.empty:
        st.info("No data for this ticker in the selected range.")
        return

    # Price + Sentiment dual axis chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Candlestick(
            x=tdf["date"],
            open=tdf["open"], high=tdf["high"], low=tdf["low"], close=tdf["close"],
            name="Price",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
        ),
        secondary_y=False,
    )

    if "weighted_avg_sentiment" in tdf.columns:
        sent_smooth = tdf["weighted_avg_sentiment"].rolling(5, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=tdf["date"], y=sent_smooth,
                name="Sentiment (5d avg)",
                line=dict(color="#f28e2b", width=2),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=f"{ticker} — Price & Sentiment",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Technical indicators
    st.subheader("Technical Indicators")
    col1, col2 = st.columns(2)

    with col1:
        if "rsi_30" in tdf.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["rsi_30"], name="RSI(30)", line=dict(color="#4e79a7")))
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
            fig.update_layout(title="RSI (30)", template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "macd" in tdf.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["macd"], name="MACD", line=dict(color="#b07aa1")))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(title="MACD", template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Bollinger Bands
    if "boll_ub" in tdf.columns and "boll_lb" in tdf.columns:
        st.subheader("Bollinger Bands")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["close"], name="Close", line=dict(color="#4e79a7")))
        fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["boll_ub"], name="Upper Band",
                                 line=dict(color="#EF5350", dash="dash")))
        fig.add_trace(go.Scatter(x=tdf["date"], y=tdf["boll_lb"], name="Lower Band",
                                 line=dict(color="#59a14f", dash="dash"),
                                 fill="tonexty", fillcolor="rgba(76,175,80,0.1)"))
        fig.update_layout(title=f"{ticker} Bollinger Bands", template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Volume
    if "volume" in tdf.columns:
        st.subheader("Volume")
        fig = px.bar(tdf, x="date", y="volume", title=f"{ticker} Trading Volume")
        fig.update_layout(template="plotly_white", height=250)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: LangGraph-Gated Backtest
# ---------------------------------------------------------------------------
def page_langgraph_gated(data, start_date, end_date):
    st.title("LangGraph-Gated DRL Backtest")
    st.caption(
        "Analyst + manager decisions from the LangGraph pipeline "
        "(`graph/trade_generation_pipeline.py`) modify DRL actions: "
        "**SELL** → veto buys, **BUY** → veto sells, **HOLD** → dampen 50%, "
        "strong sentiment disagreement → clip to 0."
    )

    if data["accounts"] is None:
        st.warning("No account data found. Run `python pipeline.py` first.")
        return

    accts = data["accounts"].copy()
    accts = accts[(accts["date"].dt.date >= start_date) & (accts["date"].dt.date <= end_date)]

    # Pair baseline vs gated agents
    agent_names = accts["agent"].unique().tolist()
    pairs = []
    for a in agent_names:
        if a.endswith("_GATED"):
            continue
        if f"{a}_GATED" in agent_names:
            pairs.append((a, f"{a}_GATED"))

    if not pairs:
        st.info(
            "No gated backtest results yet. The gate fires only when LangGraph "
            "signals exist for tickers/dates in the test window. "
            "Generate signals with `python Capstone/graph/trade_generation_pipeline.py` "
            "(requires `OPENROUTER_API_KEY` and `ALPHAVANTAGE_API_KEY`), then re-run `python pipeline.py`."
        )

    # ---- KPI: baseline vs gated deltas ----
    if pairs and data["metrics"] is not None:
        m = data["metrics"]
        cols = st.columns(len(pairs))
        for i, (base, gated) in enumerate(pairs):
            base_row = m[m["agent"] == base]
            gated_row = m[m["agent"] == gated]
            if base_row.empty or gated_row.empty:
                continue
            base_final = base_row.iloc[0]["final_value"]
            gated_final = gated_row.iloc[0]["final_value"]
            delta_pct = (gated_final - base_final) / base_final * 100.0 if base_final else 0.0
            g = gated_row.iloc[0]
            b = base_row.iloc[0]
            with cols[i]:
                st.metric(
                    label=f"{base} → {gated}",
                    value=f"${gated_final:,.0f}",
                    delta=f"{delta_pct:+.2f}% vs baseline",
                )
                st.caption(
                    f"Sharpe `{g['sharpe_ratio']:.2f}` (Δ {g['sharpe_ratio']-b['sharpe_ratio']:+.2f})  •  "
                    f"MDD `{g['max_drawdown_pct']:.1f}%` (Δ {g['max_drawdown_pct']-b['max_drawdown_pct']:+.1f})  •  "
                    f"Vol `{g['annual_volatility_pct']:.1f}%` (Δ {g['annual_volatility_pct']-b['annual_volatility_pct']:+.1f})"
                )
        st.markdown("---")

    # ---- PnL comparison chart ----
    if pairs:
        fig = go.Figure()
        for base, gated in pairs:
            # base and gated share the same hue family in STRATEGY_COLORS but
            # gated is drawn dashed so solid/dashed carries the distinction.
            for agent, dash in [(base, "solid"), (gated, "dash")]:
                sub = accts[accts["agent"] == agent].sort_values("date")
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["date"],
                    y=sub["account_value"],
                    name=agent,
                    mode="lines",
                    line=dict(color=strategy_color(agent), width=2, dash=dash),
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + agent + "</extra>",
                ))
        fig.update_layout(
            title="Account Value: Baseline (solid) vs LangGraph-Gated (dashed)",
            xaxis_title="Date",
            yaxis_title="Account Value ($)",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Signal cache ----
    st.subheader("LangGraph Signal Cache")
    sig_df = data.get("langgraph_signals")
    if sig_df is None or sig_df.empty:
        st.info(
            "No signals in cache. "
            "Files at `Capstone/reports/*_trade_recommendation_*.md` are parsed at each pipeline run."
        )
    else:
        st.caption(f"{len(sig_df)} signal(s) parsed from markdown reports.")
        show = sig_df.copy()
        if "date" in show.columns:
            show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(show, use_container_width=True, hide_index=True)

    # ---- Override log ----
    st.subheader("Trades Modified by LangGraph Gate")
    ovr = data.get("langgraph_overrides")
    if ovr is None or ovr.empty:
        st.info("No action overrides recorded. Either no signals fell within the backtest window, or signals agreed with DRL.")
    else:
        ovr_f = ovr.copy()
        if "date" in ovr_f.columns:
            ovr_f["date"] = pd.to_datetime(ovr_f["date"])
            ovr_f = ovr_f[(ovr_f["date"].dt.date >= start_date) & (ovr_f["date"].dt.date <= end_date)]
            ovr_f["date"] = ovr_f["date"].dt.strftime("%Y-%m-%d")

        # Summary by reason
        if "reason" in ovr_f.columns and not ovr_f.empty:
            by_reason = ovr_f["reason"].astype(str).str.replace(r"=.*", "=...", regex=True).value_counts().reset_index()
            by_reason.columns = ["reason", "count"]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.caption("Overrides by rule:")
                st.dataframe(by_reason, use_container_width=True, hide_index=True)
            with c2:
                fig2 = px.bar(by_reason, x="reason", y="count", title="Override reasons")
                fig2.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig2, use_container_width=True)

        st.caption(f"{len(ovr_f)} trade(s) modified in selected date range.")
        st.dataframe(ovr_f, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE: Pipeline Architecture
# ---------------------------------------------------------------------------
def page_pipeline_architecture(data, start_date, end_date):
    st.title("Pipeline Architecture")
    st.caption(
        "End-to-end flow of the Sentiment-Enhanced DRL Trading System — from raw social data "
        "to gated trade execution."
    )

    # ---- Trading timeframe ----
    st.subheader("Trading Timeframe")
    t1, t2, t3 = st.columns(3)
    t1.metric(
        "Full Data Window",
        "2022-03-02 → 2026-04-21",
        help="~4 years of daily OHLCV + sentiment",
    )
    t2.metric(
        "Training Split",
        "2022-03-02 → 2025-01-01",
        help="DRL agents learn trading policy on this window (includes 2024 AI rally)",
    )
    t3.metric(
        "Test / Dashboard Split",
        "2025-01-01 → 2026-04-21",
        help="~15 months — every chart and metric on this dashboard is drawn from this out-of-sample window",
    )
    st.caption(
        "Bar frequency is **daily**. The train/test boundary is hard-coded in "
        "`pipeline.py` (`TRAIN_START`, `TRAIN_END`, `TEST_START`, `TEST_END`). "
        "All equity curves, trade actions, S&P 500 baseline, and LangGraph gate "
        "overrides shown elsewhere in this dashboard live inside the test split."
    )

    # ---- High-level data flow ----
    st.subheader("End-to-End Data Flow")
    dot = """
    digraph G {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#444", fillcolor="#eef3f9"];
        edge [color="#666", fontname="Helvetica", fontsize=9];

        AV_cached    [label="Alpha Vantage\\n(cached CSV)", fillcolor="#dde9f5"];
        AV_live      [label="Alpha Vantage\\nNEWS_SENTIMENT (live)", fillcolor="#dde9f5"];
        Apify        [label="Apify Scrapers\\n(Reddit · Telegram · YT · StockTwits)", fillcolor="#e8f1d8"];
        Tele         [label="Telegram\\nweb collector", fillcolor="#e8f1d8"];

        FinBERT      [label="FinBERT\\nscoring", fillcolor="#f7e6c7"];
        Combined     [label="df_combined\\n(unified sentiment frame)", fillcolor="#fff4cf"];

        Price        [label="Yahoo Finance\\nOHLCV + technicals", fillcolor="#dde9f5"];
        Env          [label="FinRL\\nStockTradingEnv", fillcolor="#f0e1f2"];
        DRL          [label="DRL Agents\\nPPO · A2C · DDPG", fillcolor="#f0e1f2"];

        LG           [label="LangGraph Pipeline\\n(analysts → debate → executor)", fillcolor="#f9d9d9"];
        Reports      [label="reports/\\n*_trade_recommendation_*.md", fillcolor="#f9d9d9"];
        Signals      [label="langgraph_signals.csv", fillcolor="#f9d9d9"];
        Gate         [label="Gate Layer\\n(veto · dampen · tilt)", fillcolor="#f9d9d9"];
        DRL_G        [label="DRL (Gated)\\nPPO_GATED · A2C_GATED · DDPG_GATED", fillcolor="#f0e1f2"];

        RB           [label="Rule-Based Strategies\\nSMA · RSI · Dynamic · RegimeAdaptive\\n(+ SPY trend gate)", fillcolor="#e4eedd"];
        Ens          [label="Ensemble\\n(rolling-Sharpe weighted vote)", fillcolor="#f4e5bd"];

        Dash         [label="Streamlit\\nDashboard", fillcolor="#d9ecf2"];

        AV_cached -> Combined;
        AV_live   -> Combined;
        Apify     -> FinBERT -> Combined;
        Tele      -> FinBERT;
        Combined  -> Env;
        Price     -> Env;
        Env       -> DRL;

        Combined  -> LG;
        LG        -> Reports -> Signals -> Gate;
        DRL       -> Gate [label="raw actions"];
        Gate      -> DRL_G [label="modified trades"];
        DRL_G     -> Dash;
        DRL       -> Dash [style=dashed, label="shipped in parallel\\nfor comparison"];

        Price     -> RB;
        Combined  -> RB;
        RB        -> Ens;
        DRL       -> Ens;
        Ens       -> Dash;
        RB        -> Dash;
    }
    """
    st.graphviz_chart(dot, use_container_width=True)

    # ---- Stage-by-stage explanation ----
    st.subheader("Stage-by-Stage Explanation")

    with st.expander("1 · Sentiment Ingestion", expanded=True):
        st.markdown(
            """
            **Sources** (all merged into a single `df_combined` frame keyed by `ticker` + `date`):

            | Source | Module | Status |
            | --- | --- | --- |
            | Alpha Vantage cached CSV | `pipeline.fetch_alphavantage_sentiment` | always on |
            | Alpha Vantage live news | `pipeline.fetch_alphavantage_live_sentiment` | gated by `USE_LIVE_ALPHAVANTAGE=1` |
            | Telegram public channels | `Capstone/.../telegram_collector.py` | always on |
            | Apify (Reddit · Telegram · YouTube · StockTwits) | `pipeline.fetch_apify_social_sentiment` | gated by `USE_APIFY=1` |
            | Twitter/X (Apify) | same | gated by `USE_APIFY_TWITTER=1` |

            Scraped text is scored by **FinBERT** (`ProsusAI/finbert`) into a
            normalized sentiment score in `[-1, +1]`.
            """
        )

    with st.expander("2 · Price & Technical Features"):
        st.markdown(
            """
            Daily OHLCV pulled from **Yahoo Finance**. The FinRL preprocessor adds
            technical indicators (MACD, RSI, Bollinger, turbulence) and joins them
            with the aggregated sentiment score per ticker/day. Missing sentiment
            defaults to `0`.
            """
        )

    with st.expander("3 · DRL Agents (FinRL StockTradingEnv)"):
        st.markdown(
            """
            Three agents trained from `stable_baselines3`:

            - **PPO** — on-policy, clipped surrogate objective
            - **A2C** — synchronous advantage actor-critic
            - **DDPG** — deterministic actor-critic with replay

            Each observation includes the sentiment feature, so the agent can learn
            to condition actions on social mood. Output: per-day `(ticker → shares)`
            vectors written to `dashboard_data/actions.csv`.
            """
        )

    with st.expander("4 · LangGraph Analyst Pipeline"):
        st.markdown(
            """
            A parallel reasoning track runs in `Capstone/graph/trade_generation_pipeline.py`:

            1. **Fundamentals Analyst** — earnings, valuation, ratios
            2. **Sentiment Analyst** — aggregates Alpha Vantage + social sentiment
            3. **Bull / Bear Debate** — two manager agents argue both sides
            4. **Executor** — writes a `BUY` / `SELL` / `HOLD` recommendation with a
               **conviction** (High / Medium / Low) and a **sentiment_score**

            Each run produces a markdown file in `Capstone/reports/` named
            `<TICKER>_trade_recommendation_<DATE>.md`.
            `langgraph_signals.py` parses these into
            `dashboard_data/langgraph_signals.csv`.
            """
        )

    with st.expander("5 · Gate Layer (DRL × LangGraph fusion)"):
        st.markdown(
            """
            For every DRL action on each `(ticker, date)`, the gate looks up the
            most recent LangGraph signal on or before that date and **modifies
            the trade** before it hits the simulator:

            - **Veto** — High-conviction `SELL` blocks a BUY (and vice versa)
            - **Dampen** — Medium conviction shrinks action size
            - **Tilt** — sentiment_score nudges the share count

            The modified trade stream runs through a *second* backtest, producing
            a new agent series — `PPO_GATED`, `A2C_GATED`, `DDPG_GATED`. These
            are distinct strategies with their own equity curves and metrics.

            The **raw** DRL series is also kept and shown alongside the gated
            version (dashed edge in the diagram). This is deliberate — the pair
            lets you quantify the gate's contribution by comparing the two
            strategies directly (see the **LangGraph-Gated** page). If there is
            no in-window signal for a ticker on a given date, that gated action
            equals the raw DRL action.
            """
        )

    with st.expander("6 · Rule-Based Strategies & Ensemble"):
        st.markdown(
            """
            Alongside the DRL + LangGraph track, `pipeline.py` runs a suite of
            deterministic strategies on the full 65-ticker expanded universe:

            | Strategy | Signal | Sizing |
            | --- | --- | --- |
            | `RuleBased (SMA)` | SMA50 × SMA200 crossover | equal-dollar per ticker |
            | `RuleBased (RSI)` | RSI(14) < 30 buy / > 70 sell | equal-dollar per ticker |
            | `RuleBased (SMA_RSI)` | SMA trend × RSI oversold/overbought | equal-dollar per ticker |
            | `RuleBased (SMA_RSI_Sentiment)` | above + sentiment confirmation | equal-dollar per ticker |
            | `RuleBased (Dynamic)` | long-only, inverse-vol weighted across tickers | vol-scaled allocation |
            | `RuleBased (RegimeAdaptive)` | per-ticker regime × SPY trend gate × vol scale | target-weight rebalancing (supports shorts) |

            **RegimeAdaptive** in detail — at `pipeline.run_regime_adaptive_backtest`:
            - Per-ticker regime from `(SMA50 vs SMA200)` + `(SMA20 vs SMA50)`
            - Target weight scaled by `clip(VOL60_MED252 / VOL60, 0.5, 1.5)` so
              elevated vol shrinks exposure and calm vol mildly amplifies it
            - Bull + ST-up → full long; bull + ST-down → half long
            - Bear + ST-down → short *only* when SPY's own SMA50 < SMA200
              (market-wide trend gate to avoid squeezes in rallies)
            - Daily rebalancing toward target weight with 0.1% transaction cost

            **Ensemble** at `pipeline.run_ensemble_backtest`:
            - Sub-agents: PPO, A2C, DDPG, RuleBased (RSI), RuleBased (RegimeAdaptive)
            - Each votes BUY/SELL/HOLD per (ticker, date)
            - Vote weighted by that agent's rolling 30-day Sharpe ratio
            - Position sizing from vote margin (conviction) + volatility
            - 8% trailing stop-loss per ticker
            """
        )

    with st.expander("7 · Dashboard"):
        st.markdown(
            """
            Streamlit reads the six flat CSVs written by `pipeline.py`:

            - `account_values.csv` — daily equity curves per agent
            - `actions.csv` — wide-format per-step share decisions
            - `metrics.csv` — return / Sharpe / drawdown KPIs
            - `trades.csv` — yfinance-joined executed trades
            - `sentiment_summary.csv` — sentiment stats per ticker
            - `langgraph_signals.csv` — parsed LangGraph recommendations
            """
        )

    # ---- File-level pointer ----
    st.subheader("Source-Code Map")
    st.markdown(
        """
        | Concern | File |
        | --- | --- |
        | Orchestration | `pipeline.py` |
        | Rule-based backtests (incl. RegimeAdaptive) | `pipeline.run_multistrategy_backtest`, `pipeline.run_regime_adaptive_backtest` |
        | Ensemble | `pipeline.run_ensemble_backtest` |
        | LangGraph pipeline | `Capstone/graph/trade_generation_pipeline.py` |
        | Analyst agents | `Capstone/agents/analysts/*.py` |
        | Manager / debate | `Capstone/agents/managers/*.py` |
        | LangGraph memory (TF-IDF similarity) | `Capstone/graph/memory_store.py` |
        | Apify scraping | `social_media_sentiment/collectors/apify_collector.py` |
        | Telegram collector | `Capstone/social_media_sentiment/collectors/telegram_collector.py` |
        | Signal cache | `langgraph_signals.py` |
        | Dashboard | `dashboard.py` |
        """
    )


# ---------------------------------------------------------------------------
# PAGE: Strategy Detail (per-strategy drill-in)
# ---------------------------------------------------------------------------
def page_strategy_detail(data, start_date, end_date):
    metrics = data.get("metrics")
    accounts = data.get("accounts")
    if metrics is None or metrics.empty:
        st.warning("No metrics data found. Run `python pipeline.py` first.")
        return

    available = metrics["agent"].tolist()
    default = st.session_state.get("selected_strategy", available[0])
    if default not in available:
        default = available[0]
    chosen = st.selectbox(
        "Strategy", available, index=available.index(default), key="strategy_picker",
    )
    st.session_state["selected_strategy"] = chosen

    info = si.get_info(chosen)
    color = strategy_color(chosen)

    # Header
    st.markdown(
        f"<h1 style='color:{color}; margin-bottom:0;'>{chosen}</h1>"
        f"<p style='color:#555; font-size:1.05rem; margin-top:4px;'>{info['signal_type']}</p>",
        unsafe_allow_html=True,
    )

    # Headline metrics
    row = metrics[metrics["agent"] == chosen]
    if row.empty:
        st.error(f"No metrics row for '{chosen}'.")
        return
    r = row.iloc[0]

    sp_row = metrics[metrics["agent"] == "S&P 500 Baseline"]
    sp_sharpe = float(sp_row["sharpe_ratio"].iloc[0]) if not sp_row.empty else None
    sp_ret = float(sp_row["total_return_pct"].iloc[0]) if not sp_row.empty else None
    sp_mdd = float(sp_row["max_drawdown_pct"].iloc[0]) if not sp_row.empty else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return",
              f"{r['total_return_pct']:.2f}%",
              f"{r['total_return_pct'] - sp_ret:+.2f}pp vs S&P" if sp_ret is not None else None)
    c2.metric("Sharpe",
              f"{r['sharpe_ratio']:.3f}",
              f"{r['sharpe_ratio'] - sp_sharpe:+.3f} vs S&P" if sp_sharpe is not None else None)
    c3.metric("Sortino", f"{r['sortino_ratio']:.3f}")
    c4.metric("Max Drawdown",
              f"{r['max_drawdown_pct']:.2f}%",
              f"{r['max_drawdown_pct'] - sp_mdd:+.2f}pp vs S&P" if sp_mdd is not None else None,
              delta_color="inverse")
    c5.metric("Final Value", f"${r['final_value']:,.0f}")

    st.markdown("---")

    # Description
    st.subheader("How it works")
    st.markdown(info["description"])

    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown("**Data sources**")
        for s in info["data_sources"] or ["—"]:
            st.markdown(f"- {s}")
    with cols[1]:
        st.markdown("**Parameters**")
        if info["parameters"]:
            for k, v in info["parameters"].items():
                st.markdown(f"- **{k}**: {v}")
        else:
            st.markdown("—")

    # Architecture diagram
    st.markdown("---")
    st.subheader("Architecture")
    dot = si.get_arch_dot(chosen)
    if dot:
        st.graphviz_chart(dot, use_container_width=True)
    else:
        st.info("No architecture diagram registered for this strategy.")

    # Equity curve + benchmark
    if accounts is not None and not accounts.empty:
        st.markdown("---")
        st.subheader("Equity curve")
        df = accounts[
            (accounts["date"].dt.date >= start_date) &
            (accounts["date"].dt.date <= end_date)
        ].copy()
        fig = go.Figure()
        for ag in [chosen, "S&P 500 Baseline"]:
            sub = df[df["agent"] == ag].sort_values("date")
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["account_value"], name=ag,
                line=dict(color=strategy_color(ag), width=2.5 if ag == chosen else 1.5,
                          dash="solid" if ag == chosen else "dot"),
            ))
        fig.update_layout(template="plotly_white", height=380, hovermode="x unified",
                          yaxis_title="Account Value ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        st.subheader("Drawdown")
        sub = df[df["agent"] == chosen].sort_values("date")
        if not sub.empty:
            vals = sub["account_value"].astype(float).values
            peak = np.maximum.accumulate(vals)
            dd = (vals - peak) / peak * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=sub["date"], y=dd, fill="tozeroy", name="Drawdown",
                line=dict(color=color, width=1),
            ))
            fig_dd.update_layout(template="plotly_white", height=260,
                                 yaxis_title="Drawdown (%)", hovermode="x unified")
            st.plotly_chart(fig_dd, use_container_width=True)

    # Trade activity preview
    actions_df = data.get("actions")
    if actions_df is not None and not actions_df.empty:
        sub = actions_df[actions_df["agent"] == chosen]
        if not sub.empty:
            st.markdown("---")
            st.subheader("Recent trades (last 100 non-zero actions)")
            tic_cols = [c for c in sub.columns if c not in ("date", "agent")]
            long = sub.melt(id_vars=["date"], value_vars=tic_cols,
                            var_name="ticker", value_name="action")
            long = long[long["action"].fillna(0) != 0].tail(100)
            if long.empty:
                st.caption("No non-zero actions in the test window.")
            else:
                st.dataframe(long.sort_values("date", ascending=False).reset_index(drop=True),
                             use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    data = load_data()
    page, start_date, end_date = render_sidebar(data)

    if start_date is None:
        st.error(
            "No data found in `dashboard_data/`. "
            "Run the pipeline first:\n\n```bash\npython pipeline.py\n```"
        )
        return

    if page == "Portfolio Overview":
        page_portfolio_overview(data, start_date, end_date)
    elif page == "Agent Comparison":
        page_agent_comparison(data, start_date, end_date)
    elif page == "Strategy Detail":
        page_strategy_detail(data, start_date, end_date)
    elif page == "Trade Activity":
        page_trade_activity(data, start_date, end_date)
    elif page == "Sentiment Analysis":
        page_sentiment(data, start_date, end_date)
    elif page == "Per-Ticker Drill Down":
        page_ticker_drilldown(data, start_date, end_date)
    elif page == "LangGraph-Gated":
        page_langgraph_gated(data, start_date, end_date)
    elif page == "Pipeline Architecture":
        page_pipeline_architecture(data, start_date, end_date)


if __name__ == "__main__":
    main()
