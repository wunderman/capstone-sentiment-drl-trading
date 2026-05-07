# Sentiment-Aware DRL Trading — Pipeline Codebase

NYUAD Engineering Capstone, Spring 2026 — Group 7
Viktor Mekvabidze, Moncif Dahaji Bouffi, Aalia Imran, Bibek Poudel
Advisors: Prof. Muhammad Shafique, Muhammad Abdullah Hanif (eBrain Lab)

This repository contains the executable pipeline behind the capstone report. It
reproduces every backtest, ablation, LangGraph debate, MetaModel training, and
the Streamlit dashboard.

## What's here

```
project/
├── pipeline.py                    Main orchestrator (data → strategies → ensemble → metrics)
├── dashboard.py                   Streamlit dashboard
├── train_drl.py                   PPO / A2C / DDPG training (Stable-Baselines3 + FinRL)
├── generate_langgraph_reports.py  Bull / Bear / Executor debate over OpenRouter
├── seed_langgraph_memory.py       Bootstrap TF-IDF memory store for LangGraph
├── langgraph_signals.py           Parse Markdown verdicts → langgraph_signals.csv
├── backfill_av_news.py            Alpha Vantage NEWS_SENTIMENT historical backfill
├── walkforward_meta.py            Walk-forward MetaModel evaluation (Ablation D)
├── ablation_sentiment.py          Ablation A — gate sweep
├── ablation_windows.py            Ablation B — RSI / SMA window sweep
├── ablation_meta_sentiment.py     Ablation C — MetaModel feature ablation
├── strategy_info.py               Strategy metadata (display names, colors)
├── fix_unknown_reports.py         Repair utility for malformed LangGraph reports
├── social_media_sentiment/        Multi-platform sentiment scrapers + FinBERT scorer
├── dashboard_data/                Cached input data (CSVs, ~28MB)
├── reports/                       Cached LangGraph trade verdicts (~165 reports)
├── requirements.txt               Pipeline runtime deps
├── requirements-dev.txt           Dev / test extras
└── .env.example                   Template for required API keys

requirements.txt                   Slim deps for Streamlit Cloud deployment
runtime.txt                        Python 3.11 pin for Streamlit Cloud
```

## Quick start

```bash
# 1. Clone + create venv (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install deps
pip install -r project/requirements.txt

# 3. Configure API keys
cp project/.env.example project/.env
# Fill in OPENROUTER_API_KEY, ALPHAVANTAGE_API_KEY, APIFY_API_KEY at minimum

# 4. Run the full pipeline (~5–10 min on a CPU laptop with cached data)
cd project && python pipeline.py

# 5. Launch the dashboard
streamlit run dashboard.py
```

`dashboard_data/` ships with cached market, sentiment, analyst, insider, and
earnings CSVs so the lab can reproduce results without re-paying for API calls.
`reports/` ships with the 162 cached LangGraph verdicts so the LLM-debate
output is byte-identical to what the report references.

## Reproducing report results

The headline backtest table, all six ablations, and every figure in the
capstone report (`capstone_report.pdf`, available in the team's separate
documentation repo) are produced by:

```bash
cd project
python pipeline.py                        # all strategies + ensemble + metrics
python ablation_sentiment.py              # Ablation A — gate sweep
python ablation_windows.py                # Ablation B — RSI/SMA window sweep
python ablation_meta_sentiment.py         # Ablation C — MetaModel features
python walkforward_meta.py                # Ablation D — walk-forward
```

All scripts write outputs to `project/results/` and `project/logs/` (both
gitignored).

## Required API keys

| Service        | Used by                                              | Free tier? |
|----------------|------------------------------------------------------|------------|
| OpenRouter     | LangGraph bull/bear/executor debate (gpt-4o-mini)    | Pay-as-you-go |
| Alpha Vantage  | `NEWS_SENTIMENT` endpoint (live + backfilled)        | 25/day free, $50/mo premium |
| Apify          | Social-media scrapers (Reddit, Telegram, X, YouTube) | $49/mo starter |

Optional (only if regenerating from scratch — cached data already covers these):

| Service     | Used by                                |
|-------------|----------------------------------------|
| Finnhub     | News RSS aggregator (free 60 req/min)  |
| Reddit API  | Reddit collector (`social_media_sentiment/collectors/reddit_collector.py`) |
| Bluesky     | Bluesky collector                      |
| Twitter v2  | X collector (legacy)                   |
| YouTube v3  | YouTube collector                      |
| StockTwits  | StockTwits collector                   |

The full `.env.example` template documents every variable.

## Environment

- Python 3.11 (use `runtime.txt` pin for Streamlit Cloud; 3.10–3.12 works locally)
- CPU-only — no GPU required; pipeline completes in <90 min on a course laptop
- Tested on macOS 14 / Linux

## Repository note

This repo contains **only the executable pipeline**. The capstone report,
LaTeX sources, poster, and prof feedback live in a separate documentation
repository.
