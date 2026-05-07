# Social Media Sentiment Pipeline

Generates sentiment scores for stock trading using social media data. Designed to feed into the DRL Trading Bot as part of the Capstone project.

## Folder Structure

```
social_media_sentiment/
├── __init__.py                  # Package exports
├── __main__.py                  # CLI entry point
├── stock_sentiment_agent.py     # Main orchestrator (collect → filter → analyze → signal)
├── sentiment_analyzer.py        # FinBERT-based financial sentiment scoring
├── relevance_filter.py          # Post quality & relevance filtering
├── database_manager.py          # SQLite storage for analysis runs
├── dataset_collector.py         # HuggingFace dataset loader (no APIs needed)
├── llm_ticker_extractor.py      # Ollama LLM-based ticker extraction
├── requirements.txt             # Python dependencies
├── .env.example                 # API keys template
├── collectors/                  # Platform-specific collectors
│   ├── base_collector.py        # Abstract base class
│   ├── reddit_collector.py      # Reddit (Tier 1 - free)
│   ├── stocktwits_collector.py  # StockTwits (Tier 1 - free)
│   ├── telegram_collector.py    # Telegram (Tier 1 - free, web scraping)
│   ├── youtube_collector.py     # YouTube (Tier 2)
│   ├── bluesky_collector.py     # Bluesky (Tier 2)
│   └── twitter_collector.py     # Twitter/X (Tier 3 - paid)
├── datasets/                    # Cached datasets & DRL export
│   └── sentiment_for_drl.csv   # Output for DRL Trading Bot
├── demos/                       # Demo scripts
│   ├── demo_simple.py
│   ├── demo_llm_real.py
│   └── demo_telegram_workflow.py
└── tests/                       # Test scripts
    ├── test_agent_offline.py
    ├── test_filter.py
    ├── test_telegram.py
    └── test_telegram_discovery.py
```

## Quick Start

### 1. Install dependencies

```bash
cd "/path/to/Capstone - Social Media Sentiment"
python -m venv .venv
source .venv/bin/activate
pip install -r social_media_sentiment/requirements.txt
```

### 2. Run the dataset pipeline (no APIs needed)

```bash
python -m social_media_sentiment dataset
```

This will:
- Load 9,500+ tweets from HuggingFace (Twitter Financial News Sentiment dataset)
- Extract tickers via regex + company name matching (finds ~700 records across 34 tickers)
- Export a DRL-compatible CSV to `social_media_sentiment/datasets/sentiment_for_drl.csv`

### 3. Run offline tests (no APIs, no internet)

```bash
python -m social_media_sentiment test
```

Tests the relevance filter + FinBERT sentiment analysis on sample data.

### 4. Run live analysis with Telegram (no API key needed)

```bash
python -m social_media_sentiment analyze TSLA
```

Scrapes public Telegram channels, filters posts, runs FinBERT, and outputs a BUY/SELL/HOLD signal.

> **Note:** For LLM-based ticker extraction, have [Ollama](https://ollama.ai) running with `ollama run llama3.2`.

## Using as a Python Package

```python
from social_media_sentiment import (
    StockSentimentAgent,
    DatasetCollector,
    SentimentAnalyzer,
)

# Option A: Dataset-based (offline, no APIs)
collector = DatasetCollector()
df = collector.load_datasets()
posts = collector.get_posts_for_ticker("TSLA", limit=50)
agg_df = collector.export_for_drl()  # → CSV for DRL Trading Bot

# Option B: Live analysis (Telegram, no API key)
agent = StockSentimentAgent(
    use_telegram=True,
    use_reddit=False,
    use_twitter=False,
    use_stocktwits=False,
)
result = agent.analyze_ticker("AAPL")
print(result['signal'])  # BUY / SELL / HOLD

# Option C: Just sentiment scoring
analyzer = SentimentAnalyzer()
score = analyzer.analyze_sentiment("Tesla beats earnings expectations!")
print(score)  # {'label': 'positive', 'score': 0.95, ...}
```

## DRL Trading Bot Integration

The exported CSV at `datasets/sentiment_for_drl.csv` has columns:

| Column | Description |
|--------|-------------|
| `date` | Date of the sentiment data |
| `ticker` | Stock ticker symbol |
| `raw_avg_sentiment` | Mean sentiment score for the day |
| `total_relevance` | Sum of relevance scores |
| `weighted_sum` | Sum of (sentiment × relevance) |
| `article_count` | Number of posts/articles |
| `weighted_avg_sentiment` | Final weighted score for DRL |

This matches the format expected by the Trading Bot notebook (cell 37+):

```python
df_sentiment = pd.read_csv("social_media_sentiment/datasets/sentiment_for_drl.csv")
df_sentiment.rename(columns={'ticker': 'tic'}, inplace=True)
merged_df = df.merge(df_sentiment, on=['date', 'tic'], how='inner')
```

## API Keys (Optional)

Copy `.env.example` to `.env` in the workspace root and fill in keys for platforms you want to use:

| Platform | Tier | Cost | Key Required? |
|----------|------|------|--------------|
| Telegram | 1 | Free | No |
| StockTwits | 1 | Free | Optional |
| Reddit | 1 | Free | Yes (free API) |
| YouTube | 2 | Free | Yes (free API) |
| Bluesky | 2 | Free | Optional |
| Twitter/X | 3 | $100/mo | Yes |
