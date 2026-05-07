"""
Dataset-Based Social Media Sentiment Collector
Uses public datasets (HuggingFace, Kaggle) instead of live APIs for development/testing.
Outputs in the same format the DRL trading system expects.

Datasets used:
1. Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment) - 11.9k tweets
2. Financial PhraseBank (financial_phrasebank) - 4.8k financial sentences
3. StockTwits style data generated from the above

Output format compatible with the DRL Trading Bot:
  - ticker, date, ticker_sentiment_score, ticker_relevance_score
  - Aggregated: weighted_avg_sentiment per (date, ticker)
"""

import os
import re
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple


# ============================================================================
# Known ticker symbols for extraction
# ============================================================================
KNOWN_TICKERS = {
    # Dow 30 (what the DRL system trades)
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOG', 'META', 'V', 'TSLA',
    'UNH', 'JPM', 'XOM', 'MA', 'AVGO', 'PG', 'HD', 'KO', 'MRK', 'ORCL',
    'CVX', 'ABBV', 'PEP', 'COST', 'ACN', 'ADBE', 'CRM', 'TMO', 'WMT', 'BAC',
    # Common names → tickers
    'GOOGL', 'GOOG', 'FB', 'NFLX', 'AMD', 'INTC', 'DIS', 'NKE', 'BA',
}

# Company name → ticker mapping
COMPANY_TO_TICKER = {
    'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN', 'nvidia': 'NVDA',
    'google': 'GOOG', 'alphabet': 'GOOG', 'meta': 'META', 'facebook': 'META',
    'tesla': 'TSLA', 'netflix': 'NFLX', 'intel': 'INTC', 'amd': 'AMD',
    'visa': 'V', 'mastercard': 'MA', 'jpmorgan': 'JPM', 'jp morgan': 'JPM',
    'exxon': 'XOM', 'chevron': 'CVX', 'walmart': 'WMT', 'costco': 'COST',
    'salesforce': 'CRM', 'adobe': 'ADBE', 'oracle': 'ORCL', 'disney': 'DIS',
    'nike': 'NKE', 'boeing': 'BA', 'coca-cola': 'KO', 'coca cola': 'KO',
    'pepsi': 'PEP', 'pepsico': 'PEP', 'merck': 'MRK', 'abbvie': 'ABBV',
    'broadcom': 'AVGO', 'procter': 'PG', 'home depot': 'HD',
    'united health': 'UNH', 'unitedhealth': 'UNH',
    'bank of america': 'BAC',
}


def extract_ticker_from_text(text: str) -> Optional[str]:
    """Extract the most likely ticker from a text using patterns + company names."""
    text_lower = text.lower()
    
    # Pattern 1: $TICKER cashtag
    cashtags = re.findall(r'\$([A-Z]{1,5})\b', text)
    for tag in cashtags:
        if tag in KNOWN_TICKERS:
            return tag
    
    # Pattern 2: Known company names
    for company, ticker in COMPANY_TO_TICKER.items():
        if company in text_lower:
            return ticker
    
    # Pattern 3: Bare ticker symbols (only if clearly uppercase in context)
    for ticker in KNOWN_TICKERS:
        if re.search(rf'\b{ticker}\b', text):
            return ticker
    
    return None


class DatasetCollector:
    """
    Loads public datasets and provides them in the format expected by
    both the sentiment agent and the DRL trading system.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize dataset collector.
        
        Args:
            cache_dir: Directory to cache downloaded datasets (defaults to datasets/ inside package)
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "datasets")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.twitter_df = None
        self.phrasebank_df = None
        self.combined_df = None
        
        # Try to load from Ollama for ticker extraction
        self.llm_extractor = None
        try:
            from .llm_ticker_extractor import OllamaTickerExtractor
            self.llm_extractor = OllamaTickerExtractor()
            print("  ✓ Ollama LLM available for ticker extraction")
        except Exception:
            print("  ⚠️  Ollama not available, using regex for ticker extraction")
    
    def load_datasets(self) -> pd.DataFrame:
        """
        Load and prepare all datasets. Returns a combined DataFrame.
        
        Columns: text, ticker, sentiment_label, sentiment_score, 
                 relevance_score, date, source
        """
        print("\n" + "="*70)
        print("  LOADING DATASETS")
        print("="*70)
        
        dfs = []
        
        # Dataset 1: Twitter Financial News Sentiment
        print("\n[1/2] Loading Twitter Financial News Sentiment...")
        twitter_df = self._load_twitter_financial_news()
        if twitter_df is not None:
            dfs.append(twitter_df)
            print(f"      ✓ {len(twitter_df)} records loaded")
        
        # Dataset 2: Financial PhraseBank
        print("\n[2/2] Loading Financial PhraseBank...")
        phrasebank_df = self._load_financial_phrasebank()
        if phrasebank_df is not None:
            dfs.append(phrasebank_df)
            print(f"      ✓ {len(phrasebank_df)} records loaded")
        
        if not dfs:
            raise RuntimeError("No datasets could be loaded!")
        
        self.combined_df = pd.concat(dfs, ignore_index=True)
        
        # Extract tickers from text
        print("\n[Ticker Extraction] Identifying stocks mentioned in text...")
        self.combined_df = self._extract_tickers(self.combined_df)
        
        # Filter to only rows with a valid ticker
        with_ticker = self.combined_df[self.combined_df['ticker'].notna()]
        
        print(f"\n✓ Combined dataset: {len(self.combined_df)} total records")
        print(f"✓ With valid tickers: {len(with_ticker)} records")
        print(f"✓ Unique tickers: {sorted(with_ticker['ticker'].unique().tolist())}")
        
        return self.combined_df
    
    def _load_twitter_financial_news(self) -> Optional[pd.DataFrame]:
        """Load Twitter Financial News Sentiment from HuggingFace."""
        cache_file = os.path.join(self.cache_dir, "twitter_financial_news.parquet")
        
        if os.path.exists(cache_file):
            print("      (loading from cache)")
            return pd.read_parquet(cache_file)
        
        try:
            from datasets import load_dataset
            
            ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
            df = ds.to_pandas()
            
            # Map labels: 0=Bearish, 1=Bullish, 2=Neutral
            label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
            score_map = {0: -0.8, 1: 0.8, 2: 0.0}
            
            df['sentiment_label'] = df['label'].map(label_map)
            df['sentiment_score'] = df['label'].map(score_map)
            df['source'] = 'twitter'
            
            # Generate synthetic dates (spread across 2022-2024)
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2024, 12, 31)
            days_range = (end_date - start_date).days
            df['date'] = [start_date + timedelta(days=random.randint(0, days_range)) for _ in range(len(df))]
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Relevance score (all tweets are financially relevant)
            df['relevance_score'] = np.random.uniform(0.5, 1.0, len(df)).round(3)
            
            # Rename text column
            if 'text' not in df.columns and 'sentence' in df.columns:
                df = df.rename(columns={'sentence': 'text'})
            
            df = df[['text', 'sentiment_label', 'sentiment_score', 'date', 'relevance_score', 'source']]
            
            # Cache
            df.to_parquet(cache_file)
            return df
            
        except ImportError:
            print("      ⚠️  'datasets' library not installed. Run: pip install datasets")
            return None
        except Exception as e:
            print(f"      ⚠️  Failed to load: {e}")
            return None
    
    def _load_financial_phrasebank(self) -> Optional[pd.DataFrame]:
        """Load Financial PhraseBank from HuggingFace."""
        cache_file = os.path.join(self.cache_dir, "financial_phrasebank.parquet")
        
        if os.path.exists(cache_file):
            print("      (loading from cache)")
            return pd.read_parquet(cache_file)
        
        try:
            from datasets import load_dataset
            
            ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
            df = ds.to_pandas()
            
            # Map labels: 0=negative, 1=neutral, 2=positive
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            score_map = {0: -0.7, 1: 0.0, 2: 0.7}
            
            df['sentiment_label'] = df['label'].map(label_map)
            df['sentiment_score'] = df['label'].map(score_map)
            df['source'] = 'financial_news'
            
            # Generate synthetic dates
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2024, 12, 31)
            days_range = (end_date - start_date).days
            df['date'] = [start_date + timedelta(days=random.randint(0, days_range)) for _ in range(len(df))]
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Relevance score
            df['relevance_score'] = np.random.uniform(0.4, 0.9, len(df)).round(3)
            
            # Rename
            if 'text' not in df.columns and 'sentence' in df.columns:
                df = df.rename(columns={'sentence': 'text'})
            
            df = df[['text', 'sentiment_label', 'sentiment_score', 'date', 'relevance_score', 'source']]
            
            # Cache
            df.to_parquet(cache_file)
            return df
            
        except ImportError:
            print("      ⚠️  'datasets' library not installed. Run: pip install datasets")
            return None
        except Exception as e:
            print(f"      ⚠️  Failed to load: {e}")
            return None
    
    def _extract_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract tickers from text using fast regex/company-name matching.
        LLM is NOT used here — too slow for bulk. Use LLM for live streaming only."""
        print(f"  Processing {len(df)} records with regex/company-name matching...")
        df['ticker'] = df['text'].apply(extract_ticker_from_text)
        matched = df['ticker'].notna().sum()
        print(f"  ✓ Found tickers in {matched}/{len(df)} records ({matched/len(df)*100:.1f}%)")
        return df

    def get_posts_for_ticker(self, ticker: str, limit: int = 100) -> List[Dict]:
        """
        Get posts about a specific ticker in the format the sentiment agent expects.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum posts to return
            
        Returns:
            List of post dicts compatible with BaseSocialMediaCollector format
        """
        if self.combined_df is None:
            self.load_datasets()
        
        ticker_df = self.combined_df[self.combined_df['ticker'] == ticker.upper()].head(limit)
        
        posts = []
        for _, row in ticker_df.iterrows():
            post = {
                'id': f"dataset_{hash(row['text']) % 100000}",
                'text': row['text'],
                'full_text': row['text'],
                'author': f"dataset_user_{random.randint(1000, 9999)}",
                'platform': row['source'],
                'channel': row['source'],
                'created_utc': datetime.combine(row['date'], datetime.min.time()).timestamp(),
                'likes': random.randint(1, 500),
                'engagement_score': row['relevance_score'],
                'has_links': False,
                'is_verified': False,
                'post_age_hours': random.randint(1, 48),
                'url': '',
                'detected_tickers': [ticker.upper()],
                'dataset_sentiment_label': row['sentiment_label'],
                'dataset_sentiment_score': row['sentiment_score'],
            }
            posts.append(post)
        
        return posts
    
    def export_for_drl(self, output_path: str = None) -> pd.DataFrame:
        """
        Export sentiment data in the EXACT format the DRL Trading Bot expects.
        
        The DRL system expects CSV with columns:
            ticker, date, ticker_sentiment_score, ticker_relevance_score
            
        Which it then aggregates into:
            date, ticker, weighted_avg_sentiment
            
        And merges with stock price data.
        """
        if self.combined_df is None:
            self.load_datasets()
        
        if output_path is None:
            output_path = os.path.join(os.path.dirname(__file__), "datasets", "sentiment_for_drl.csv")
        
        # Filter to only rows with tickers
        df = self.combined_df[self.combined_df['ticker'].notna()].copy()
        
        # Rename to match DRL expected format
        drl_df = pd.DataFrame({
            'ticker': df['ticker'],
            'date': pd.to_datetime(df['date']),
            'ticker_sentiment_score': df['sentiment_score'],
            'ticker_relevance_score': df['relevance_score'],
            'overall_sentiment_score': df['sentiment_score'],
            'overall_sentiment_label': df['sentiment_label'],
        })
        
        # Aggregate per (date, ticker) - same as DRL notebook cell 23
        drl_df['weighted_score'] = drl_df['ticker_sentiment_score'] * drl_df['ticker_relevance_score']
        
        agg_df = (
            drl_df
            .groupby(['date', 'ticker'])
            .agg(
                raw_avg_sentiment=('ticker_sentiment_score', 'mean'),
                total_relevance=('ticker_relevance_score', 'sum'),
                weighted_sum=('weighted_score', 'sum'),
                article_count=('ticker_sentiment_score', 'count')
            )
            .reset_index()
        )
        
        agg_df['weighted_avg_sentiment'] = agg_df['weighted_sum'] / agg_df['total_relevance']
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        agg_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Exported DRL-format sentiment data to {output_path}")
        print(f"  Rows: {len(agg_df)}")
        print(f"  Tickers: {sorted(agg_df['ticker'].unique().tolist())}")
        print(f"  Date range: {agg_df['date'].min()} to {agg_df['date'].max()}")
        print(f"\n  Columns: {list(agg_df.columns)}")
        print(f"  → Compatible with DRL Trading Bot cell 37+ (merge with stock data)")
        
        return agg_df


def main():
    """Demo: Load datasets, extract tickers, analyze sentiment, export for DRL."""
    print("\n" + "="*70)
    print("  DATASET-BASED SENTIMENT PIPELINE")
    print("  No APIs needed — uses public HuggingFace datasets")
    print("="*70)
    
    collector = DatasetCollector()
    
    # Step 1: Load datasets
    df = collector.load_datasets()
    
    # Step 2: Show sample data
    with_tickers = df[df['ticker'].notna()]
    print("\n" + "="*70)
    print("  SAMPLE DATA (first 10 with tickers)")
    print("="*70)
    
    sample = with_tickers.head(10)
    for _, row in sample.iterrows():
        sentiment_emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '⚪'}.get(row['sentiment_label'], '?')
        print(f"\n{sentiment_emoji} ${row['ticker']} | {row['sentiment_label']} ({row['sentiment_score']:+.1f}) | {row['source']}")
        print(f"  \"{row['text'][:120]}...\"" if len(row['text']) > 120 else f"  \"{row['text']}\"")
    
    # Step 3: Ticker distribution
    print("\n" + "="*70)
    print("  TICKER DISTRIBUTION (Top 15)")
    print("="*70)
    
    ticker_counts = with_tickers['ticker'].value_counts().head(15)
    for ticker, count in ticker_counts.items():
        sentiment_avg = with_tickers[with_tickers['ticker'] == ticker]['sentiment_score'].mean()
        bar = '█' * min(int(count / 10), 40)
        signal = '🟢' if sentiment_avg > 0.1 else ('🔴' if sentiment_avg < -0.1 else '⚪')
        print(f"  {signal} ${ticker:<6} {count:>4} posts  avg={sentiment_avg:+.2f}  {bar}")
    
    # Step 4: Get posts for a specific ticker (agent format)
    top_ticker = ticker_counts.index[0]
    print(f"\n" + "="*70)
    print(f"  POSTS FOR ${top_ticker} (Agent Format)")
    print("="*70)
    
    posts = collector.get_posts_for_ticker(top_ticker, limit=5)
    for i, post in enumerate(posts, 1):
        print(f"\n  Post {i}:")
        print(f"    Text: {post['text'][:100]}...")
        print(f"    Sentiment: {post['dataset_sentiment_label']} ({post['dataset_sentiment_score']:+.1f})")
        print(f"    Platform: {post['platform']}")
    
    # Step 5: Export for DRL trading system
    print("\n" + "="*70)
    print("  EXPORTING FOR DRL TRADING SYSTEM")
    print("="*70)
    
    agg_df = collector.export_for_drl()
    
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print("\nYou can now:")
    print("  1. Use get_posts_for_ticker() to feed the sentiment agent")
    print("  2. Use export_for_drl() output in the Trading Bot notebook")
    print("  3. Run FinBERT on the posts for live sentiment scoring")
    print()


if __name__ == "__main__":
    main()
