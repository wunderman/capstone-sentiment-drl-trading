"""
Stock Sentiment Agent
Main agent class that orchestrates data collection, filtering, and sentiment analysis.
"""

from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from .collectors import (
    RedditCollector,
    TwitterCollector,
    StockTwitsCollector,
    TelegramCollector,
    YouTubeCollector,
    BlueskyCollector,
    ApifyCollector,
    NewsCollector,
)
from .relevance_filter import RelevanceFilter
from .llm_relevance_filter import LLMRelevanceFilter
from .sentiment_analyzer import SentimentAnalyzer
from .database_manager import DatabaseManager


class StockSentimentAgent:
    """
    Main agent for generating sentiment-based trade signals from social media.
    """
    
    def __init__(self,
                 use_reddit: bool = True,
                 use_twitter: bool = False,
                 use_stocktwits: bool = True,
                 use_telegram: bool = True,
                 use_youtube: bool = False,
                 use_bluesky: bool = False,
                 use_apify: bool = False,
                 apify_api_key: str = None,
                 use_news: bool = True,
                 finnhub_api_key: str = None,
                 use_llm_filter: bool = False,
                 llm_model: str = "llama3.2",
                 min_quality_score: int = 40,
                 min_account_score: int = 20,
                 db_path: str = "sentiment_data.db",
                 auto_save: bool = True):
        """
        Initialize the stock sentiment agent.
        
        Args:
            use_reddit: Whether to collect from Reddit (Tier 1)
            use_twitter: Whether to collect from Twitter/X (Tier 3 - expensive)
            use_stocktwits: Whether to collect from StockTwits (Tier 1)
            use_telegram: Whether to collect from Telegram (Tier 1)
            use_youtube: Whether to collect from YouTube (Tier 2)
            use_bluesky: Whether to collect from Bluesky (Tier 2)
            min_quality_score: Minimum quality score for posts
            min_account_score: Minimum account credibility score
            db_path: Path to SQLite database file
            auto_save: Whether to automatically save results to database
        """
        print("Initializing Stock Sentiment Agent...")
        print(f"Active sources: ", end='')
        
        # Initialize collectors
        self.collectors = []
        active_sources = []
        
        if use_reddit:
            try:
                print("Reddit ", end='')
                self.reddit = RedditCollector()
                self.collectors.append(('reddit', self.reddit))
                active_sources.append('Reddit')
            except Exception as e:
                print(f"\n  ⚠️  Reddit init failed: {e}")
        
        if use_stocktwits:
            try:
                print("StockTwits ", end='')
                self.stocktwits = StockTwitsCollector()
                self.collectors.append(('stocktwits', self.stocktwits))
                active_sources.append('StockTwits')
            except Exception as e:
                print(f"\n  ⚠️  StockTwits init failed: {e}")
        
        if use_telegram:
            try:
                print("Telegram ", end='')
                self.telegram = TelegramCollector()
                self.collectors.append(('telegram', self.telegram))
                active_sources.append('Telegram')
            except Exception as e:
                print(f"\n  ⚠️  Telegram init failed: {e}")
        
        if use_youtube:
            try:
                print("YouTube ", end='')
                self.youtube = YouTubeCollector()
                self.collectors.append(('youtube', self.youtube))
                active_sources.append('YouTube')
            except Exception as e:
                print(f"\n  ⚠️  YouTube init failed: {e}")
        
        if use_bluesky:
            try:
                print("Bluesky ", end='')
                self.bluesky = BlueskyCollector()
                self.collectors.append(('bluesky', self.bluesky))
                active_sources.append('Bluesky')
            except Exception as e:
                print(f"\n  ⚠️  Bluesky init failed: {e}")
        
        if use_twitter:
            try:
                print("Twitter/X ", end='')
                self.twitter = TwitterCollector()
                self.collectors.append(('twitter', self.twitter))
                active_sources.append('Twitter/X')
            except Exception as e:
                print(f"\n  ⚠️  Twitter init failed: {e}")

        if use_apify:
            try:
                print("Apify ", end='')
                self.apify = ApifyCollector(api_key=apify_api_key)
                self.collectors.append(('apify', self.apify))
                active_sources.append('Apify (Twitter+Reddit+Telegram+YouTube+StockTwits)')
            except Exception as e:
                print(f"\n  ⚠️  Apify init failed: {e}")

        if use_news:
            try:
                print("News ", end='')
                self.news = NewsCollector(finnhub_api_key=finnhub_api_key)
                self.collectors.append(('news', self.news))
                src = 'News (Yahoo+Google'
                if self.news.use_finnhub:
                    src += '+Finnhub'
                src += ')'
                active_sources.append(src)
            except Exception as e:
                print(f"\n  ⚠️  News init failed: {e}")
        
        print()  # New line
        
        if not self.collectors:
            raise ValueError("No collectors initialized! Enable at least one data source.")
        
        # Initialize filter
        self.use_llm_filter = use_llm_filter
        if use_llm_filter:
            print("  - Setting up LLM relevance filter (Ollama)...")
            self.llm_filter = LLMRelevanceFilter(
                model=llm_model,
                min_quality=min_quality_score,
            )
        else:
            print("  - Setting up relevance filter")
        self.filter = RelevanceFilter(
            min_quality_score=min_quality_score,
            min_account_score=min_account_score
        )
        
        # Initialize sentiment analyzer
        print("  - Loading sentiment analysis model (this may take a moment)...")
        self.sentiment = SentimentAnalyzer()
        
        # Initialize database
        print("  - Setting up database connection...")
        self.db = DatabaseManager(db_path)
        self.auto_save = auto_save
        
        print(f"\n✓ Agent initialized with {len(active_sources)} sources: {', '.join(active_sources)}")
        print(f"✓ Database: {db_path} (auto-save: {'ON' if auto_save else 'OFF'})\n")
    
    def analyze_ticker(self,
                      ticker: str,
                      reddit_limit: int = 50,
                      twitter_limit: int = 50,
                      stocktwits_limit: int = 100,
                      telegram_limit: int = 50,
                      youtube_limit: int = 30,
                      bluesky_limit: int = 50,
                      apify_limit: int = 50,
                      news_limit: int = 100,
                      reddit_time_filter: str = 'day',
                      twitter_hours_back: int = 24,
                      hours_back: int = 48,
                      min_confidence: str = "LOW") -> Dict:
        """
        Analyze sentiment for a specific stock ticker across all enabled platforms.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            reddit_limit: Max Reddit posts to fetch
            twitter_limit: Max tweets to fetch
            stocktwits_limit: Max StockTwits messages to fetch
            telegram_limit: Max Telegram messages to fetch
            youtube_limit: Max YouTube comments to fetch
            bluesky_limit: Max Bluesky posts to fetch
            reddit_time_filter: Reddit time filter ('hour', 'day', 'week')
            twitter_hours_back: How many hours back to search Twitter
            min_confidence: Minimum confidence level ('LOW', 'MEDIUM', 'HIGH')
            
        Returns:
            Dictionary with analysis results and trade signal
        """
        print(f"\n{'='*60}")
        print(f"Analyzing sentiment for ${ticker}")
        print(f"{'='*60}\n")
        
        # Step 1: Collect data
        all_posts = self._collect_data(
            ticker, 
            reddit_limit, 
            twitter_limit,
            stocktwits_limit,
            telegram_limit,
            youtube_limit,
            bluesky_limit,
            reddit_time_filter,
            twitter_hours_back,
            apify_limit,
            hours_back,
            news_limit,
        )
        
        if not all_posts:
            print("⚠️  No posts found for this ticker")
            return {
                'ticker': ticker,
                'error': 'No posts found',
                'trade_signal': None,
            }
        
        print(f"✓ Collected {len(all_posts)} total posts\n")
        
        # Show collected posts preview
        print(f"{'─'*70}")
        print(f"  COLLECTED POSTS PREVIEW")
        print(f"{'─'*70}")
        for idx, post in enumerate(all_posts[:20], 1):
            text = (post.get('full_text') or post.get('text', ''))[:100].replace('\n', ' ')
            src = post.get('platform', '?')
            author = post.get('author', '?')[:15]
            print(f"  {idx:>2}. [{src}] @{author}: {text}{'…' if len(post.get('full_text','')) > 100 else ''}")
        if len(all_posts) > 20:
            print(f"  ... and {len(all_posts) - 20} more")
        print(f"{'─'*70}\n")
        
        # Step 2: Filter posts
        filtered_posts = self._filter_posts(all_posts, min_confidence)
        
        if not filtered_posts:
            print("⚠️  No posts passed the relevance filter")
            return {
                'ticker': ticker,
                'total_posts': len(all_posts),
                'filtered_posts': 0,
                'error': 'No relevant posts found',
                'trade_signal': None,
            }
        
        print(f"✓ {len(filtered_posts)}/{len(all_posts)} posts passed filter\n")
        
        # Step 3: Analyze sentiment
        sentiments = self._analyze_sentiments(filtered_posts)
        
        # Step 4: Generate trade signal
        trade_signal = self._generate_trade_signal(sentiments, filtered_posts)
        
        # Step 5: Compile results
        results = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'total_posts_collected': len(all_posts),
            'posts_passed_filter': len(filtered_posts),
            'filter_rate': len(filtered_posts) / len(all_posts) * 100,
            'trade_signal': trade_signal,
            'platform_breakdown': self._get_platform_breakdown(filtered_posts),
            'top_posts': self._get_top_posts(filtered_posts, sentiments, limit=5),
            'raw_data': {
                'all_posts': all_posts,
                'filtered_posts': filtered_posts,
                'sentiments': sentiments,
            }
        }
        
        # Print summary
        self._print_summary(results)
        
        # Save to database if auto-save is enabled
        if self.auto_save:
            run_id = self.db.save_analysis_run(results)
            results['db_run_id'] = run_id
        
        return results
    
    def _collect_data(self,
                     ticker: str,
                     reddit_limit: int,
                     twitter_limit: int,
                     stocktwits_limit: int,
                     telegram_limit: int,
                     youtube_limit: int,
                     bluesky_limit: int,
                     reddit_time_filter: str,
                     twitter_hours_back: int,
                     apify_limit: int = 50,
                     hours_back: int = 48,
                     news_limit: int = 100) -> List[Dict]:
        """Collect data from all enabled sources."""
        all_posts = []
        
        print("Step 1: Collecting data from social media...")
        
        for platform, collector in self.collectors:
            try:
                if platform == 'reddit':
                    print(f"  - Fetching from Reddit (limit={reddit_limit}, time={reddit_time_filter})...")
                    posts = collector.search_ticker(ticker, limit=reddit_limit, time_filter=reddit_time_filter)
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} Reddit posts")
                    
                elif platform == 'twitter':
                    print(f"  - Fetching from Twitter (limit={twitter_limit}, hours={twitter_hours_back})...")
                    posts = collector.search_ticker(ticker, limit=twitter_limit, hours_back=twitter_hours_back)
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} tweets")
                
                elif platform == 'stocktwits':
                    print(f"  - Fetching from StockTwits (limit={stocktwits_limit})...")
                    posts = collector.search_ticker(ticker, limit=stocktwits_limit)
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} StockTwits messages")
                
                elif platform == 'telegram':
                    print(f"  - Scraping Telegram channels (limit={telegram_limit})...")
                    posts = collector.search_ticker(ticker, limit=telegram_limit)
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} Telegram messages")
                
                elif platform == 'youtube':
                    print(f"  - Fetching YouTube comments (limit={youtube_limit})...")
                    posts = collector.search_ticker(ticker, limit=youtube_limit)
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} YouTube comments")
                
                elif platform == 'bluesky':
                    print(f"  - Fetching from Bluesky (limit={bluesky_limit})...")
                    posts = collector.search_ticker(ticker, limit=bluesky_limit)
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} Bluesky posts")

                elif platform == 'news':
                    print(f"  - Fetching news (Yahoo+Google+Finnhub, limit={news_limit})...")
                    posts = collector.search_ticker(
                        ticker, limit=news_limit, hours_back=hours_back
                    )
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} news articles")

                elif platform == 'apify':
                    print(f"  - Fetching via Apify (Twitter+Reddit, limit={apify_limit})...")
                    posts = collector.search_ticker(
                        ticker, limit=apify_limit, hours_back=hours_back
                    )
                    all_posts.extend(posts)
                    print(f"    Found {len(posts)} Apify posts")
                    
            except Exception as e:
                print(f"  ⚠️  Error collecting from {platform}: {e}")
                continue
        
        return all_posts
    
    def _filter_posts(self, posts: List[Dict], min_confidence: str) -> List[Dict]:
        """Filter posts based on relevance and quality."""
        
        if self.use_llm_filter:
            print("Step 2: Filtering posts with LLM (Ollama)...")
            return self.llm_filter.filter_batch(posts)
        
        print("Step 2: Filtering posts for relevance and quality...")
        
        filtered = []
        confidence_map = {"LOW": 40, "MEDIUM": 60, "HIGH": 90}
        min_conf_score = confidence_map.get(min_confidence, 40)
        
        for post in tqdm(posts, desc="Filtering"):
            # Get text to analyze
            text = post.get('full_text') or post.get('text', '')
            
            # Apply filter
            filter_result = self.filter.filter_post(
                text=text,
                account_age_days=post.get('account_age_days', 0),
                karma_or_followers=post.get('author_karma') or post.get('author_followers', 0),
                engagement_score=post.get('engagement_score', 0),
                is_verified=post.get('is_verified', False),
                has_links=post.get('has_links', False),
                subreddit=post.get('subreddit'),
                post_age_hours=post.get('post_age_hours', 0)
            )
            
            # Check if passes filter and meets confidence threshold
            if filter_result['passes_filter'] and filter_result['confidence_percentage'] >= min_conf_score:
                # Add filter results to post
                post['filter_results'] = filter_result
                filtered.append(post)
        
        return filtered
    
    def _analyze_sentiments(self, posts: List[Dict]) -> List[Dict]:
        """Analyze sentiment of filtered posts."""
        print("Step 3: Analyzing sentiment with FinBERT...")
        
        # Extract texts
        texts = [post.get('full_text') or post.get('text', '') for post in posts]
        
        # Analyze in batch for efficiency
        sentiments = self.sentiment.analyze_batch(texts, batch_size=16)
        
        # Show sentiment results for each post
        print(f"\n{'─'*70}")
        print(f"  SENTIMENT ANALYSIS RESULTS")
        print(f"{'─'*70}")
        for idx, (post, sent) in enumerate(zip(posts, sentiments), 1):
            text = (post.get('full_text') or post.get('text', ''))[:80].replace('\n', ' ')
            label = sent.get('label', '?').upper()
            score = sent.get('score', 0)
            icon = {'POSITIVE': '\U0001f7e2', 'NEGATIVE': '\U0001f534', 'NEUTRAL': '\u26aa'}.get(label, '\u2753')
            print(f"  {idx:>2}. {icon} {label:<8} ({score:+.3f})  \"{text}{'…' if len(post.get('full_text','')) > 80 else ''}\"")
        print(f"{'─'*70}\n")
        
        print(f"  ✓ Analyzed {len(sentiments)} posts")
        
        return sentiments
    
    def _generate_trade_signal(self, sentiments: List[Dict], posts: List[Dict]) -> Dict:
        """Generate trade signal from sentiments."""
        print("Step 4: Generating trade signal...")
        
        # Use confidence scores as weights
        weights = [post['filter_results']['confidence_percentage'] / 100 for post in posts]
        
        # Aggregate sentiment
        agg_sentiment = self.sentiment.aggregate_sentiment(sentiments, weights)
        
        # Calculate trade signal
        trade_signal = self.sentiment.calculate_trade_signal(agg_sentiment)
        
        return trade_signal
    
    def _get_platform_breakdown(self, posts: List[Dict]) -> Dict:
        """Get breakdown by platform."""
        breakdown = {}
        
        for post in posts:
            platform = post.get('platform', 'unknown')
            breakdown[platform] = breakdown.get(platform, 0) + 1
        
        return breakdown
    
    def _get_top_posts(self, posts: List[Dict], sentiments: List[Dict], limit: int = 5) -> List[Dict]:
        """Get top posts by confidence and engagement."""
        # Combine posts with sentiments
        combined = []
        for post, sentiment in zip(posts, sentiments):
            combined.append({
                'platform': post.get('platform'),
                'text': (post.get('full_text') or post.get('text', ''))[:200] + '...',
                'confidence': post['filter_results']['confidence_percentage'],
                'sentiment_score': sentiment['score'],
                'sentiment_label': sentiment['label'],
                'engagement': post.get('score') or post.get('likes', 0),
                'author': post.get('author'),
                'created': post.get('created_utc'),
            })
        
        # Sort by confidence * engagement
        combined.sort(key=lambda x: x['confidence'] * (x['engagement'] + 1), reverse=True)
        
        return combined[:limit]
    
    def _print_summary(self, results: Dict):
        """Print analysis summary."""
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        signal = results['trade_signal']
        
        print(f"\nTicker: ${results['ticker']}")
        print(f"Posts Analyzed: {results['posts_passed_filter']} (from {results['total_posts_collected']} collected)")
        print(f"Filter Rate: {results['filter_rate']:.1f}%")
        
        print(f"\nPlatform Breakdown:")
        for platform, count in results['platform_breakdown'].items():
            print(f"  - {platform.capitalize()}: {count} posts")
        
        print(f"\n{'─'*60}")
        print("TRADE SIGNAL")
        print(f"{'─'*60}")
        print(f"Action: {signal['action']}")
        print(f"Sentiment Score: {signal['sentiment_score']:.3f} (-1 to 1)")
        print(f"Signal Strength: {signal['signal_strength']:.1f}/100")
        print(f"Reliability: {signal['reliability']:.1f}/100")
        print(f"Confidence: {signal['confidence']:.1f}%")
        print(f"\nRecommendation: {signal['recommendation']}")
        print(f"{'='*60}\n")
    
    def monitor_ticker(self,
                      ticker: str,
                      interval_minutes: int = 60,
                      duration_hours: int = 24):
        """
        Monitor a ticker over time (continuous analysis).
        
        Args:
            ticker: Stock ticker to monitor
            interval_minutes: How often to run analysis
            duration_hours: How long to monitor
        """
        import time
        
        print(f"Starting continuous monitoring of ${ticker}")
        print(f"Interval: {interval_minutes} minutes")
        print(f"Duration: {duration_hours} hours\n")
        
        results_history = []
        end_time = datetime.now().timestamp() + (duration_hours * 3600)
        
        while datetime.now().timestamp() < end_time:
            # Run analysis
            results = self.analyze_ticker(ticker)
            results_history.append(results)
            
            # Wait for next interval
            print(f"\nNext analysis in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
        
        print("\n✓ Monitoring complete!")
        return results_history
