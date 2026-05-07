"""
Simple Test Agent - Test without API credentials
This script demonstrates the filtering and sentiment analysis without requiring API setup.
"""

from ..relevance_filter import RelevanceFilter
from ..sentiment_analyzer import SentimentAnalyzer


def test_without_apis():
    """Test the agent's core functionality without needing API credentials."""
    
    print("="*80)
    print("STOCK SENTIMENT AGENT - OFFLINE TEST")
    print("="*80)
    print("\nThis test demonstrates filtering and sentiment analysis")
    print("without requiring Reddit/Twitter API credentials.\n")
    
    # Sample posts (simulating data from social media)
    sample_posts = [
        {
            'ticker': 'AAPL',
            'text': '$AAPL looking strong here. Just bought 200 shares at $175. Target is $200 based on strong iPhone sales and services growth. Stop loss at $170.',
            'account_age_days': 500,
            'karma': 8000,
            'engagement_score': 75,
            'platform': 'reddit',
            'subreddit': 'stocks'
        },
        {
            'ticker': 'TSLA',
            'text': 'TSLA breaking out above $250 resistance. RSI overbought but momentum strong. Could see $280 if volume holds. Bullish on EV sector.',
            'account_age_days': 730,
            'karma': 15000,
            'engagement_score': 90,
            'platform': 'reddit',
            'subreddit': 'wallstreetbets'
        },
        {
            'ticker': 'NVDA',
            'text': '$NVDA earnings next week. Expecting a beat on AI chip demand. Stock could pop 10% on good guidance. Adding to my position.',
            'account_age_days': 365,
            'karma': 5000,
            'engagement_score': 60,
            'platform': 'twitter',
            'subreddit': None
        },
        {
            'ticker': 'MSFT',
            'text': 'Microsoft Azure revenue crushing it. Cloud growth accelerating. MSFT undervalued at current P/E. Long term hold.',
            'account_age_days': 1000,
            'karma': 20000,
            'engagement_score': 85,
            'platform': 'reddit',
            'subreddit': 'investing'
        },
        {
            'ticker': 'GOOGL',
            'text': 'Selling my GOOGL position. Ad revenue slowing and AI competition from MSFT is concerning. Looking for better opportunities.',
            'account_age_days': 600,
            'karma': 12000,
            'engagement_score': 55,
            'platform': 'reddit',
            'subreddit': 'stocks'
        },
        {
            'ticker': 'META',
            'text': '🚀🚀🚀 META TO THE MOON!!! BUY NOW!!! GUARANTEED 100X RETURNS!!! 🚀🚀🚀',
            'account_age_days': 10,
            'karma': 50,
            'engagement_score': 200,
            'platform': 'reddit',
            'subreddit': 'wallstreetbets'
        },
        {
            'ticker': 'AAPL',
            'text': 'Just got my new iPhone and the camera is amazing! Love this phone!',
            'account_age_days': 180,
            'karma': 1000,
            'engagement_score': 20,
            'platform': 'reddit',
            'subreddit': 'apple'
        }
    ]
    
    print(f"Sample data: {len(sample_posts)} posts from Reddit and Twitter\n")
    
    # Step 1: Initialize filter
    print("Step 1: Initializing relevance filter...")
    relevance_filter = RelevanceFilter(
        min_relevance_score=1,
        min_quality_score=40,
        min_account_score=20
    )
    print("✓ Filter ready\n")
    
    # Step 2: Filter posts
    print("Step 2: Filtering posts for relevance and quality...")
    filtered_posts = []
    
    for i, post in enumerate(sample_posts, 1):
        result = relevance_filter.filter_post(
            text=post['text'],
            account_age_days=post['account_age_days'],
            karma_or_followers=post['karma'],
            engagement_score=post['engagement_score'],
            is_verified=False,
            has_links=False,
            subreddit=post.get('subreddit')
        )
        
        print(f"\n  Post {i} ({post['ticker']}): ", end='')
        
        if result['passes_filter']:
            print(f"✓ ACCEPTED (confidence: {result['confidence_percentage']}%)")
            post['filter_results'] = result
            filtered_posts.append(post)
        else:
            print(f"✗ REJECTED ({result['confidence_level']}, quality: {result['quality_score']})")
    
    print(f"\n✓ {len(filtered_posts)}/{len(sample_posts)} posts passed filter\n")
    
    if not filtered_posts:
        print("No posts passed the filter. Exiting.")
        return
    
    # Step 3: Analyze sentiment
    print("Step 3: Loading sentiment model (FinBERT)...")
    print("(This may take a minute on first run - downloading model)\n")
    
    sentiment_analyzer = SentimentAnalyzer()
    
    print("\nAnalyzing sentiment of filtered posts...")
    texts = [post['text'] for post in filtered_posts]
    sentiments = sentiment_analyzer.analyze_batch(texts)
    
    print(f"✓ Analyzed {len(sentiments)} posts\n")
    
    # Step 4: Show results by ticker
    print("="*80)
    print("RESULTS BY TICKER")
    print("="*80)
    
    # Group by ticker
    ticker_data = {}
    for post, sentiment in zip(filtered_posts, sentiments):
        ticker = post['ticker']
        if ticker not in ticker_data:
            ticker_data[ticker] = {
                'posts': [],
                'sentiments': [],
                'weights': []
            }
        
        ticker_data[ticker]['posts'].append(post)
        ticker_data[ticker]['sentiments'].append(sentiment)
        ticker_data[ticker]['weights'].append(post['filter_results']['confidence_percentage'] / 100)
    
    # Analyze each ticker
    for ticker, data in sorted(ticker_data.items()):
        print(f"\n{'-'*80}")
        print(f"${ticker}")
        print(f"{'-'*80}")
        
        # Aggregate sentiment
        agg = sentiment_analyzer.aggregate_sentiment(
            data['sentiments'],
            data['weights']
        )
        
        # Generate trade signal
        signal = sentiment_analyzer.calculate_trade_signal(agg)
        
        print(f"\nSample Size: {len(data['posts'])} posts")
        print(f"Sentiment Distribution:")
        print(f"  Positive: {agg['distribution']['positive_count']}")
        print(f"  Negative: {agg['distribution']['negative_count']}")
        print(f"  Neutral: {agg['distribution']['neutral_count']}")
        
        print(f"\nAggregated Sentiment:")
        print(f"  Score: {agg['score']:.3f} (-1 to 1)")
        print(f"  Label: {agg['label'].upper()}")
        print(f"  Confidence: {agg['confidence']:.1%}")
        
        print(f"\nTRADE SIGNAL:")
        print(f"  Action: {signal['action']}")
        print(f"  Signal Strength: {signal['signal_strength']:.1f}/100")
        print(f"  Reliability: {signal['reliability']:.1f}/100")
        print(f"  Recommendation: {signal['recommendation']}")
        
        # Show sample post
        print(f"\nExample Post:")
        sample_post = data['posts'][0]
        print(f"  Platform: {sample_post['platform']}")
        print(f"  Text: \"{sample_post['text'][:150]}...\"")
        print(f"  Sentiment: {data['sentiments'][0]['label']} ({data['sentiments'][0]['score']:.2f})")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print("="*80)
    print("\nThis demonstrates the core functionality without API credentials.")
    print("To analyze real-time data, set up APIs and run example_usage.py\n")


if __name__ == "__main__":
    test_without_apis()
