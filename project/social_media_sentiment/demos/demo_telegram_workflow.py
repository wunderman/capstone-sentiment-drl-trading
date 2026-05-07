"""
Telegram Sentiment Analysis Demo
Demonstrates the complete workflow:
1. Discover what tickers are trending
2. Collect messages for specific ticker
3. Analyze sentiment using FinBERT
4. Generate trading signal
"""

from ..stock_sentiment_agent import StockSentimentAgent
from ..collectors.telegram_collector import TelegramCollector
from ..database_manager import DatabaseManager

def main():
    print("\n" + "="*70)
    print("  TELEGRAM SENTIMENT ANALYSIS WORKFLOW DEMO")
    print("="*70)
    
    # Initialize components
    collector = TelegramCollector()
    agent = StockSentimentAgent(auto_save=True)
    db = DatabaseManager()
    
    # Step 1: Discover trending tickers
    print("\n" + "="*70)
    print("  STEP 1: DISCOVER TRENDING TICKERS")
    print("="*70)
    print("\nScraping recent messages to see what stocks are being discussed...")
    print("(This takes ~30-60 seconds)\n")
    
    messages = collector.scrape_all_messages(limit=200)
    
    # Count ticker mentions
    ticker_counts = {}
    for msg in messages:
        for ticker in msg.get('detected_tickers', []):
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
    
    # Show top 10
    sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"✓ Analyzed {len(messages)} messages")
    print(f"\n{'Ticker':<15} {'Mentions':<10} {'Stock Name'}")
    print("-" * 70)
    
    stock_names = {
        'ADANI': 'Adani Group',
        'HDFC': 'HDFC Bank',
        'SBI': 'State Bank of India',
        'BAJAJ': 'Bajaj Group',
        'NTPC': 'NTPC (Power)',
        'REC': 'REC Limited (Finance)',
        'ICICI': 'ICICI Bank',
        'RELIANCE': 'Reliance Industries',
        'BPCL': 'Bharat Petroleum',
        'PFC': 'Power Finance Corp',
        'BHARTI': 'Bharti Airtel',
        'HINDALCO': 'Hindalco Industries',
        'ITC': 'ITC Limited',
    }
    
    for ticker, count in sorted_tickers:
        name = stock_names.get(ticker, '')
        print(f"${ticker:<14} {count:<10} {name}")
    
    if not sorted_tickers:
        print("No tickers found in recent messages.")
        print("Channels may not be discussing stocks right now.")
        return
    
    # Step 2: Analyze most discussed ticker
    top_ticker = sorted_tickers[0][0]
    mention_count = sorted_tickers[0][1]
    
    print("\n" + "="*70)
    print(f"  STEP 2: ANALYZE ${top_ticker} ({stock_names.get(top_ticker, 'Unknown')})")
    print("="*70)
    print(f"\nFound {mention_count} mentions. Collecting messages...\n")
    
    # Get messages for this ticker
    ticker_messages = [msg for msg in messages if top_ticker in msg.get('detected_tickers', [])]
    
    # Show sample messages
    print(f"Sample messages about ${top_ticker}:\n")
    for i, msg in enumerate(ticker_messages[:3], 1):
        text = msg['text'][:150] + "..." if len(msg['text']) > 150 else msg['text']
        print(f"{i}. [{msg['channel']}]")
        print(f"   {text}")
        print()
    
    # Step 3: Sentiment analysis
    print("="*70)
    print(f"  STEP 3: SENTIMENT ANALYSIS FOR ${top_ticker}")
    print("="*70)
    print("\nRunning FinBERT sentiment analysis...")
    print("(This may take 30-60 seconds for model loading)\n")
    
    result = agent.analyze_ticker(
        ticker=top_ticker,
        platforms=['telegram'],
        min_posts=3
    )
    
    # Display results
    print("\n" + "="*70)
    print("  ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n📊 Ticker: ${result['ticker']}")
    print(f"📈 Sentiment Score: {result['sentiment_score']:.2f}")
    print(f"💭 Posts Analyzed: {result['total_posts']}")
    print(f"🎯 Confidence: {result['confidence']}")
    
    # Sentiment breakdown
    print(f"\nSentiment Distribution:")
    print(f"  Positive: {result['sentiment_counts']['positive']:>3} ({result['sentiment_counts']['positive']/result['total_posts']*100:.0f}%)")
    print(f"  Neutral:  {result['sentiment_counts']['neutral']:>3} ({result['sentiment_counts']['neutral']/result['total_posts']*100:.0f}%)")
    print(f"  Negative: {result['sentiment_counts']['negative']:>3} ({result['sentiment_counts']['negative']/result['total_posts']*100:.0f}%)")
    
    # Trading signal
    signal_emoji = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '🟡'
    }
    emoji = signal_emoji.get(result['signal'], '⚪')
    
    print(f"\n{emoji} TRADING SIGNAL: {result['signal']}")
    print(f"📝 Reasoning: {result['reasoning']}")
    
    # Platform breakdown
    print(f"\nPlatform Data:")
    for platform, data in result['platform_breakdown'].items():
        print(f"  {platform.capitalize()}: {data['posts']} posts, avg sentiment {data['avg_sentiment']:.2f}")
    
    # Database status
    print("\n" + "="*70)
    print("  DATABASE STATUS")
    print("="*70)
    
    history = db.get_ticker_history(top_ticker, limit=5)
    print(f"\n✓ Analysis saved to database")
    print(f"Total analyses for ${top_ticker}: {len(history)}")
    
    if len(history) > 1:
        print(f"\nRecent signals:")
        for record in history[:3]:
            timestamp = record[1][:19]  # Remove microseconds
            signal = record[2]
            sentiment = record[3]
            posts = record[4]
            emoji = signal_emoji.get(signal, '⚪')
            print(f"  {emoji} {timestamp} | {signal:>4} | Sentiment: {sentiment:>5.2f} | Posts: {posts}")
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70)
    print("\nYou can now:")
    print("1. Analyze other tickers from the list")
    print("2. Check historical data in sentiment_data.db")
    print("3. Add more platforms (Reddit, StockTwits, etc.)")
    print()

if __name__ == "__main__":
    main()
