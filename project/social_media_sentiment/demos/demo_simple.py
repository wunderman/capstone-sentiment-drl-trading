"""
Simple Demo - Test Without API Keys
Demonstrates the system with mock data (no real API calls).
"""

from ..stock_sentiment_agent import StockSentimentAgent
from ..database_manager import DatabaseManager


def demo_with_stocktwits():
    """
    Demo using StockTwits (works without API key!)
    StockTwits public data doesn't require authentication.
    """
    print("\n" + "="*70)
    print("  SIMPLE DEMO - StockTwits Only (No Auth Required)")
    print("="*70)
    
    try:
        # Initialize with only StockTwits (no auth needed)
        agent = StockSentimentAgent(
            use_reddit=False,        # Requires API key
            use_stocktwits=True,     # Works without auth!
            use_telegram=True,       # Works without auth!
            use_youtube=False,       # Requires API key
            use_bluesky=False,       # Optional
            use_twitter=False,       # Expensive
            auto_save=True
        )
        
        # Analyze a popular stock
        print("\nAnalyzing $AAPL sentiment from StockTwits + Telegram...\n")
        
        results = agent.analyze_ticker(
            ticker='AAPL',
            stocktwits_limit=50,
            telegram_limit=30
        )
        
        # Check if we got results
        if not results or results.get('error'):
            print(f"\n⚠️  {results.get('error', 'No data collected')}")
            print("\nPossible issues:")
            print("  - Network connectivity problems")
            print("  - StockTwits API rate limiting")
            print("  - Telegram channels unavailable")
            print("\nTry again in a few moments or set up Reddit API for more reliable data.")
            return False
        
        # Show results
        print("\n" + "="*70)
        print("  RESULTS SUMMARY")
        print("="*70)
        print(f"\nTicker: ${results['ticker']}")
        print(f"Trade Signal: {results['trade_signal']['action']}")
        print(f"Confidence: {results['trade_signal']['confidence']:.1f}%")
        print(f"Sentiment Score: {results['trade_signal']['sentiment_score']:.3f}")
        print(f"Posts Analyzed: {results['posts_passed_filter']}/{results['total_posts_collected']}")
        
        if 'db_run_id' in results:
            print(f"\n✓ Saved to database as run #{results['db_run_id']}")
            print("\nQuery your data:")
            print("  python query_database.py")
        
        return True
        
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("\nThis demo requires internet connection to scrape StockTwits.")
        print("StockTwits public data works without API keys!")
        return False


def show_database_stats():
    """Show what's stored in the database."""
    print("\n" + "="*70)
    print("  DATABASE CONTENTS")
    print("="*70)
    
    try:
        db = DatabaseManager()
        stats = db.get_database_stats()
        
        if stats['total_runs'] == 0:
            print("\nNo data yet. Run an analysis first!")
        else:
            print(f"\nTotal Analyses: {stats['total_runs']}")
            print(f"Total Posts Stored: {stats['total_posts']}")
            print(f"Unique Tickers: {stats['unique_tickers']}")
            print(f"Database Size: {stats['db_size_mb']} MB")
            
            if stats.get('signal_breakdown'):
                print(f"\nSignals:")
                for signal, count in stats['signal_breakdown'].items():
                    print(f"  {signal}: {count}")
        
        db.close()
        
    except Exception as e:
        print(f"Database not yet created: {e}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  STOCK SENTIMENT ANALYSIS - SIMPLE DEMO")
    print("  Testing with StockTwits (No API Key Needed)")
    print("="*70)
    
    # Run demo
    success = demo_with_stocktwits()
    
    if success:
        # Show database stats
        show_database_stats()
        
        print("\n" + "="*70)
        print("  NEXT STEPS")
        print("="*70)
        print("\n1. View your data:")
        print("   python query_database.py")
        print("\n2. Add more platforms:")
        print("   - Edit .env with Reddit credentials")
        print("   - See collectors/README.md for setup guides")
        print("\n3. Run full examples:")
        print("   python example_database_usage.py")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("  TROUBLESHOOTING")
        print("="*70)
        print("\n1. Check internet connection")
        print("2. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n3. For full functionality, set up API keys:")
        print("   - Copy .env.example to .env")
        print("   - Add Reddit credentials (easiest to start)")
        print("   - See collectors/README.md for guides")
        print("="*70 + "\n")
