"""
Test Improved Telegram Collector - Ticker Discovery
Scrapes all messages and shows what tickers are being discussed.
"""

from ..collectors.telegram_collector import TelegramCollector
from collections import Counter


def test_ticker_discovery():
    """Scrape messages and discover what tickers are being discussed."""
    print("="*70)
    print("  TELEGRAM TICKER DISCOVERY")
    print("  Scraping ALL messages to see what's being discussed")
    print("="*70)
    
    collector = TelegramCollector()
    
    print("\nScraping 100 recent messages from all channels...")
    print("(This will take 30-60 seconds)\n")
    
    # Scrape ALL messages without filtering
    all_messages = collector.scrape_all_messages(limit=100, max_retries=2)
    
    print(f"\n✓ Collected {len(all_messages)} total messages")
    
    if not all_messages:
        print("\n⚠️  No messages collected - all channels may have failed")
        return
    
    # Analyze what tickers were found
    all_tickers = []
    messages_with_tickers = 0
    
    for msg in all_messages:
        tickers = msg.get('detected_tickers', [])
        if tickers:
            all_tickers.extend(tickers)
            messages_with_tickers += 1
    
    print(f"Messages with tickers: {messages_with_tickers}/{len(all_messages)}")
    
    if all_tickers:
        print(f"\n{'='*70}")
        print("  DISCOVERED TICKERS")
        print(f"{'='*70}\n")
        
        # Count ticker frequency
        ticker_counts = Counter(all_tickers)
        
        print("Top 20 most mentioned tickers:\n")
        for ticker, count in ticker_counts.most_common(20):
            print(f"  ${ticker:6} - {count:3} mentions")
        
        # Show sample messages for top ticker
        if ticker_counts:
            top_ticker = ticker_counts.most_common(1)[0][0]
            print(f"\n{'='*70}")
            print(f"  SAMPLE MESSAGES FOR ${top_ticker}")
            print(f"{'='*70}\n")
            
            samples = [m for m in all_messages if top_ticker in m.get('detected_tickers', [])][:3]
            for i, msg in enumerate(samples, 1):
                print(f"{i}. [{msg.get('channel', 'unknown')}]")
                print(f"   {msg.get('text', '')[:200]}...")
                print(f"   Tickers: {', '.join(msg.get('detected_tickers', []))}")
                print()
    else:
        print("\n⚠️  No tickers detected in any messages")
        print("\nPossible reasons:")
        print("  - Channels post general news without ticker symbols")
        print("  - Messages use company names instead of tickers")
        print("  - Ticker format is different (e.g., not $SYMBOL)")
        
        print(f"\n{'='*70}")
        print("  SAMPLE MESSAGES (to understand format)")
        print(f"{'='*70}\n")
        
        for i, msg in enumerate(all_messages[:5], 1):
            print(f"{i}. [{msg.get('channel', 'unknown')}]")
            print(f"   {msg.get('text', '')[:250]}...")
            print()


def test_specific_ticker_search():
    """Test searching for a specific ticker using new method."""
    print("\n" + "="*70)
    print("  TESTING SPECIFIC TICKER SEARCH")
    print("="*70)
    
    collector = TelegramCollector()
    
    # First discover what tickers exist
    print("\n1. Discovering what tickers are being discussed...")
    all_messages = collector.scrape_all_messages(limit=50, max_retries=1)
    
    all_tickers = []
    for msg in all_messages:
        all_tickers.extend(msg.get('detected_tickers', []))
    
    if all_tickers:
        ticker_counts = Counter(all_tickers)
        top_ticker = ticker_counts.most_common(1)[0][0]
        
        print(f"   Most discussed: ${top_ticker} ({ticker_counts[top_ticker]} mentions)")
        
        print(f"\n2. Now searching specifically for ${top_ticker}...")
        specific_messages = collector.search_ticker(top_ticker, limit=5)
        
        print(f"   Found {len(specific_messages)} messages about ${top_ticker}")
        
        if specific_messages:
            print(f"\n   Sample:")
            msg = specific_messages[0]
            print(f"   Channel: {msg.get('channel')}")
            print(f"   Text: {msg.get('text', '')[:200]}...")
            print(f"   Tickers: {msg.get('detected_tickers')}")
    else:
        print("   No tickers found - trying with US tickers AAPL, TSLA...")
        for ticker in ['AAPL', 'TSLA', 'MSFT']:
            msgs = collector.search_ticker(ticker, limit=3)
            print(f"   ${ticker}: {len(msgs)} messages")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TELEGRAM TICKER DISCOVERY TEST")
    print("  New approach: Scrape everything, detect tickers automatically")
    print("="*70)
    
    test_ticker_discovery()
    test_specific_ticker_search()
    
    print("\n" + "="*70)
    print("  TEST COMPLETE")
    print("="*70 + "\n")
