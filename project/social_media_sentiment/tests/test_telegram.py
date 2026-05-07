"""
Test Telegram Collector - Improved Version
Tests the updated Telegram collector with better channels and error handling.
"""

from ..collectors.telegram_collector import TelegramCollector
import json


def test_channel_verification():
    """Test channel verification before scraping."""
    print("="*70)
    print("  TESTING CHANNEL VERIFICATION")
    print("="*70)
    
    collector = TelegramCollector()
    
    print(f"\nTesting {len(collector.finance_channels)} channels:")
    print(f"Channels: {', '.join(collector.finance_channels[:5])}...\n")
    
    working_channels = []
    failed_channels = []
    
    for channel in collector.finance_channels[:5]:  # Test first 5
        print(f"  Testing {channel}...", end=" ")
        try:
            is_accessible = collector.verify_channel(channel)
            if is_accessible:
                print("✓ Accessible")
                working_channels.append(channel)
            else:
                print("✗ Failed")
                failed_channels.append(channel)
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_channels.append(channel)
    
    print(f"\n✓ Working: {len(working_channels)}")
    print(f"✗ Failed: {len(failed_channels)}")
    
    return working_channels


def test_scraping(ticker='AAPL', limit=5):
    """Test actual scraping with improved logic."""
    print("\n" + "="*70)
    print(f"  TESTING SCRAPING FOR ${ticker}")
    print("="*70)
    
    collector = TelegramCollector()
    
    print(f"\nScraping up to {limit} messages from Telegram channels...")
    print("(This may take 30-60 seconds with new timeout settings)\n")
    
    try:
        messages = collector.search_ticker(ticker, limit=limit, max_retries=2)
        
        print(f"\n✓ Collected {len(messages)} messages")
        
        if messages:
            print(f"\n{'='*70}")
            print("  SAMPLE MESSAGES")
            print(f"{'='*70}\n")
            
            for i, msg in enumerate(messages[:3], 1):
                print(f"{i}. [{msg.get('channel', 'unknown')}]")
                print(f"   Author: {msg.get('author', 'N/A')}")
                print(f"   Text: {msg.get('text', '')[:150]}...")
                print(f"   Views: {msg.get('likes', 0)}")
                print(f"   Posted: {msg.get('created_utc', 'N/A')}")
                print(f"   Engagement: {msg.get('engagement_score', 0):.1f}")
                print()
        else:
            print("\n⚠️  No messages found")
            print("\nPossible reasons:")
            print("  - Channels don't have recent posts about this ticker")
            print("  - Network connectivity issues")
            print("  - All channels failed (see error messages above)")
            
            if collector.failed_channels:
                print(f"\nFailed channels ({len(collector.failed_channels)}):")
                for ch in collector.failed_channels:
                    print(f"  - {ch}")
        
        return messages
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_different_tickers():
    """Test with multiple tickers to see which gets more results."""
    print("\n" + "="*70)
    print("  TESTING MULTIPLE TICKERS")
    print("="*70)
    
    collector = TelegramCollector()
    
    tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT']
    
    results = {}
    for ticker in tickers:
        print(f"\nTrying ${ticker}...", end=" ")
        try:
            messages = collector.search_ticker(ticker, limit=3, max_retries=1)
            results[ticker] = len(messages)
            print(f"{len(messages)} messages")
        except Exception as e:
            results[ticker] = 0
            print(f"0 messages (error)")
    
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    for ticker, count in results.items():
        print(f"  ${ticker}: {count} messages")
    
    best = max(results, key=results.get)
    print(f"\n✓ Best ticker: ${best} with {results[best]} messages")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TELEGRAM COLLECTOR - IMPROVED VERSION TEST")
    print("  Enhanced with: Better channels, retry logic, 30s timeout")
    print("="*70)
    
    # Test 1: Verify channels
    working = test_channel_verification()
    
    if working:
        # Test 2: Scrape for AAPL
        test_scraping('AAPL', limit=5)
        
        # Test 3: Try different tickers
        test_different_tickers()
    else:
        print("\n⚠️  No working channels found!")
        print("\nThis could mean:")
        print("  1. Network connectivity issues")
        print("  2. Telegram is blocking access")
        print("  3. Channels have changed/don't exist")
        print("\nTry:")
        print("  - Check internet connection")
        print("  - Visit https://t.me/s/marketfeed in browser")
        print("  - Use VPN if Telegram is blocked")
    
    print("\n" + "="*70)
    print("  TEST COMPLETE")
    print("="*70 + "\n")
