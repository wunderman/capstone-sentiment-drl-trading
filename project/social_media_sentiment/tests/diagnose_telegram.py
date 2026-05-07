"""
Diagnose Telegram Channels - See what they actually post
"""

from ..collectors.telegram_collector import TelegramCollector
import requests
from bs4 import BeautifulSoup


def inspect_channel_content(channel_name):
    """See what a channel is actually posting."""
    url = f"https://t.me/s/{channel_name}"
    
    print(f"\n{'='*70}")
    print(f"  CHANNEL: {channel_name}")
    print(f"{'='*70}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get channel info
            title = soup.find('div', class_='tgme_channel_info_header_title')
            desc = soup.find('div', class_='tgme_channel_info_description')
            
            print(f"\nTitle: {title.get_text() if title else 'N/A'}")
            print(f"Description: {desc.get_text()[:200] if desc else 'N/A'}...")
            
            # Get recent messages
            messages = soup.find_all('div', class_='tgme_widget_message')
            print(f"\nTotal messages on page: {len(messages)}")
            
            if messages:
                print(f"\n{'─'*70}")
                print("LAST 3 MESSAGES:")
                print(f"{'─'*70}")
                
                for i, msg in enumerate(messages[:3], 1):
                    text_div = msg.find('div', class_='tgme_widget_message_text')
                    if text_div:
                        text = text_div.get_text()[:300]
                        print(f"\n{i}. {text}...")
                        
                        # Check for stock symbols
                        if '$' in text:
                            print("   → Contains $ symbols")
                        if any(ticker in text.upper() for ticker in ['AAPL', 'TSLA', 'MSFT', 'NVDA']):
                            print("   → Contains US tickers!")
                        if any(word in text.upper() for word in ['NSE', 'BSE', 'SENSEX', 'NIFTY']):
                            print("   → Indian market focus (NSE/BSE)")
            else:
                print("⚠️  No messages found on page")
        else:
            print(f"✗ HTTP {response.status_code}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("="*70)
    print("  TELEGRAM CHANNEL CONTENT INSPECTOR")
    print("  Finding out what these channels actually post")
    print("="*70)
    
    # Check each channel
    channels = [
        'marketfeed',
        'TheFinancialExpressOnline',
        'stock_market_addaa',
        'StockPro_Online',
        'stockmarketinfomania'
    ]
    
    for channel in channels:
        inspect_channel_content(channel)
    
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. If channels are India-focused (NSE/BSE), we need different channels")
    print("  2. If they use different ticker format, we need to adjust search logic")
    print("  3. If they're just inactive, we need more active channels")
    print("="*70 + "\n")
