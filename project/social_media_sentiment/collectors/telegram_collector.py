"""
Telegram Collector
Scrapes messages from public Telegram channels via web preview.
"""

import requests
from typing import List, Dict, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import time
import re

from .base_collector import BaseSocialMediaCollector

# Try to import Ollama LLM extractor
try:
    from ..llm_ticker_extractor import OllamaTickerExtractor
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False


class TelegramCollector(BaseSocialMediaCollector):
    """Collects messages from public Telegram channels."""
    
    def __init__(self, use_llm: bool = True, llm_model: str = "llama3.2"):
        """
        Initialize Telegram collector.
        
        Args:
            use_llm: Use Ollama LLM for ticker extraction (smarter but slower)
            llm_model: Ollama model to use (llama3.2, mistral, gemma2, etc.)
        """
        super().__init__('telegram')
        
        # Initialize LLM extractor if available
        self.use_llm = False
        if use_llm and OLLAMA_AVAILABLE:
            try:
                self.llm_extractor = OllamaTickerExtractor(model=llm_model)
                self.use_llm = True
                print(f"    ✓ Using Ollama LLM ({llm_model}) for ticker extraction")
            except Exception as e:
                print(f"    ⚠️  Ollama not available: {e}")
                print("    ⚠️  Falling back to regex-based extraction")
        elif use_llm:
            print("    ⚠️  Ollama module not found, using regex extraction")
        
        # Stock-focused Telegram channels — verified working, US stocks
        self.finance_channels = [
            # Tier 1: High ticker density (earnings, stock-specific)
            'earningswhispers',              # 100+ $TICKER mentions per page
            'stockmarketnewsfeed',           # Yahoo Finance stock news reposts
            'earningscall',                  # Earnings transcripts, GS/NVDA/TSLA/WMT

            # Tier 2: Market news with ticker mentions
            'financialjuice',                # Real-time financial news ($INTC $GOOGL)
            'tradingview',                   # AAPL, AMZN, GOOGL, META, NVDA, TSLA
            'marketfeed',                    # Market news summaries

            # Tier 3: Broader finance/trading
            'investingcom',                  # Investing.com news
            'seekingalpha',                  # Seeking Alpha articles
            'cryptosignals',                 # Crypto + some stocks
            'swingtrading',                  # Swing trade setups
        ]

        # Company name → ticker mapping for DJIA stocks
        # Enables matching "Apple" or "Microsoft" in text, not just $AAPL
        self.company_to_ticker = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'amazon': 'AMZN',
            'nvidia': 'NVDA', 'tesla': 'TSLA', 'meta': 'META',
            'alphabet': 'GOOGL', 'google': 'GOOGL',
            'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'chase': 'JPM',
            'goldman sachs': 'GS', 'goldman': 'GS',
            'visa': 'V', 'mastercard': 'MA',
            'johnson & johnson': 'JNJ', 'johnson and johnson': 'JNJ',
            'walmart': 'WMT', 'procter & gamble': 'PG', 'procter and gamble': 'PG',
            'coca-cola': 'KO', 'coca cola': 'KO', 'coke': 'KO',
            'disney': 'DIS', 'walt disney': 'DIS',
            'nike': 'NKE', 'boeing': 'BA', 'caterpillar': 'CAT',
            'home depot': 'HD', 'honeywell': 'HON',
            'chevron': 'CVX', 'merck': 'MRK', 'amgen': 'AMGN',
            'cisco': 'CSCO', 'intel': 'INTC', 'ibm': 'IBM',
            'verizon': 'VZ', 'salesforce': 'CRM',
            'unitedhealth': 'UNH', 'united health': 'UNH',
            'american express': 'AXP', 'amex': 'AXP',
            'travelers': 'TRV', '3m': 'MMM', 'dow inc': 'DOW',
            'mcdonald': 'MCD', 'mcdonalds': 'MCD', "mcdonald's": 'MCD',
            # Expanded universe additions
            'netflix': 'NFLX', 'paypal': 'PYPL', 'uber': 'UBER',
            'coinbase': 'COIN', 'palantir': 'PLTR', 'shopify': 'SHOP',
            'snowflake': 'SNOW', 'crowdstrike': 'CRWD', 'palo alto': 'PANW',
            'datadog': 'DDOG', 'cloudflare': 'NET', 'rivian': 'RIVN',
            'lucid': 'LCID', 'airbnb': 'ABNB', 'block inc': 'SQ', 'square': 'SQ',
            'advanced micro': 'AMD', 'qualcomm': 'QCOM', 'texas instruments': 'TXN',
            'eli lilly': 'LLY', 'abbvie': 'ABBV', 'costco': 'COST',
            'accenture': 'ACN', 'adobe': 'ADBE', 'broadcom': 'AVGO',
            'oracle': 'ORCL', 'pepsico': 'PEP', 'pepsi': 'PEP',
            'thermo fisher': 'TMO', 'exxon': 'XOM', 'exxonmobil': 'XOM',
            'bank of america': 'BAC',
        }
        
        self.base_url = "https://t.me/s"  # Public channel preview URL
        self.verified_channels = set()  # Cache of working channels
        self.failed_channels = set()    # Cache of failed channels
    
    def search_ticker(self,
                     ticker: str,
                     limit: int = 100,
                     channels: Optional[List[str]] = None,
                     max_retries: int = 2) -> List[Dict]:
        """
        Search for messages mentioning a ticker in Telegram channels.
        NOW SMARTER: Scrapes all messages, extracts tickers, then filters.
        
        Args:
            ticker: Stock ticker symbol to filter by (e.g., 'AAPL')
            limit: Maximum number of messages to return
            channels: List of channel names (None = use defaults)
            max_retries: Number of retries for failed channels
            
        Returns:
            List of message dictionaries with detected tickers
        """
        all_messages = self.scrape_all_messages(
            limit=limit * 3,  # Get more messages to increase chance of matches
            channels=channels,
            max_retries=max_retries
        )
        
        # Filter messages that mention the ticker
        filtered = []
        for msg in all_messages:
            detected_tickers = msg.get('detected_tickers', [])
            if ticker.upper() in detected_tickers or ticker.upper() in msg.get('text', '').upper():
                filtered.append(msg)
        
        return filtered[:limit]
    
    def scrape_all_messages(self,
                           limit: int = 300,
                           channels: Optional[List[str]] = None,
                           max_retries: int = 2) -> List[Dict]:
        """
        Scrape ALL recent messages from channels, detecting any tickers mentioned.
        
        Args:
            limit: Maximum total messages to collect
            channels: List of channel names (None = use defaults)
            max_retries: Number of retries for failed channels
            
        Returns:
            List of all message dictionaries with detected_tickers field
        """
        messages = []
        channels_to_search = channels or self.finance_channels
        
        # Skip previously failed channels
        channels_to_search = [c for c in channels_to_search if c not in self.failed_channels]
        
        for channel in channels_to_search:
            # Try with retries
            for attempt in range(max_retries):
                try:
                    channel_messages = self._scrape_channel(channel, limit)
                    
                    if channel_messages:
                        messages.extend(channel_messages)
                        self.verified_channels.add(channel)  # Mark as working
                        break  # Success, move to next channel
                    elif attempt == max_retries - 1:
                        # No messages after all retries
                        break
                    else:
                        # Retry with backoff
                        time.sleep(2 ** attempt)
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Final attempt failed
                        print(f"  ⚠️  Channel {channel} failed after {max_retries} attempts: {e}")
                        self.failed_channels.add(channel)
                    else:
                        # Retry
                        time.sleep(2 ** attempt)
                        continue
            
            if len(messages) >= limit:
                break
            
            # Rate limiting between channels
            time.sleep(1)
        
        return messages[:limit]
    
    def _scrape_channel(self, channel: str, limit: int) -> List[Dict]:
        """Scrape recent messages from a Telegram channel with pagination."""
        messages = []
        max_pages = max(1, (limit // 20) + 1)  # ~20 msgs per page
        before_id = None

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        try:
            for page in range(max_pages):
                url = f"{self.base_url}/{channel}"
                if before_id:
                    url += f"?before={before_id}"

                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                message_divs = soup.find_all('div', class_='tgme_widget_message')

                if not message_divs:
                    break

                page_count = 0
                for msg_div in message_divs:
                    try:
                        post_data = self._extract_post_data(msg_div, channel)
                        if post_data:
                            post_data['detected_tickers'] = self._extract_tickers(post_data['text'])
                            messages.append(post_data)
                            page_count += 1
                            if len(messages) >= limit:
                                return messages[:limit]
                    except Exception:
                        continue

                # Get the oldest message ID on this page for pagination
                oldest_post = message_divs[0].get('data-post', '')
                msg_id = oldest_post.split('/')[-1] if '/' in oldest_post else oldest_post
                if msg_id and msg_id.isdigit():
                    before_id = int(msg_id)
                else:
                    break  # Can't paginate further

                if page_count == 0:
                    break

                time.sleep(0.5)  # Rate limit between pages

        except requests.exceptions.Timeout:
            raise Exception(f"Timeout accessing {channel} - network slow or blocked")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Connection failed to {channel} - check internet")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception(f"Channel {channel} not found")
            elif e.response.status_code == 403:
                raise Exception(f"Channel {channel} access forbidden")
            else:
                raise Exception(f"HTTP error {e.response.status_code}")
        except Exception as e:
            raise Exception(f"Unknown error scraping {channel}: {str(e)}")

        return messages
    
    def _extract_post_data(self, msg_div, channel: str) -> Optional[Dict]:
        """Extract data from Telegram message div."""
        try:
            # Extract message ID
            msg_id = msg_div.get('data-post', '').split('/')[-1]
            if not msg_id:
                msg_id = str(hash(str(msg_div)[:100]))
            
            # Extract text
            text_div = msg_div.find('div', class_='tgme_widget_message_text')
            text = text_div.get_text(strip=True) if text_div else ''
            
            if not text:
                return None
            
            # Extract author (channel name or forwarded from)
            author_elem = msg_div.find('a', class_='tgme_widget_message_owner_name')
            author = author_elem.get_text(strip=True) if author_elem else channel
            
            # Extract timestamp
            time_elem = msg_div.find('time', class_='datetime')
            if time_elem and time_elem.get('datetime'):
                try:
                    created_utc = datetime.fromisoformat(time_elem['datetime'].replace('Z', '+00:00'))
                    created_utc = created_utc.replace(tzinfo=None)
                except:
                    created_utc = datetime.now()
            else:
                created_utc = datetime.now()
            
            # Extract views (engagement metric for Telegram)
            views_elem = msg_div.find('span', class_='tgme_widget_message_views')
            views = 0
            if views_elem:
                views_text = views_elem.get_text(strip=True)
                # Parse "1.2K" or "500" format
                views = self._parse_number(views_text)
            
            # Check for links
            has_links = bool(msg_div.find('a', class_='tgme_widget_message_link_preview'))
            
            # Calculate engagement score (based on views)
            engagement_score = self._calculate_engagement_score(
                likes=views,
                multiplier=0.01  # Telegram views are high, scale down
            )
            
            # Calculate post age
            post_age_hours = self._calculate_post_age_hours(created_utc)
            
            # Build message URL
            msg_url = f"https://t.me/{channel}/{msg_id}"
            
            return self._standardize_post({
                'id': f"{channel}_{msg_id}",
                'channel': channel,
                'text': text,
                'full_text': text,
                'author': author,
                'created_utc': created_utc,
                'likes': views,  # Using views as 'likes'
                'account_age_days': 365,  # Unknown for Telegram
                'author_followers': 0,  # Not available
                'engagement_score': engagement_score,
                'has_links': has_links,
                'is_verified': False,  # Telegram doesn't have verification
                'post_age_hours': post_age_hours,
                'url': msg_url,
            })
            
        except Exception as e:
            return None
    
    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock ticker symbols from text.
        Uses LLM if available, otherwise falls back to regex.
        
        Args:
            text: Message text
            
        Returns:
            List of detected ticker symbols (uppercase)
        """
        if self.use_llm:
            try:
                return self.llm_extractor.extract_tickers(text)
            except Exception:
                pass  # Fall through to regex

        # Regex extraction + company name matching
        tickers = set(self._extract_tickers_regex(text))

        # Also match company names (case-insensitive)
        text_lower = text.lower()
        for name, ticker in self.company_to_ticker.items():
            if name in text_lower:
                tickers.add(ticker)

        return sorted(list(tickers))
    
    def _extract_tickers_regex(self, text: str) -> List[str]:
        """
        Extract stock ticker symbols using regex patterns.
        Fallback method when LLM is not available.
        
        Args:
            text: Message text
            
        Returns:
            List of detected ticker symbols (uppercase)
        """
        tickers = set()
        
        # Expanded exclusion list for common words
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'NOT', 'BUT', 'CAN', 'ALL', 'NEW', 'GET', 'NOW', 'TOP', 
            'BUY', 'SELL', 'TO', 'FROM', 'AT', 'IN', 'OF', 'ON', 'BY', 'UP', 'DOWN', 'OUT',
            'US', 'UK', 'EU', 'AS', 'OR', 'IF', 'AN', 'BE', 'WE', 'IT', 'MY', 'NO', 'SO',
            'DO', 'GO', 'HE', 'SHE', 'YOU', 'ME', 'HIS', 'HER', 'ITS', 'OUR', 'THEIR',
            'ABOVE', 'BELOW', 'OVER', 'UNDER', 'WITH', 'INTO', 'ONTO',
            'GOVT', 'SEBI', 'RBI', 'SEC', 'FED', 'INDEX', 'FUND', 'ETF', 'GAINS', 'LOSS',
            'HIGH', 'LOW', 'NEAR', 'HELP', 'THEM', 'THOSE', 'THESE', 'THAT', 'THIS',
            'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'DOES', 'DID',
            'HAVE', 'HAS', 'HAD', 'BEEN', 'BEING', 'WAS', 'WERE', 'IS', 'AM',
            'BILLION', 'MILLION', 'TRILLION', 'POUND', 'DOLLAR', 'EURO', 'RUPEE', 'YEN',
            'LENDERS', 'AGREE', 'PACKAGE', 'PACKAGES', 'YEAR', 'YEARS', 'MONTH', 'MONTHS',
            'DAY', 'DAYS', 'WEEK', 'WEEKS', 'TODAY', 'YESTERDAY', 'TOMORROW'
        }
        
        # Pattern 1: $SYMBOL format (most reliable when filtered)
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text)
        valid_dollar = [t for t in dollar_tickers if t not in common_words and len(t) >= 2]
        tickers.update(valid_dollar)
        
        # Pattern 2: Known stock tickers only (high confidence)
        # Only match known major stocks to avoid false positives
        known_tickers = {
            # DJIA 30 Components
            'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX',
            'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM',
            'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'TRV',
            'UNH', 'V', 'VZ', 'WMT',
            # Other major US stocks
            'TSLA', 'META', 'GOOGL', 'GOOG', 'NFLX', 'AMD', 'AVGO', 'ORCL',
            'QCOM', 'TXN',
            # Major ETFs
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO',
        }
        # Find known tickers in text (case insensitive)
        for known in known_tickers:
            if re.search(rf'\b{known}\b', text, re.IGNORECASE):
                tickers.add(known)
        
        # Pattern 3: Indian stocks (NSE/BSE format) - very reliable
        # Examples: RELIANCE.NS, TCS.BO, INFY.BSE
        indian_tickers = re.findall(r'\b([A-Z]+)\.(NS|BO|BSE|NSE)\b', text)
        tickers.update([t[0] for t in indian_tickers])
        
        # Pattern 4: Crypto (optional, for channels that mix stocks/crypto)
        crypto_tickers = re.findall(r'\b(BTC|ETH|SOL|DOGE|XRP|ADA|MATIC|BNB|USDT|USDC)\b', text)
        tickers.update(crypto_tickers)
        
        return sorted(list(tickers))
    
    def _parse_number(self, text: str) -> int:
        """Parse numbers like '1.2K', '500', '1M' into integers."""
        text = text.strip().upper()
        
        try:
            if 'K' in text:
                return int(float(text.replace('K', '')) * 1000)
            elif 'M' in text:
                return int(float(text.replace('M', '')) * 1000000)
            else:
                # Remove non-numeric characters
                clean = re.sub(r'[^\d]', '', text)
                return int(clean) if clean else 0
        except:
            return 0
    
    def verify_channel(self, channel: str) -> bool:
        """
        Verify if a Telegram channel is accessible.
        
        Args:
            channel: Channel name
            
        Returns:
            True if channel is accessible, False otherwise
        """
        if channel in self.verified_channels:
            return True
        if channel in self.failed_channels:
            return False
            
        url = f"{self.base_url}/{channel}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            success = response.status_code == 200
            
            if success:
                self.verified_channels.add(channel)
            else:
                self.failed_channels.add(channel)
                
            return success
        except:
            self.failed_channels.add(channel)
            return False
    
    def get_channel_info(self, channel: str) -> Dict:
        """
        Get information about a Telegram channel.
        
        Args:
            channel: Channel name
            
        Returns:
            Dictionary with channel info
        """
        url = f"{self.base_url}/{channel}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract channel info
            title_elem = soup.find('div', class_='tgme_channel_info_header_title')
            title = title_elem.get_text(strip=True) if title_elem else channel
            
            desc_elem = soup.find('div', class_='tgme_channel_info_description')
            description = desc_elem.get_text(strip=True) if desc_elem else ''
            
            # Extract subscriber count
            counter_elem = soup.find('div', class_='tgme_channel_info_counter')
            subscribers_text = counter_elem.get_text(strip=True) if counter_elem else '0'
            subscribers = self._parse_number(subscribers_text.split()[0])
            
            return {
                'channel': channel,
                'title': title,
                'description': description,
                'subscribers': subscribers,
                'url': url,
            }
        
        except Exception as e:
            print(f"Error getting channel info: {e}")
            return {'channel': channel, 'error': str(e)}
