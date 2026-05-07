"""
StockTwits Collector
Fetches messages from StockTwits with ticker-native mapping.
Supports both Official API and web scraping as backup.
"""

import os
import requests
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time

from .base_collector import BaseSocialMediaCollector


class StockTwitsCollector(BaseSocialMediaCollector):
    """Collects messages from StockTwits about specific stock tickers."""
    
    def __init__(self, use_api: bool = True):
        """
        Initialize StockTwits collector.
        
        Args:
            use_api: Use official API (True) or web scraping (False)
        """
        super().__init__('stocktwits')
        load_dotenv()
        
        self.use_api = use_api
        self.api_base_url = "https://api.stocktwits.com/api/2"
        self.web_base_url = "https://stocktwits.com"
        
        # Optional: StockTwits API token (not required for public data)
        self.api_token = os.getenv('STOCKTWITS_API_TOKEN')
        
    def search_ticker(self,
                     ticker: str,
                     limit: int = 100,
                     filter_by: str = 'all') -> List[Dict]:
        """
        Search for messages about a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of messages
            filter_by: Filter type ('all', 'top', 'charts', 'videos')
            
        Returns:
            List of message dictionaries
        """
        if self.use_api:
            try:
                return self._search_via_api(ticker, limit, filter_by)
            except Exception as e:
                print(f"API search failed: {e}, falling back to web scraping")
                return self._search_via_web(ticker, limit)
        else:
            return self._search_via_web(ticker, limit)
    
    def _search_via_api(self, ticker: str, limit: int, filter_by: str) -> List[Dict]:
        """Search using official StockTwits API."""
        messages = []
        
        # API endpoint for symbol stream
        url = f"{self.api_base_url}/streams/symbol/{ticker}.json"
        
        params = {
            'filter': filter_by,
            'limit': min(30, limit)  # API max is 30 per request
        }
        
        headers = {}
        if self.api_token:
            headers['Authorization'] = f'Bearer {self.api_token}'
        
        try:
            # Make multiple requests if needed to reach limit
            max_id = None
            remaining = limit
            
            while remaining > 0:
                if max_id:
                    params['max'] = max_id
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if 'messages' not in data or not data['messages']:
                    break
                
                for msg in data['messages']:
                    if len(messages) >= limit:
                        break
                    
                    post_data = self._extract_post_data(msg, source='api')
                    messages.append(post_data)
                    max_id = msg['id']
                
                remaining = limit - len(messages)
                
                # Check if we've reached the end
                if len(data['messages']) < params['limit']:
                    break
                
                # Rate limiting
                time.sleep(0.5)
        
        except Exception as e:
            print(f"Error fetching from StockTwits API: {e}")
        
        return messages
    
    def _search_via_web(self, ticker: str, limit: int) -> List[Dict]:
        """Search using web scraping (backup method)."""
        messages = []
        
        url = f"{self.web_base_url}/symbol/{ticker}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find message containers
            # StockTwits uses dynamic loading, so we may get limited results
            message_divs = soup.find_all('div', class_='st_', limit=limit)
            
            for idx, msg_div in enumerate(message_divs):
                try:
                    # Extract message data from HTML
                    post_data = self._extract_web_data(msg_div, ticker, idx)
                    if post_data:
                        messages.append(post_data)
                except Exception as e:
                    continue
            
            # If we couldn't parse the page, return empty list
            if not messages:
                print(f"Warning: Web scraping found no messages. StockTwits may have changed their HTML structure.")
        
        except Exception as e:
            print(f"Error scraping StockTwits: {e}")
        
        return messages
    
    def _extract_post_data(self, msg: Dict, source: str = 'api') -> Dict:
        """Extract data from API response message."""
        # User data
        user = msg.get('user', {})
        username = user.get('username', 'unknown')
        followers = user.get('followers', 0)
        
        # Calculate account age (if join_date available)
        join_date = user.get('join_date')
        if join_date:
            try:
                created = datetime.strptime(join_date, '%Y-%m-%d')
                account_age_days = (datetime.now() - created).days
            except:
                account_age_days = 365  # Default estimate
        else:
            account_age_days = 365
        
        # Message data
        message_id = str(msg.get('id', ''))
        body = msg.get('body', '')
        created_at = msg.get('created_at', '')
        
        # Parse timestamp
        try:
            if 'T' in created_at:
                created_utc = datetime.strptime(created_at, '%Y-%m-%dT%H:%M:%SZ')
            else:
                created_utc = datetime.now()
        except:
            created_utc = datetime.now()
        
        # Sentiment (StockTwits users can tag bullish/bearish!)
        entities = msg.get('entities', {})
        sentiment = entities.get('sentiment')
        sentiment_label = None
        if sentiment:
            if sentiment.get('basic') == 'Bullish':
                sentiment_label = 'bullish'
            elif sentiment.get('basic') == 'Bearish':
                sentiment_label = 'bearish'
        
        # Engagement (likes on StockTwits)
        likes = msg.get('likes', {}).get('total', 0)
        
        # Links
        has_links = bool(msg.get('entities', {}).get('links'))
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(
            likes=likes,
            multiplier=0.5  # StockTwits engagement is generally lower
        )
        
        # Calculate post age
        post_age_hours = self._calculate_post_age_hours(created_utc)
        
        return self._standardize_post({
            'id': message_id,
            'text': body,
            'full_text': body,
            'author': username,
            'created_utc': created_utc,
            'likes': likes,
            'account_age_days': account_age_days,
            'author_followers': followers,
            'engagement_score': engagement_score,
            'has_links': has_links,
            'is_verified': user.get('official', False),
            'post_age_hours': post_age_hours,
            'sentiment_label': sentiment_label,  # StockTwits-specific
            'url': f"https://stocktwits.com/{username}/message/{message_id}",
        })
    
    def _extract_web_data(self, msg_div, ticker: str, idx: int) -> Optional[Dict]:
        """Extract data from web scraping (simplified version)."""
        try:
            # This is a simplified parser - actual StockTwits HTML may vary
            # You may need to inspect the page and adjust selectors
            
            text_elem = msg_div.find('div', class_='body')
            text = text_elem.get_text(strip=True) if text_elem else ''
            
            author_elem = msg_div.find('a', class_='username')
            author = author_elem.get_text(strip=True) if author_elem else 'unknown'
            
            # Simplified data (web scraping has limitations)
            return self._standardize_post({
                'id': f'web_{ticker}_{idx}',
                'text': text,
                'full_text': text,
                'author': author,
                'created_utc': datetime.now(),
                'likes': 0,  # Hard to extract from web
                'account_age_days': 365,  # Unknown
                'author_followers': 0,
                'engagement_score': 0,
                'has_links': 'http' in text,
                'is_verified': False,
                'post_age_hours': 0,
                'url': f"https://stocktwits.com/symbol/{ticker}",
            })
        except:
            return None
    
    def get_trending(self, limit: int = 30) -> List[Dict]:
        """
        Get trending symbols/messages from StockTwits.
        
        Args:
            limit: Maximum number of trending items
            
        Returns:
            List of trending data
        """
        url = f"{self.api_base_url}/trending/symbols.json"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('symbols', [])[:limit]
        
        except Exception as e:
            print(f"Error fetching trending: {e}")
            return []
