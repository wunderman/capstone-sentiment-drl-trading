"""
Bluesky Collector
Fetches posts from Bluesky social network using AT Protocol.
"""

import os
import requests
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import time

from .base_collector import BaseSocialMediaCollector


class BlueskyCollector(BaseSocialMediaCollector):
    """Collects posts from Bluesky social network."""
    
    def __init__(self):
        """Initialize Bluesky collector."""
        super().__init__('bluesky')
        load_dotenv()
        
        self.api_base = "https://public.api.bsky.app/xrpc"
        
        # Optional: Bluesky credentials for authenticated access
        self.handle = os.getenv('BLUESKY_HANDLE')
        self.password = os.getenv('BLUESKY_PASSWORD')
        self.session_token = None
        
        # Try to authenticate if credentials provided
        if self.handle and self.password:
            try:
                self._authenticate()
            except Exception as e:
                print(f"Bluesky authentication failed: {e}")
    
    def _authenticate(self):
        """Authenticate with Bluesky (optional, for better access)."""
        url = f"{self.api_base}/com.atproto.server.createSession"
        
        data = {
            "identifier": self.handle,
            "password": self.password
        }
        
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        self.session_token = result.get('accessJwt')
    
    def search_ticker(self,
                     ticker: str,
                     limit: int = 100,
                     method: str = 'search') -> List[Dict]:
        """
        Search for posts mentioning a ticker on Bluesky.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of posts
            method: 'search' (if available) or 'feed' (public timeline)
            
        Returns:
            List of post dictionaries
        """
        if method == 'search' and self._search_available():
            return self._search_posts(ticker, limit)
        else:
            # Fallback: get public feed and filter
            return self._filter_public_feed(ticker, limit)
    
    def _search_available(self) -> bool:
        """Check if search API is available."""
        # As of 2024, Bluesky search API may be limited
        # This is a placeholder - check actual API status
        return False
    
    def _search_posts(self, ticker: str, limit: int) -> List[Dict]:
        """Search posts using Bluesky search API (if available)."""
        posts = []
        
        # This is a placeholder for when search API becomes available
        url = f"{self.api_base}/app.bsky.feed.searchPosts"
        
        params = {
            'q': f'${ticker}',
            'limit': min(100, limit)
        }
        
        headers = {}
        if self.session_token:
            headers['Authorization'] = f'Bearer {self.session_token}'
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data.get('posts', []):
                    post_data = self._extract_post_data(post)
                    posts.append(post_data)
        
        except Exception as e:
            print(f"Bluesky search error: {e}")
        
        return posts[:limit]
    
    def _filter_public_feed(self, ticker: str, limit: int) -> List[Dict]:
        """Get public feed and filter for ticker mentions."""
        posts = []
        
        # Get public timeline
        url = f"{self.api_base}/app.bsky.feed.getTimeline"
        
        headers = {}
        if self.session_token:
            headers['Authorization'] = f'Bearer {self.session_token}'
        
        try:
            params = {'limit': 100}
            cursor = None
            
            while len(posts) < limit:
                if cursor:
                    params['cursor'] = cursor
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                feed = data.get('feed', [])
                
                if not feed:
                    break
                
                for item in feed:
                    post = item.get('post', {})
                    record = post.get('record', {})
                    text = record.get('text', '')
                    
                    # Check if ticker is mentioned
                    if ticker.upper() in text.upper() or f'${ticker.upper()}' in text:
                        post_data = self._extract_post_data(post)
                        posts.append(post_data)
                        
                        if len(posts) >= limit:
                            break
                
                # Get next cursor for pagination
                cursor = data.get('cursor')
                if not cursor:
                    break
                
                time.sleep(0.5)  # Rate limiting
        
        except Exception as e:
            print(f"Error fetching Bluesky feed: {e}")
        
        return posts[:limit]
    
    def _extract_post_data(self, post: Dict) -> Dict:
        """Extract data from Bluesky post."""
        # Post content
        record = post.get('record', {})
        text = record.get('text', '')
        
        # Author info
        author_data = post.get('author', {})
        author = author_data.get('handle', 'unknown')
        display_name = author_data.get('displayName', author)
        
        # Timestamp
        created_at = record.get('createdAt', '')
        try:
            created_utc = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            created_utc = created_utc.replace(tzinfo=None)
        except:
            created_utc = datetime.now()
        
        # Engagement metrics
        like_count = post.get('likeCount', 0)
        repost_count = post.get('repostCount', 0)
        reply_count = post.get('replyCount', 0)
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(
            likes=like_count,
            retweets=repost_count,
            comments=reply_count
        )
        
        # Calculate post age
        post_age_hours = self._calculate_post_age_hours(created_utc)
        
        # Check for links
        has_links = bool(record.get('facets')) or 'http' in text
        
        # Post URI
        post_uri = post.get('uri', '')
        post_id = post_uri.split('/')[-1] if post_uri else str(hash(text[:50]))
        
        return self._standardize_post({
            'id': post_id,
            'text': text,
            'full_text': text,
            'author': author,
            'created_utc': created_utc,
            'likes': like_count,
            'retweets': repost_count,
            'replies': reply_count,
            'account_age_days': 180,  # Unknown, estimate
            'author_followers': author_data.get('followersCount', 0),
            'engagement_score': engagement_score,
            'has_links': has_links,
            'is_verified': False,  # Bluesky doesn't have traditional verification
            'post_age_hours': post_age_hours,
            'url': f"https://bsky.app/profile/{author}/post/{post_id}",
        })
    
    def get_profile(self, handle: str) -> Dict:
        """
        Get profile information for a Bluesky user.
        
        Args:
            handle: User handle (e.g., 'user.bsky.social')
            
        Returns:
            Dictionary with profile info
        """
        url = f"{self.api_base}/app.bsky.actor.getProfile"
        
        params = {'actor': handle}
        
        headers = {}
        if self.session_token:
            headers['Authorization'] = f'Bearer {self.session_token}'
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            print(f"Error getting profile: {e}")
            return {}
