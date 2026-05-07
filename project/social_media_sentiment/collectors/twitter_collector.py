"""
Twitter/X Collector
Fetches tweets about specific stock tickers.
"""

import os
from typing import List, Dict
from datetime import datetime, timedelta
import tweepy
from dotenv import load_dotenv

from .base_collector import BaseSocialMediaCollector


class TwitterCollector(BaseSocialMediaCollector):
    """Collects tweets about specific stock tickers."""
    
    def __init__(self):
        """Initialize Twitter API client."""
        super().__init__('twitter')
        load_dotenv()
        
        # Twitter API v2 client
        self.client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_SECRET'),
            wait_on_rate_limit=True
        )
    
    def search_ticker(self,
                     ticker: str,
                     limit: int = 100,
                     hours_back: int = 24) -> List[Dict]:
        """
        Search for tweets mentioning a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of tweets
            hours_back: How many hours back to search
            
        Returns:
            List of tweet dictionaries
        """
        tweets = []
        
        # Build search query (use cashtag for financial context)
        query = f'${ticker} -is:retweet lang:en'
        
        # Calculate start time
        start_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(100, limit),
                start_time=start_time,
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'entities'],
                user_fields=['created_at', 'public_metrics', 'verified'],
                expansions=['author_id']
            )
            
            if not response.data:
                return tweets
            
            # Create user lookup
            users = {user.id: user for user in response.includes.get('users', [])}
            
            for tweet in response.data:
                author = users.get(tweet.author_id)
                tweet_data = self._extract_post_data(tweet, author)
                tweets.append(tweet_data)
                
        except Exception as e:
            print(f"Error searching Twitter: {e}")
            
        return tweets
    
    def _extract_post_data(self, tweet, author) -> Dict:
        """Extract relevant data from tweet."""
        # Calculate account age
        if author and author.created_at:
            account_age_days = (datetime.now(author.created_at.tzinfo) - author.created_at).days
        else:
            account_age_days = 0
        
        # Get follower count
        followers = author.public_metrics.get('followers_count', 0) if author else 0
        
        # Get metrics
        metrics = tweet.public_metrics
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        replies = metrics.get('reply_count', 0)
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(
            likes=likes,
            retweets=retweets,
            comments=replies
        )
        
        # Check for links
        has_links = bool(tweet.entities and 'urls' in tweet.entities)
        
        # Calculate post age
        post_age_hours = self._calculate_post_age_hours(tweet.created_at.replace(tzinfo=None))
        
        return self._standardize_post({
            'id': str(tweet.id),
            'text': tweet.text,
            'full_text': tweet.text,
            'author': author.username if author else 'unknown',
            'created_utc': tweet.created_at.replace(tzinfo=None),
            'likes': likes,
            'retweets': retweets,
            'replies': replies,
            'account_age_days': account_age_days,
            'author_followers': followers,
            'engagement_score': engagement_score,
            'has_links': has_links,
            'is_verified': author.verified if author else False,
            'post_age_hours': post_age_hours,
        })
