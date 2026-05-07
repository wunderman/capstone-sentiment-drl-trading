"""
Base Collector Interface
Abstract base class for all social media collectors to ensure consistent output format.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime


class BaseSocialMediaCollector(ABC):
    """Abstract base class for social media data collectors."""
    
    def __init__(self, platform_name: str):
        """
        Initialize the base collector.
        
        Args:
            platform_name: Name of the platform (e.g., 'reddit', 'stocktwits')
        """
        self.platform_name = platform_name
    
    @abstractmethod
    def search_ticker(self, ticker: str, limit: int = 100, **kwargs) -> List[Dict]:
        """
        Search for posts mentioning a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of posts to retrieve
            **kwargs: Platform-specific parameters
            
        Returns:
            List of standardized post dictionaries
        """
        pass
    
    @abstractmethod
    def _extract_post_data(self, raw_data, **kwargs) -> Dict:
        """
        Extract relevant data from platform-specific post object.
        
        Args:
            raw_data: Platform-specific post/message object
            **kwargs: Additional context
            
        Returns:
            Dictionary with extracted data
        """
        pass
    
    def _standardize_post(self, post_data: Dict) -> Dict:
        """
        Ensure post data has all required fields with consistent naming.
        
        Args:
            post_data: Extracted post data
            
        Returns:
            Standardized post dictionary
        """
        standardized = {
            'id': post_data.get('id', ''),
            'platform': self.platform_name,
            'text': post_data.get('text', ''),
            'full_text': post_data.get('full_text', post_data.get('text', '')),
            'author': post_data.get('author', 'unknown'),
            'created_utc': post_data.get('created_utc', datetime.now()),
            'account_age_days': post_data.get('account_age_days', 0),
            'author_karma': post_data.get('author_karma', 0),
            'author_followers': post_data.get('author_followers', 0),
            'engagement_score': post_data.get('engagement_score', 0),
            'has_links': post_data.get('has_links', False),
            'is_verified': post_data.get('is_verified', False),
            'post_age_hours': post_data.get('post_age_hours', 0),
            
            # Platform-specific fields (optional)
            'url': post_data.get('url'),
            'sentiment_label': post_data.get('sentiment_label'),  # For StockTwits
            'subreddit': post_data.get('subreddit'),  # For Reddit
            'channel': post_data.get('channel'),  # For Telegram
            'video_id': post_data.get('video_id'),  # For YouTube
            
            # Engagement metrics
            'likes': post_data.get('likes', 0),
            'score': post_data.get('score', 0),
            'retweets': post_data.get('retweets', 0),
            'replies': post_data.get('replies', 0),
            'num_comments': post_data.get('num_comments', 0),
        }
        
        return standardized
    
    def _calculate_engagement_score(self, 
                                    likes: int = 0,
                                    comments: int = 0,
                                    retweets: int = 0,
                                    score: int = 0,
                                    multiplier: float = 1.0) -> int:
        """
        Calculate normalized engagement score (0-100).
        
        Args:
            likes: Number of likes/upvotes
            comments: Number of comments/replies
            retweets: Number of shares/retweets
            score: Platform score (Reddit upvotes)
            multiplier: Platform-specific multiplier
            
        Returns:
            Engagement score (0-100)
        """
        # Weighted sum of engagement metrics
        engagement = (
            likes * 1.0 +
            comments * 1.5 +  # Comments indicate more engagement
            retweets * 2.0 +  # Shares are high-value
            score * 1.0
        ) * multiplier
        
        # Normalize to 0-100 scale
        # Using logarithmic scaling for better distribution
        import math
        if engagement == 0:
            return 0
        
        # Log scale with base adjustment
        normalized = min(100, (math.log1p(engagement) / math.log1p(1000)) * 100)
        
        return int(normalized)
    
    def _calculate_post_age_hours(self, created_utc: datetime) -> float:
        """
        Calculate how many hours ago the post was created.
        
        Args:
            created_utc: Post creation timestamp
            
        Returns:
            Age in hours
        """
        if not created_utc:
            return 0.0
        
        # Ensure timezone-naive datetime for comparison
        if created_utc.tzinfo is not None:
            created_utc = created_utc.replace(tzinfo=None)
        
        now = datetime.now()
        age_delta = now - created_utc
        
        return age_delta.total_seconds() / 3600
