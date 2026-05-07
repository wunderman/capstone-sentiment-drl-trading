"""
Reddit Collector
Fetches posts from Reddit about specific stock tickers.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import praw
from dotenv import load_dotenv

from .base_collector import BaseSocialMediaCollector


class RedditCollector(BaseSocialMediaCollector):
    """Collects posts from Reddit about specific stock tickers."""
    
    def __init__(self):
        """Initialize Reddit API client."""
        super().__init__('reddit')
        load_dotenv()
        
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'StockSentimentAgent/1.0')
        )
        
        # Popular finance subreddits
        self.finance_subreddits = [
            'stocks',
            'investing',
            'wallstreetbets',
            'StockMarket',
            'options',
            'thetagang',
            'securityanalysis',
            'valueinvesting',
            'dividends',
        ]
    
    def search_ticker(self, 
                     ticker: str, 
                     limit: int = 100,
                     time_filter: str = 'day',
                     subreddits: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for posts mentioning a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of posts to retrieve
            time_filter: Time filter ('hour', 'day', 'week', 'month')
            subreddits: List of subreddits to search (None = use defaults)
            
        Returns:
            List of post dictionaries
        """
        posts = []
        subreddits_to_search = subreddits or self.finance_subreddits
        
        # Search patterns for ticker
        search_queries = [
            f'${ticker}',  # $AAPL format
            ticker,        # AAPL
            f'{ticker} stock',
        ]
        
        for subreddit_name in subreddits_to_search:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for query in search_queries:
                    results = subreddit.search(
                        query,
                        time_filter=time_filter,
                        limit=limit // len(search_queries)
                    )
                    
                    for submission in results:
                        # Check if ticker is actually mentioned
                        full_text = f"{submission.title} {submission.selftext}"
                        if ticker.upper() in full_text.upper() or f'${ticker.upper()}' in full_text:
                            post_data = self._extract_post_data(submission, subreddit_name)
                            posts.append(post_data)
                            
            except Exception as e:
                print(f"Error searching r/{subreddit_name}: {e}")
                continue
        
        # Remove duplicates based on post ID
        unique_posts = {post['id']: post for post in posts}.values()
        return list(unique_posts)
    
    def _extract_post_data(self, submission, subreddit_name: str) -> Dict:
        """Extract relevant data from Reddit submission."""
        # Calculate account age
        account_created = datetime.fromtimestamp(submission.author.created_utc) if submission.author else datetime.now()
        account_age_days = (datetime.now() - account_created).days
        
        # Get karma
        author_karma = submission.author.link_karma + submission.author.comment_karma if submission.author else 0
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(
            score=submission.score,
            comments=submission.num_comments
        )
        
        # Check for links
        has_links = bool(submission.url and not submission.is_self) or 'http' in submission.selftext
        
        # Calculate post age
        post_age_hours = self._calculate_post_age_hours(
            datetime.fromtimestamp(submission.created_utc)
        )
        
        return self._standardize_post({
            'id': submission.id,
            'subreddit': subreddit_name,
            'title': submission.title,
            'text': submission.selftext,
            'full_text': f"{submission.title}\n{submission.selftext}",
            'author': str(submission.author) if submission.author else '[deleted]',
            'created_utc': datetime.fromtimestamp(submission.created_utc),
            'score': submission.score,
            'num_comments': submission.num_comments,
            'url': submission.url,
            'account_age_days': account_age_days,
            'author_karma': author_karma,
            'engagement_score': engagement_score,
            'has_links': has_links,
            'is_verified': False,
            'post_age_hours': post_age_hours,
        })
    
    def get_comments(self, post_id: str, limit: int = 50) -> List[Dict]:
        """
        Get comments from a specific post.
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:limit]:
                if comment.author:
                    account_created = datetime.fromtimestamp(comment.author.created_utc)
                    account_age_days = (datetime.now() - account_created).days
                    author_karma = comment.author.link_karma + comment.author.comment_karma
                    
                    engagement_score = self._calculate_engagement_score(score=comment.score)
                    
                    comments.append(self._standardize_post({
                        'id': comment.id,
                        'text': comment.body,
                        'full_text': comment.body,
                        'author': str(comment.author),
                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                        'score': comment.score,
                        'account_age_days': account_age_days,
                        'author_karma': author_karma,
                        'engagement_score': engagement_score,
                        'has_links': 'http' in comment.body,
                        'is_verified': False,
                    }))
        except Exception as e:
            print(f"Error fetching comments: {e}")
            
        return comments
