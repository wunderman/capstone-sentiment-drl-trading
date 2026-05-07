"""
YouTube Collector
Fetches comments from YouTube videos for event-driven sentiment analysis.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .base_collector import BaseSocialMediaCollector


class YouTubeCollector(BaseSocialMediaCollector):
    """Collects comments from YouTube videos about stock tickers."""
    
    def __init__(self):
        """Initialize YouTube Data API client."""
        super().__init__('youtube')
        load_dotenv()
        
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            print("Warning: YOUTUBE_API_KEY not found. YouTube collector will not work.")
            self.youtube = None
        else:
            try:
                self.youtube = build('youtube', 'v3', developerKey=api_key)
            except Exception as e:
                print(f"Error initializing YouTube API: {e}")
                self.youtube = None
    
    def search_ticker(self,
                     ticker: str,
                     limit: int = 100,
                     max_videos: int = 5,
                     order: str = 'relevance') -> List[Dict]:
        """
        Search for comments on YouTube videos about a ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of comments total
            max_videos: Maximum number of videos to search
            order: Search order ('relevance', 'date', 'viewCount')
            
        Returns:
            List of comment dictionaries
        """
        if not self.youtube:
            print("YouTube API not initialized")
            return []
        
        all_comments = []
        
        # Step 1: Find relevant videos
        video_ids = self._search_videos(ticker, max_videos, order)
        
        if not video_ids:
            print(f"No YouTube videos found for {ticker}")
            return []
        
        # Step 2: Get comments from each video
        comments_per_video = limit // len(video_ids)
        
        for video_id in video_ids:
            try:
                comments = self._get_video_comments(video_id, ticker, comments_per_video)
                all_comments.extend(comments)
                
                if len(all_comments) >= limit:
                    break
                    
            except Exception as e:
                print(f"Error getting comments for video {video_id}: {e}")
                continue
        
        return all_comments[:limit]
    
    def _search_videos(self, ticker: str, max_results: int, order: str) -> List[str]:
        """Search for relevant videos about the ticker."""
        video_ids = []
        
        # Build search queries
        queries = [
            f'{ticker} stock analysis',
            f'{ticker} earnings',
            f'{ticker} stock news',
        ]
        
        try:
            for query in queries:
                request = self.youtube.search().list(
                    part='id',
                    q=query,
                    type='video',
                    order=order,
                    maxResults=max_results // len(queries),
                    relevanceLanguage='en'
                )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    if item['id']['kind'] == 'youtube#video':
                        video_ids.append(item['id']['videoId'])
                
                if len(video_ids) >= max_results:
                    break
        
        except HttpError as e:
            print(f"YouTube API error during search: {e}")
        
        return video_ids[:max_results]
    
    def _get_video_comments(self, video_id: str, ticker: str, limit: int) -> List[Dict]:
        """Get comments from a specific video."""
        comments = []
        
        try:
            # Get video info first
            video_request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            video_response = video_request.execute()
            
            video_info = None
            if video_response.get('items'):
                video_info = video_response['items'][0]
            
            # Get comments
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, limit),  # API max is 100
                order='relevance',  # Get most relevant comments
                textFormat='plainText'
            )
            
            response = request.execute()
            
            for item in response.get('items', []):
                try:
                    comment_data = item['snippet']['topLevelComment']['snippet']
                    
                    # Only include if ticker is mentioned
                    comment_text = comment_data['textDisplay']
                    if ticker.upper() in comment_text.upper() or f'${ticker.upper()}' in comment_text:
                        post_data = self._extract_post_data(comment_data, video_id, video_info)
                        comments.append(post_data)
                        
                except Exception as e:
                    continue
            
        except HttpError as e:
            # Comments might be disabled
            if 'commentsDisabled' in str(e):
                pass
            else:
                print(f"Error fetching comments: {e}")
        
        return comments
    
    def _extract_post_data(self, comment: Dict, video_id: str, video_info: Optional[Dict]) -> Dict:
        """Extract data from YouTube comment."""
        # Comment text
        text = comment.get('textDisplay', '')
        
        # Author info
        author = comment.get('authorDisplayName', 'unknown')
        author_channel_id = comment.get('authorChannelId', {}).get('value', '')
        
        # Timestamp
        published_at = comment.get('publishedAt', '')
        try:
            created_utc = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            created_utc = created_utc.replace(tzinfo=None)
        except:
            created_utc = datetime.now()
        
        # Engagement
        likes = comment.get('likeCount', 0)
        
        # Video context
        video_title = ''
        if video_info:
            video_title = video_info.get('snippet', {}).get('title', '')
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(
            likes=likes,
            multiplier=0.5
        )
        
        # Calculate post age
        post_age_hours = self._calculate_post_age_hours(created_utc)
        
        # Check for links
        has_links = 'http' in text or 'www.' in text
        
        return self._standardize_post({
            'id': f"{video_id}_{author_channel_id}_{int(created_utc.timestamp())}",
            'video_id': video_id,
            'video_title': video_title,
            'text': text,
            'full_text': text,
            'author': author,
            'created_utc': created_utc,
            'likes': likes,
            'account_age_days': 365,  # Unknown
            'author_followers': 0,  # Not available in comment API
            'engagement_score': engagement_score,
            'has_links': has_links,
            'is_verified': False,  # Not available in basic API
            'post_age_hours': post_age_hours,
            'url': f"https://www.youtube.com/watch?v={video_id}",
        })
    
    def get_video_info(self, video_id: str) -> Dict:
        """
        Get information about a YouTube video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video information
        """
        if not self.youtube:
            return {}
        
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            
            response = request.execute()
            
            if response.get('items'):
                video = response['items'][0]
                snippet = video.get('snippet', {})
                stats = video.get('statistics', {})
                
                return {
                    'video_id': video_id,
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'channel': snippet.get('channelTitle', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'view_count': int(stats.get('viewCount', 0)),
                    'like_count': int(stats.get('likeCount', 0)),
                    'comment_count': int(stats.get('commentCount', 0)),
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                }
        
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}
