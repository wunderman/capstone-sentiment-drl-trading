"""
Relevance Filter Module
Implements the relevance policy for filtering stock trading sentiment from social media.
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import math


class RelevanceFilter:
    """Filters social media posts based on relevance to stock trading."""
    
    # Trading intent keywords
    TRADING_INTENT = [
        r'\b(buy|buying|bought|sell|selling|sold|short|shorting|shorted)\b',
        r'\b(hold|holding|hodl|long|going long|going short)\b',
        r'\b(my position|entered at|exit|target price|stop loss|take profit)\b',
        r'\b(looking to buy|planning to sell|waiting for|averaging down)\b',
        r'\b(accumulated|accumulating|dumped|dumping|loading|unloading)\b',
    ]
    
    # Price analysis keywords
    PRICE_ANALYSIS = [
        r'\$\d+\.?\d*\s*(PT|price target|target)',
        r'\b(support|resistance|breakout|breakdown|consolidation)\b',
        r'\b(RSI|MACD|moving average|MA|EMA|SMA|fibonacci|bollinger)\b',
        r'\b(bullish|bearish|bull|bear|neutral)\b',
        r'\b(moon|rocket|crash|rally|dump|pump|squeeze)\b',
        r'\b(trend|momentum|volume|reversal|pattern)\b',
        r'\b(overbought|oversold|overvalued|undervalued)\b',
    ]
    
    # Fundamental analysis keywords
    FUNDAMENTAL = [
        r'\b(earnings|EPS|revenue|guidance|beat|miss)\b',
        r'\b(quarterly|annual report|10-K|10-Q|SEC filing)\b',
        r'\b(P/E ratio|price to earnings|market cap|valuation)\b',
        r'\b(merger|acquisition|M&A|partnership|contract)\b',
        r'\b(FDA approval|product launch|new product|patent)\b',
        r'\b(layoffs|hiring|expansion|restructuring)\b',
        r'\b(dividend|buyback|stock split|dilution)\b',
    ]
    
    # Due diligence indicators
    DUE_DILIGENCE = [
        r'\b(DD|due diligence|analysis|research|investigation)\b',
        r'\b(thesis|my take|opinion|thoughts on)\b',
        r'\b(fundamentals|technicals|catalysts)\b',
    ]
    
    # Ticker mention (weak positive signal for relevance)
    TICKER_MENTION = [
        r'\$[A-Z]{1,5}\b',  # Cashtag format like $TSLA $AAPL
    ]
    
    # Spam/low quality indicators
    SPAM_PATTERNS = [
        r'\b(get rich quick|guaranteed returns|100% profit|can\'t lose)\b',
        r'\b(click here|sign up|join now|limited time)\b',
        r'\b(pump and dump|coordinated|everyone buy)\b',
        r'(.)\1{4,}',  # Repeated characters
    ]
    
    # Off-topic patterns
    OFF_TOPIC = [
        r'\b(job posting|hiring|career|resume|interview)\b',
        r'\b(customer service|support ticket|complaint|refund)\b',
        r'\b(how do I|how to use|setup|installation)\b',
    ]
    
    def __init__(self, 
                 min_quality_score: int = 40,
                 min_account_score: int = 20,
                 min_relevance_score: int = 1):
        """
        Initialize the relevance filter.
        
        Args:
            min_quality_score: Minimum quality score (0-100)
            min_account_score: Minimum account credibility score (0-100)
            min_relevance_score: Minimum relevance score to pass filter
        """
        self.min_relevance_score = min_relevance_score
        self.min_quality_score = min_quality_score
        self.min_account_score = min_account_score
        
    def check_relevance(self, text: str) -> Tuple[bool, int, Dict[str, int]]:
        """
        Check if text is relevant to stock trading.
        
        Returns:
            Tuple of (is_relevant, score, breakdown)
        """
        text_lower = text.lower()
        breakdown = {
            'trading_intent': 0,
            'price_analysis': 0,
            'fundamental': 0,
            'due_diligence': 0,
            'ticker_mention': 0,
            'spam_penalty': 0,
            'off_topic_penalty': 0,
        }
        
        # Count trading intent matches
        for pattern in self.TRADING_INTENT:
            if re.search(pattern, text_lower, re.IGNORECASE):
                breakdown['trading_intent'] += 1
                
        # Count price analysis matches
        for pattern in self.PRICE_ANALYSIS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                breakdown['price_analysis'] += 1
                
        # Count fundamental analysis matches
        for pattern in self.FUNDAMENTAL:
            if re.search(pattern, text_lower, re.IGNORECASE):
                breakdown['fundamental'] += 1
                
        # Count DD indicators
        for pattern in self.DUE_DILIGENCE:
            if re.search(pattern, text_lower, re.IGNORECASE):
                breakdown['due_diligence'] += 1
                
        # Check for ticker mentions (cashtag format)
        for pattern in self.TICKER_MENTION:
            if re.search(pattern, text):  # case-sensitive: $TSLA not $tsla
                breakdown['ticker_mention'] += 1
                
        # Check for spam
        for pattern in self.SPAM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                breakdown['spam_penalty'] += 1
                
        # Check for off-topic
        for pattern in self.OFF_TOPIC:
            if re.search(pattern, text_lower, re.IGNORECASE):
                breakdown['off_topic_penalty'] += 1
        
        # Calculate total relevance score
        score = (
            breakdown['trading_intent'] * 3 +  # High weight
            breakdown['price_analysis'] * 2 +
            breakdown['fundamental'] * 2 +
            breakdown['due_diligence'] * 2 +
            breakdown['ticker_mention'] * 1 -  # Weak positive signal
            breakdown['spam_penalty'] * 5 -
            breakdown['off_topic_penalty'] * 3
        )
        
        is_relevant = (
            score >= self.min_relevance_score and
            breakdown['spam_penalty'] == 0 and
            breakdown['off_topic_penalty'] == 0
        )
        
        return is_relevant, max(0, score), breakdown
    
    def score_account_quality(self, 
                             account_age_days: int,
                             karma_or_followers: int,
                             is_verified: bool = False) -> int:
        """
        Score account credibility (0-100).
        
        Args:
            account_age_days: Age of account in days
            karma_or_followers: Reddit karma or Twitter followers
            is_verified: Twitter verified status
            
        Returns:
            Account quality score (0-100)
        """
        score = 0
        
        # Account age scoring
        if account_age_days < 30:
            score += 0
        elif account_age_days < 180:
            score += 20
        elif account_age_days < 365:
            score += 40
        elif account_age_days < 730:
            score += 60
        else:
            score += 80
            
        # Karma/followers scoring
        if karma_or_followers < 100:
            score += 0
        elif karma_or_followers < 1000:
            score += 10
        elif karma_or_followers < 10000:
            score += 20
        else:
            score += 30
            
        # Verified bonus
        if is_verified:
            score += 20
            
        return min(100, score)
    
    def score_content_quality(self,
                             text: str,
                             engagement_score: int = 0,
                             has_links: bool = False,
                             subreddit: str = None) -> int:
        """
        Score content quality (0-100).
        
        Args:
            text: Post content
            engagement_score: Upvotes/likes (normalized 0-100)
            has_links: Whether post contains source links
            subreddit: Subreddit name (if Reddit)
            
        Returns:
            Content quality score (0-100)
        """
        score = 0
        words = text.split()
        word_count = len(words)
        
        # Length scoring
        if word_count < 10:
            score += 10
        elif word_count < 50:
            score += 40
        elif word_count < 200:
            score += 80
        else:
            score += 60
            
        # Engagement (already normalized 0-100)
        score += min(35, engagement_score * 0.35)
        
        # Has source links
        if has_links:
            score += 15
            
        # Proper ticker formatting
        if re.search(r'\$[A-Z]{1,5}\b', text):
            score += 10
            
        # Penalties
        # All caps
        if text.isupper() and len(text) > 20:
            score -= 20
            
        # Excessive emojis
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', text))
        if emoji_count > 5:
            score -= 15
            
        # Pump language
        pump_words = ['moon', 'rocket', '100x', 'lambo']
        pump_count = sum(1 for word in pump_words if word in text.lower())
        score -= pump_count * 10
        
        # Subreddit adjustment
        if subreddit:
            if subreddit.lower() == 'wallstreetbets':
                score -= 10  # Higher noise
            elif subreddit.lower() in ['securityanalysis', 'valueinvesting']:
                score += 10  # Higher signal
                
        return max(0, min(100, score))
    
    def calculate_confidence(self,
                           relevance_score: int,
                           quality_score: int,
                           account_score: int) -> Tuple[str, int]:
        """
        Calculate overall confidence level.
        
        Returns:
            Tuple of (confidence_level, confidence_percentage)
        """
        # Weighted average
        confidence = (
            relevance_score * 0.4 +
            quality_score * 0.35 +
            account_score * 0.25
        )
        
        if confidence >= 90:
            level = "HIGH"
        elif confidence >= 60:
            level = "MEDIUM"
        elif confidence >= 40:
            level = "LOW"
        else:
            level = "REJECT"
            
        return level, int(confidence)
    
    def apply_time_decay(self, 
                        confidence: int,
                        post_age_hours: float) -> int:
        """
        Apply time decay to confidence score.
        
        Args:
            confidence: Original confidence score
            post_age_hours: Age of post in hours
            
        Returns:
            Decayed confidence score
        """
        days = post_age_hours / 24
        decay_factor = math.exp(-0.5 * days)
        return int(confidence * decay_factor)
    
    def filter_post(self,
                   text: str,
                   account_age_days: int,
                   karma_or_followers: int,
                   engagement_score: int = 0,
                   is_verified: bool = False,
                   has_links: bool = False,
                   subreddit: str = None,
                   post_age_hours: float = 0) -> Dict:
        """
        Complete filter pipeline for a social media post.
        
        Returns:
            Dictionary with filter results and scores
        """
        # Check relevance
        is_relevant, relevance_score, relevance_breakdown = self.check_relevance(text)
        
        # Score account quality
        account_score = self.score_account_quality(
            account_age_days, karma_or_followers, is_verified
        )
        
        # Score content quality
        quality_score = self.score_content_quality(
            text, engagement_score, has_links, subreddit
        )
        
        # Calculate confidence
        confidence_level, confidence_pct = self.calculate_confidence(
            relevance_score, quality_score, account_score
        )
        
        # Apply time decay if specified
        final_confidence = confidence_pct
        if post_age_hours > 0:
            final_confidence = self.apply_time_decay(confidence_pct, post_age_hours)
        
        # Determine if post passes filter
        passes_filter = (
            is_relevant and
            quality_score >= self.min_quality_score and
            account_score >= self.min_account_score and
            confidence_level != "REJECT"
        )
        
        return {
            'passes_filter': passes_filter,
            'relevance_score': relevance_score,
            'quality_score': quality_score,
            'account_score': account_score,
            'confidence_level': confidence_level,
            'confidence_percentage': final_confidence,
            'relevance_breakdown': relevance_breakdown,
            'text': text,
        }
