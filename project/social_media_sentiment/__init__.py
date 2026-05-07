"""
Social Media Sentiment Pipeline
Generates sentiment scores for stock trading from social media data.
"""

from .stock_sentiment_agent import StockSentimentAgent
from .sentiment_analyzer import SentimentAnalyzer
from .relevance_filter import RelevanceFilter
from .database_manager import DatabaseManager
from .dataset_collector import DatasetCollector
from .llm_ticker_extractor import OllamaTickerExtractor
from .llm_relevance_filter import LLMRelevanceFilter
from .collectors.apify_collector import ApifyCollector

__all__ = [
    'StockSentimentAgent',
    'SentimentAnalyzer',
    'RelevanceFilter',
    'DatabaseManager',
    'DatasetCollector',
    'OllamaTickerExtractor',
    'LLMRelevanceFilter',
    'ApifyCollector',
]

__version__ = "1.0.0"
