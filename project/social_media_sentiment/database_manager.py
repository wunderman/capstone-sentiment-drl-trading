"""
Database Manager for Stock Sentiment Analysis
Handles SQLite storage of analysis runs, posts, and sentiment data.
"""

import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
import json
import os


class DatabaseManager:
    """Manages SQLite database for storing sentiment analysis data."""
    
    def __init__(self, db_path: str = "sentiment_data.db"):
        """
        Initialize database connection and create tables if needed.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Analysis runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                trade_signal TEXT,
                signal_strength REAL,
                sentiment_score REAL,
                confidence REAL,
                reliability REAL,
                total_posts INTEGER,
                filtered_posts INTEGER,
                filter_rate REAL,
                recommendation TEXT,
                platform_breakdown TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Posts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_run_id INTEGER NOT NULL,
                post_id TEXT,
                platform TEXT NOT NULL,
                ticker TEXT NOT NULL,
                text TEXT,
                author TEXT,
                created_utc TEXT,
                engagement_score REAL,
                account_age_days INTEGER,
                author_karma INTEGER,
                author_followers INTEGER,
                is_verified BOOLEAN,
                has_links BOOLEAN,
                post_age_hours REAL,
                
                -- Sentiment analysis results
                sentiment_score REAL,
                sentiment_label TEXT,
                sentiment_confidence REAL,
                
                -- Filter results
                filter_passed BOOLEAN,
                filter_confidence REAL,
                trading_intent_score INTEGER,
                quality_score INTEGER,
                account_credibility_score INTEGER,
                
                -- Platform-specific fields (stored as JSON)
                platform_metadata TEXT,
                
                FOREIGN KEY (analysis_run_id) REFERENCES analysis_runs (id)
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_ticker 
            ON analysis_runs(ticker)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp 
            ON analysis_runs(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_posts_ticker 
            ON posts(ticker)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_posts_platform 
            ON posts(platform)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_posts_run_id 
            ON analysis_runs(id)
        """)
        
        self.conn.commit()
    
    def save_analysis_run(self, results: Dict) -> int:
        """
        Save an analysis run and all associated posts to the database.
        
        Args:
            results: Results dictionary from StockSentimentAgent.analyze_ticker()
            
        Returns:
            The ID of the saved analysis run
        """
        cursor = self.conn.cursor()
        
        # Extract trade signal data
        trade_signal = results.get('trade_signal', {})
        
        # Insert analysis run
        cursor.execute("""
            INSERT INTO analysis_runs (
                ticker, timestamp, trade_signal, signal_strength,
                sentiment_score, confidence, reliability,
                total_posts, filtered_posts, filter_rate,
                recommendation, platform_breakdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            results.get('ticker'),
            results.get('timestamp'),
            trade_signal.get('action'),
            trade_signal.get('signal_strength'),
            trade_signal.get('sentiment_score'),
            trade_signal.get('confidence'),
            trade_signal.get('reliability'),
            results.get('total_posts_collected'),
            results.get('posts_passed_filter'),
            results.get('filter_rate'),
            trade_signal.get('recommendation'),
            json.dumps(results.get('platform_breakdown', {}))
        ))
        
        run_id = cursor.lastrowid
        
        # Save all posts
        raw_data = results.get('raw_data', {})
        filtered_posts = raw_data.get('filtered_posts', [])
        sentiments = raw_data.get('sentiments', [])
        
        # Create sentiment lookup by index (sentiments are parallel to filtered_posts)
        for i, post in enumerate(filtered_posts):
            post_id = post.get('id')
            sentiment = sentiments[i] if i < len(sentiments) else {}
            filter_results = post.get('filter_results', {})
            
            # Extract platform-specific metadata
            platform_metadata = {
                'subreddit': post.get('subreddit'),
                'video_id': post.get('video_id'),
                'video_title': post.get('video_title'),
                'sentiment_label': post.get('sentiment_label'),
                'channel': post.get('channel'),
                'url': post.get('url'),
            }
            # Remove None values
            platform_metadata = {k: v for k, v in platform_metadata.items() if v is not None}
            
            cursor.execute("""
                INSERT INTO posts (
                    analysis_run_id, post_id, platform, ticker, text,
                    author, created_utc, engagement_score, account_age_days,
                    author_karma, author_followers, is_verified, has_links,
                    post_age_hours, sentiment_score, sentiment_label,
                    sentiment_confidence, filter_passed, filter_confidence,
                    trading_intent_score, quality_score, account_credibility_score,
                    platform_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                post_id,
                post.get('platform'),
                results.get('ticker'),
                post.get('text') or post.get('full_text'),
                post.get('author'),
                post.get('created_utc'),
                post.get('engagement_score'),
                post.get('account_age_days'),
                post.get('author_karma'),
                post.get('author_followers'),
                post.get('is_verified', False),
                post.get('has_links', False),
                post.get('post_age_hours'),
                sentiment.get('sentiment_score'),
                sentiment.get('sentiment_label'),
                sentiment.get('confidence'),
                filter_results.get('passes_filter'),
                filter_results.get('confidence_percentage'),
                filter_results.get('scores', {}).get('trading_intent', 0),
                filter_results.get('scores', {}).get('quality', 0),
                filter_results.get('scores', {}).get('account_credibility', 0),
                json.dumps(platform_metadata) if platform_metadata else None
            ))
        
        self.conn.commit()
        print(f"✓ Saved analysis run #{run_id} to database ({len(filtered_posts)} posts)")
        
        return run_id
    
    def get_ticker_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """
        Get historical analysis runs for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of runs to return
            
        Returns:
            List of analysis run dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM analysis_runs 
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (ticker, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_run_details(self, run_id: int) -> Dict:
        """
        Get detailed information about a specific analysis run.
        
        Args:
            run_id: Analysis run ID
            
        Returns:
            Dictionary with run details and all posts
        """
        cursor = self.conn.cursor()
        
        # Get run info
        cursor.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,))
        run = cursor.fetchone()
        
        if not run:
            return {}
        
        # Get all posts for this run
        cursor.execute("""
            SELECT * FROM posts 
            WHERE analysis_run_id = ?
            ORDER BY sentiment_score DESC
        """, (run_id,))
        posts = [dict(row) for row in cursor.fetchall()]
        
        return {
            'run': dict(run),
            'posts': posts,
            'post_count': len(posts)
        }
    
    def get_platform_performance(self, ticker: Optional[str] = None, days: int = 30) -> Dict:
        """
        Analyze which platforms provide the most valuable data.
        
        Args:
            ticker: Optional ticker to filter by
            days: Number of days to look back
            
        Returns:
            Dictionary with platform statistics
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT 
                platform,
                COUNT(*) as post_count,
                AVG(engagement_score) as avg_engagement,
                AVG(filter_confidence) as avg_filter_confidence,
                AVG(ABS(sentiment_score)) as avg_sentiment_strength,
                COUNT(CASE WHEN filter_passed = 1 THEN 1 END) as passed_count
            FROM posts
            WHERE datetime(created_utc) > datetime('now', '-' || ? || ' days')
        """
        
        params = [days]
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        query += " GROUP BY platform ORDER BY post_count DESC"
        
        cursor.execute(query, params)
        
        results = {}
        for row in cursor.fetchall():
            results[row['platform']] = {
                'post_count': row['post_count'],
                'avg_engagement': round(row['avg_engagement'], 2),
                'avg_filter_confidence': round(row['avg_filter_confidence'], 2),
                'avg_sentiment_strength': round(row['avg_sentiment_strength'], 3),
                'filter_pass_rate': round(row['passed_count'] / row['post_count'] * 100, 2)
            }
        
        return results
    
    def get_signal_history(self, ticker: str, days: int = 30) -> List[Dict]:
        """
        Get trade signal history for backtesting.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of signals with timestamps
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                timestamp, trade_signal, sentiment_score, 
                confidence, signal_strength, total_posts, filtered_posts
            FROM analysis_runs
            WHERE ticker = ?
            AND datetime(timestamp) > datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        """, (ticker, days))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_database_stats(self) -> Dict:
        """Get overall database statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total runs
        cursor.execute("SELECT COUNT(*) as count FROM analysis_runs")
        stats['total_runs'] = cursor.fetchone()['count']
        
        # Total posts
        cursor.execute("SELECT COUNT(*) as count FROM posts")
        stats['total_posts'] = cursor.fetchone()['count']
        
        # Unique tickers
        cursor.execute("SELECT COUNT(DISTINCT ticker) as count FROM analysis_runs")
        stats['unique_tickers'] = cursor.fetchone()['count']
        
        # Date range
        cursor.execute("SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM analysis_runs")
        row = cursor.fetchone()
        stats['first_analysis'] = row['first']
        stats['last_analysis'] = row['last']
        
        # Signal breakdown
        cursor.execute("""
            SELECT trade_signal, COUNT(*) as count 
            FROM analysis_runs 
            GROUP BY trade_signal
        """)
        stats['signal_breakdown'] = {row['trade_signal']: row['count'] for row in cursor.fetchall()}
        
        # Platform breakdown
        cursor.execute("""
            SELECT platform, COUNT(*) as count 
            FROM posts 
            GROUP BY platform
        """)
        stats['platform_breakdown'] = {row['platform']: row['count'] for row in cursor.fetchall()}
        
        # Database file size
        stats['db_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
        
        return stats
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
