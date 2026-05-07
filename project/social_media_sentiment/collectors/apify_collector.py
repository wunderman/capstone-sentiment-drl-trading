"""
Apify Social Media Collector
Uses the Apify platform to collect posts from Twitter/X, Reddit, Telegram,
YouTube Comments, and StockTwits.
Replaces paid individual API subscriptions with a single Apify key.

Actors used:
  - Twitter/X:  apidojo/tweet-scraper        (scrapes tweets by search query)
  - Reddit:     trudax/reddit-scraper         (scrapes subreddits and search results)
  - Telegram:   tri_angle/telegram-scraper    (scrapes public channels)
  - YouTube:    streamers/youtube-comments-scraper  (scrapes video comments)
  - StockTwits: saswave/stocktwits-stock-ticker-news-scraper  (scrapes ticker stream)
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from apify_client import ApifyClient
from dotenv import load_dotenv

from .base_collector import BaseSocialMediaCollector


class ApifyCollector(BaseSocialMediaCollector):
    """
    Collects social media posts via Apify actors.
    Supports Twitter/X, Reddit, Telegram, YouTube Comments, and StockTwits
    through a single API key.
    """

    # ── Apify actor IDs ────────────────────────────────────────
    TWITTER_ACTOR = "apidojo/tweet-scraper"
    REDDIT_ACTOR = "trudax/reddit-scraper"
    TELEGRAM_ACTOR = "tri_angle/telegram-scraper"
    YOUTUBE_ACTOR = "streamers/youtube-comments-scraper"
    STOCKTWITS_ACTOR = "saswave/stocktwits-stock-ticker-news-scraper"

    # Financial subreddits to search
    FINANCE_SUBREDDITS = [
        "wallstreetbets", "stocks", "investing",
        "stockmarket", "options", "SecurityAnalysis",
    ]

    # Stock-focused Telegram channels (public)
    TELEGRAM_CHANNELS = [
        "stockmarketinfomania",
        "marketfeed",
        "stock_market_addaa",
        "equitymasterofficial",
        "stockalerts",
        "wallstreetbets",
    ]

    def __init__(self, api_key: str = None):
        """
        Initialise the Apify collector.

        Args:
            api_key: Apify API token. Falls back to APIFY_API_KEY env var.
        """
        super().__init__(platform_name="apify")

        # Load from .env if not provided directly
        load_dotenv()
        self.api_key = api_key or os.getenv("APIFY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Apify API key not found. Set APIFY_API_KEY in .env or pass api_key="
            )

        self.client = ApifyClient(self.api_key)
        print(f"  ✓ Apify client initialised")

    # ── Public interface (required by ABC) ─────────────────────
    def search_ticker(self, ticker: str, limit: int = 100, **kwargs) -> List[Dict]:
        """
        Collect posts about *ticker* from all Apify-backed sources.

        Keyword args forwarded to individual scrapers:
            twitter_limit:  max tweets  (default: limit)
            reddit_limit:   max reddit posts  (default: limit)
            telegram_limit: max telegram messages  (default: limit)
            youtube_limit:  max youtube comments  (default: limit)
            stocktwits_limit: max stocktwits messages  (default: limit)
            hours_back:     time window in hours  (default: 48)
            skip_twitter:   bool – skip Twitter actor
            skip_reddit:    bool – skip Reddit actor
            skip_telegram:  bool – skip Telegram actor
            skip_youtube:   bool – skip YouTube actor
            skip_stocktwits: bool – skip StockTwits actor
        """
        twitter_limit = kwargs.get("twitter_limit", limit)
        reddit_limit = kwargs.get("reddit_limit", limit)
        telegram_limit = kwargs.get("telegram_limit", limit)
        youtube_limit = kwargs.get("youtube_limit", limit)
        stocktwits_limit = kwargs.get("stocktwits_limit", limit)
        hours_back = kwargs.get("hours_back", 48)
        skip_twitter = kwargs.get("skip_twitter", False)
        skip_reddit = kwargs.get("skip_reddit", False)
        skip_telegram = kwargs.get("skip_telegram", False)
        skip_youtube = kwargs.get("skip_youtube", False)
        skip_stocktwits = kwargs.get("skip_stocktwits", False)

        all_posts: List[Dict] = []

        # ── Twitter / X ────────────────────────────────────────
        if not skip_twitter:
            try:
                tweets = self._scrape_twitter(ticker, twitter_limit, hours_back)
                all_posts.extend(tweets)
                print(f"    Twitter/X: {len(tweets)} tweets")
            except Exception as exc:
                print(f"    ⚠️  Twitter/X scrape failed: {exc}")

        # ── Reddit ─────────────────────────────────────────────
        if not skip_reddit:
            try:
                posts = self._scrape_reddit(ticker, reddit_limit, hours_back)
                all_posts.extend(posts)
                print(f"    Reddit: {len(posts)} posts")
            except Exception as exc:
                print(f"    ⚠️  Reddit scrape failed: {exc}")

        # ── Telegram ───────────────────────────────────────────
        if not skip_telegram:
            try:
                msgs = self._scrape_telegram(ticker, telegram_limit)
                all_posts.extend(msgs)
                print(f"    Telegram: {len(msgs)} messages")
            except Exception as exc:
                print(f"    ⚠️  Telegram scrape failed: {exc}")

        # ── YouTube Comments ───────────────────────────────────
        if not skip_youtube:
            try:
                comments = self._scrape_youtube(ticker, youtube_limit)
                all_posts.extend(comments)
                print(f"    YouTube: {len(comments)} comments")
            except Exception as exc:
                print(f"    ⚠️  YouTube scrape failed: {exc}")

        # ── StockTwits ─────────────────────────────────────────
        if not skip_stocktwits:
            try:
                twits = self._scrape_stocktwits(ticker, stocktwits_limit)
                all_posts.extend(twits)
                print(f"    StockTwits: {len(twits)} messages")
            except Exception as exc:
                print(f"    ⚠️  StockTwits scrape failed: {exc}")

        return all_posts

    def _extract_post_data(self, raw_data: dict, **kwargs) -> Dict:
        """Implemented per-source in the private helpers below."""
        return raw_data

    # ── Twitter / X ────────────────────────────────────────────
    def _scrape_twitter(self, ticker: str, limit: int, hours_back: int) -> List[Dict]:
        """
        Run the Apify tweet-scraper actor and return standardised posts.
        """
        since_date = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).strftime("%Y-%m-%d")

        run_input = {
            "searchTerms": [f"${ticker}", ticker],
            "maxTweets": limit,
            "sort": "Latest",
            "tweetLanguage": "en",
            "since": since_date,
        }

        print(f"    Running Twitter/X scraper for ${ticker} (limit={limit}, since={since_date})…")
        run = self.client.actor(self.TWITTER_ACTOR).call(
            run_input=run_input,
            timeout_secs=120,  # hard cap: 2 minutes
        )

        # Collect whatever was scraped (actor may have timed out with partial data)
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            print(f"    ⚠️  No dataset returned from Twitter actor")
            return []

        items = list(self.client.dataset(dataset_id).iterate_items())
        # Actor may return more than requested — enforce limit client-side
        items = items[:limit]
        print(f"    → {len(items)} tweets retrieved from dataset")
        return [self._standardize_tweet(item) for item in items]

    def _standardize_tweet(self, raw: dict) -> Dict:
        """Map an Apify tweet-scraper item to the standard post dict."""
        text = raw.get("full_text") or raw.get("text", "")

        created = None
        raw_date = raw.get("created_at") or raw.get("createdAt")
        if raw_date:
            try:
                created = datetime.strptime(raw_date, "%a %b %d %H:%M:%S %z %Y")
            except (ValueError, TypeError):
                try:
                    created = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                except Exception:
                    created = datetime.now()
        else:
            created = datetime.now()

        user = raw.get("user") or raw.get("author") or {}
        followers = (
            user.get("followers_count")
            or user.get("followersCount")
            or raw.get("author_followers")
            or 0
        )
        verified = user.get("verified") or user.get("isVerified") or False
        username = (
            user.get("screen_name")
            or user.get("userName")
            or raw.get("author", "unknown")
        )

        # Account age
        user_created = user.get("created_at") or user.get("createdAt")
        account_age = 365  # default
        if user_created:
            try:
                acct_dt = datetime.strptime(user_created, "%a %b %d %H:%M:%S %z %Y")
                account_age = (datetime.now(timezone.utc) - acct_dt).days
            except Exception:
                pass

        likes_count = raw.get("favorite_count") or raw.get("likeCount") or 0
        retweet_count = raw.get("retweet_count") or raw.get("retweetCount") or 0
        reply_count = raw.get("reply_count") or raw.get("replyCount") or 0

        post_data = {
            "id": str(raw.get("id") or raw.get("id_str", "")),
            "text": text,
            "full_text": text,
            "author": username,
            "created_utc": created,
            "account_age_days": account_age,
            "author_followers": followers,
            "author_karma": 0,
            "is_verified": verified,
            "has_links": bool(raw.get("entities", {}).get("urls")),
            "likes": likes_count,
            "retweets": retweet_count,
            "replies": reply_count,
            "num_comments": reply_count,
            "score": likes_count,
            "url": raw.get("url", ""),
            "engagement_score": self._calculate_engagement_score(
                likes=likes_count,
                comments=reply_count,
                retweets=retweet_count,
            ),
            "post_age_hours": self._calculate_post_age_hours(created),
        }
        return self._standardize_post(post_data)

    # ── Reddit ─────────────────────────────────────────────────
    def _scrape_reddit(self, ticker: str, limit: int, hours_back: int) -> List[Dict]:
        """
        Run the Apify Reddit scraper actor.
        Searches across financial subreddits for the ticker.
        """
        search_urls = [
            f"https://www.reddit.com/r/{sub}/search.json?q={ticker}&sort=new&t=week"
            for sub in self.FINANCE_SUBREDDITS
        ]

        run_input = {
            "startUrls": [{"url": u} for u in search_urls],
            "maxItems": limit,
            "sort": "new",
            "proxy": {"useApifyProxy": True},
        }

        print(f"    Running Reddit scraper for {ticker} across {len(self.FINANCE_SUBREDDITS)} subreddits…")
        run = self.client.actor(self.REDDIT_ACTOR).call(
            run_input=run_input,
            timeout_secs=120,  # hard cap: 2 minutes
        )

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            print(f"    ⚠️  No dataset returned from Reddit actor")
            return []

        items = list(self.client.dataset(dataset_id).iterate_items())
        items = items[:limit]
        print(f"    → {len(items)} Reddit posts retrieved from dataset")
        return [self._standardize_reddit_post(item) for item in items]

    def _standardize_reddit_post(self, raw: dict) -> Dict:
        """Map an Apify Reddit item to the standard post dict."""
        title = raw.get("title", "")
        body = raw.get("body") or raw.get("selftext") or raw.get("text", "")
        text = f"{title} {body}".strip() if title else body

        created = None
        raw_ts = raw.get("created_utc") or raw.get("createdAt") or raw.get("scrapedAt")
        if raw_ts:
            try:
                if isinstance(raw_ts, (int, float)):
                    created = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
                else:
                    created = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            except Exception:
                created = datetime.now()
        else:
            created = datetime.now()

        ups = raw.get("upVotes") or raw.get("score") or raw.get("ups", 0)
        comments = raw.get("numberOfComments") or raw.get("num_comments", 0)
        author = raw.get("username") or raw.get("author", "unknown")

        post_data = {
            "id": str(raw.get("id", "")),
            "text": text[:300],
            "full_text": text,
            "author": author,
            "created_utc": created,
            "account_age_days": 365,  # not always available from scraper
            "author_karma": 0,
            "author_followers": 0,
            "is_verified": False,
            "has_links": bool(raw.get("url")),
            "likes": ups,
            "retweets": 0,
            "replies": comments,
            "num_comments": comments,
            "score": ups,
            "subreddit": raw.get("subreddit") or raw.get("communityName", ""),
            "url": raw.get("url", ""),
            "engagement_score": self._calculate_engagement_score(
                likes=ups,
                comments=comments,
                score=ups,
            ),
            "post_age_hours": self._calculate_post_age_hours(created),
        }
        return self._standardize_post(post_data)

    # ── Telegram ───────────────────────────────────────────
    def _scrape_telegram(self, ticker: str, limit: int) -> List[Dict]:
        """
        Run the Apify Telegram scraper actor across finance channels,
        then filter messages mentioning the ticker.
        """
        print(f"    Running Telegram scraper across {len(self.TELEGRAM_CHANNELS)} channels…")

        run = self.client.actor(self.TELEGRAM_ACTOR).call(
            run_input={
                "profiles": self.TELEGRAM_CHANNELS,
                "scrapeMessages": True,
                "messagesLimit": 200,  # per channel, then filter client-side
            },
            timeout_secs=120,
        )

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            print(f"    ⚠️  No dataset returned from Telegram actor")
            return []

        items = list(self.client.dataset(dataset_id).iterate_items())
        # Only keep items that have an actual message
        items = [i for i in items if isinstance(i.get("message"), dict) and i["message"].get("description")]
        print(f"    → {len(items)} total messages scraped")

        # Filter for messages mentioning the ticker
        ticker_upper = ticker.upper()
        ticker_patterns = [f"${ticker_upper}", ticker_upper]
        relevant = []
        for item in items:
            text = item["message"]["description"]
            if any(p in text.upper() for p in ticker_patterns):
                relevant.append(item)

        print(f"    → {len(relevant)} messages mention {ticker_upper}")
        return [self._standardize_telegram_msg(item) for item in relevant[:limit]]

    def _standardize_telegram_msg(self, raw: dict) -> Dict:
        """Map an Apify Telegram message to the standard post dict."""
        msg = raw.get("message", {})
        text = msg.get("description", "")
        channel = raw.get("username", "unknown")

        created = datetime.now(timezone.utc)
        raw_date = msg.get("fulldate") or msg.get("date")
        if raw_date:
            try:
                created = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            except Exception:
                pass

        views = 0
        raw_views = msg.get("views")
        if raw_views:
            try:
                views = int(str(raw_views).replace("K", "000").replace("M", "000000").replace(".", ""))
            except (ValueError, TypeError):
                pass

        post_data = {
            "id": msg.get("link", ""),
            "text": text[:300],
            "full_text": text,
            "author": raw.get("fullName") or channel,
            "created_utc": created,
            "account_age_days": 365,
            "author_karma": 0,
            "author_followers": raw.get("followers") or 0,
            "is_verified": raw.get("verified", False),
            "has_links": bool(msg.get("preview")),
            "likes": views,
            "retweets": 0,
            "replies": 0,
            "num_comments": 0,
            "score": views,
            "channel": channel,
            "url": msg.get("link", ""),
            "engagement_score": self._calculate_engagement_score(likes=views),
            "post_age_hours": self._calculate_post_age_hours(created),
        }
        return self._standardize_post(post_data)

    # ── YouTube Comments ───────────────────────────────────────
    def _scrape_youtube(self, ticker: str, limit: int) -> List[Dict]:
        """
        Run the Apify YouTube Comments scraper.
        Searches for stock-related videos and scrapes their comments.
        """
        search_urls = [
            f"https://www.youtube.com/results?search_query={ticker}+stock+analysis",
            f"https://www.youtube.com/results?search_query=${ticker}+stock",
        ]

        print(f"    Running YouTube Comments scraper for {ticker}…")
        run = self.client.actor(self.YOUTUBE_ACTOR).call(
            run_input={
                "startUrls": [{"url": u} for u in search_urls],
                "maxComments": limit,
                "maxReplies": 0,
                "mode": "comments",
            },
            timeout_secs=120,
        )

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            print(f"    ⚠️  No dataset returned from YouTube actor")
            return []

        items = list(self.client.dataset(dataset_id).iterate_items())
        items = items[:limit]
        print(f"    → {len(items)} YouTube comments retrieved from dataset")
        return [self._standardize_youtube_comment(item) for item in items]

    def _standardize_youtube_comment(self, raw: dict) -> Dict:
        """Map an Apify YouTube comment item to the standard post dict."""
        text = raw.get("text") or raw.get("comment") or raw.get("body", "")
        author = raw.get("author") or raw.get("authorName") or raw.get("username", "unknown")

        created = datetime.now(timezone.utc)
        raw_date = raw.get("publishedAt") or raw.get("date") or raw.get("publishedTimeText")
        if raw_date:
            try:
                created = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            except Exception:
                pass

        likes = raw.get("likeCount") or raw.get("votes") or raw.get("likes", 0)
        if isinstance(likes, str):
            try:
                likes = int(likes.replace(",", ""))
            except ValueError:
                likes = 0

        reply_count = raw.get("replyCount") or raw.get("replies", 0)
        if isinstance(reply_count, str):
            try:
                reply_count = int(reply_count.replace(",", ""))
            except ValueError:
                reply_count = 0

        video_id = raw.get("videoId") or raw.get("video_id", "")
        video_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""

        post_data = {
            "id": raw.get("id") or raw.get("commentId", ""),
            "text": text[:300],
            "full_text": text,
            "author": author,
            "created_utc": created,
            "account_age_days": 365,
            "author_karma": 0,
            "author_followers": 0,
            "is_verified": False,
            "has_links": "http" in text.lower(),
            "likes": likes,
            "retweets": 0,
            "replies": reply_count,
            "num_comments": reply_count,
            "score": likes,
            "video_id": video_id,
            "url": video_url,
            "engagement_score": self._calculate_engagement_score(
                likes=likes, comments=reply_count,
            ),
            "post_age_hours": self._calculate_post_age_hours(created),
        }
        return self._standardize_post(post_data)

    # ── StockTwits ─────────────────────────────────────────────
    def _scrape_stocktwits(self, ticker: str, limit: int) -> List[Dict]:
        """
        Run the Apify StockTwits scraper for a specific ticker.
        """
        print(f"    Running StockTwits scraper for ${ticker}…")
        run = self.client.actor(self.STOCKTWITS_ACTOR).call(
            run_input={"ticker": ticker},
            timeout_secs=120,
        )

        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            print(f"    ⚠️  No dataset returned from StockTwits actor")
            return []

        items = list(self.client.dataset(dataset_id).iterate_items())
        items = items[:limit]
        print(f"    → {len(items)} StockTwits messages retrieved from dataset")
        return [self._standardize_stocktwits_msg(item) for item in items]

    def _standardize_stocktwits_msg(self, raw: dict) -> Dict:
        """Map an Apify StockTwits item to the standard post dict."""
        text = raw.get("body") or raw.get("text") or raw.get("message", "")
        author = raw.get("username") or raw.get("user", {}).get("username", "unknown")

        created = datetime.now(timezone.utc)
        raw_date = raw.get("created_at") or raw.get("createdAt") or raw.get("date")
        if raw_date:
            try:
                created = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            except Exception:
                try:
                    created = datetime.strptime(str(raw_date), "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass

        likes = raw.get("likes_count") or raw.get("likes", 0)
        if isinstance(likes, dict):
            likes = likes.get("total", 0)
        if isinstance(likes, str):
            try:
                likes = int(likes)
            except ValueError:
                likes = 0

        # StockTwits users can tag bullish/bearish
        sentiment_label = None
        sentiment = raw.get("sentiment") or raw.get("entities", {}).get("sentiment")
        if isinstance(sentiment, dict):
            basic = sentiment.get("basic", "")
            if basic == "Bullish":
                sentiment_label = "bullish"
            elif basic == "Bearish":
                sentiment_label = "bearish"
        elif isinstance(sentiment, str):
            if sentiment.lower() in ("bullish", "bearish"):
                sentiment_label = sentiment.lower()

        user = raw.get("user", {}) if isinstance(raw.get("user"), dict) else {}
        followers = user.get("followers", 0) or raw.get("followers", 0)
        msg_id = str(raw.get("id") or raw.get("message_id", ""))

        post_data = {
            "id": msg_id,
            "text": text[:300],
            "full_text": text,
            "author": author,
            "created_utc": created,
            "account_age_days": 365,
            "author_karma": 0,
            "author_followers": followers,
            "is_verified": user.get("official", False),
            "has_links": "http" in text.lower(),
            "likes": likes,
            "retweets": 0,
            "replies": 0,
            "num_comments": 0,
            "score": likes,
            "sentiment_label": sentiment_label,
            "url": f"https://stocktwits.com/{author}/message/{msg_id}" if msg_id else "",
            "engagement_score": self._calculate_engagement_score(
                likes=likes, multiplier=0.5,
            ),
            "post_age_hours": self._calculate_post_age_hours(created),
        }
        return self._standardize_post(post_data)

    # ── Utility ────────────────────────────────────────────────
    def test_connection(self) -> bool:
        """Verify Apify API key is valid."""
        try:
            user = self.client.user().get()
            print(f"  ✓ Connected as: {user.get('username', 'unknown')}")
            return True
        except Exception as exc:
            print(f"  ✗ Connection failed: {exc}")
            return False
