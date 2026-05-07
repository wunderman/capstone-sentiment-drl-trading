"""
News Collector
Aggregates financial news headlines/summaries for a ticker from free sources:
  - Yahoo Finance RSS (per-ticker, no key)
  - Google News RSS (query-based, no key)
  - Finnhub /company-news (free tier, requires FINNHUB_API_KEY)

Output shape matches BaseSocialMediaCollector._standardize_post so downstream
filtering/sentiment code treats each article as a "post".
"""

import os
import time
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests

from .base_collector import BaseSocialMediaCollector


class NewsCollector(BaseSocialMediaCollector):
    """Collects financial news articles for a ticker across several free sources."""

    USER_AGENT = (
        "Mozilla/5.0 (compatible; CapstoneSentimentBot/1.0; "
        "+https://github.com/moncef09/Capstone)"
    )

    def __init__(self,
                 finnhub_api_key: Optional[str] = None,
                 use_yahoo: bool = True,
                 use_google: bool = True,
                 use_finnhub: bool = True,
                 request_timeout: int = 10):
        super().__init__(platform_name="news")
        self.finnhub_api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.use_yahoo = use_yahoo
        self.use_google = use_google
        self.use_finnhub = use_finnhub and bool(self.finnhub_api_key)
        self.timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

        # Minimal ticker -> company name mapping so Google News query is
        # meaningful (tickers alone return too many false positives).
        self.ticker_to_company = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'AMZN': 'Amazon',
            'NVDA': 'Nvidia', 'TSLA': 'Tesla', 'META': 'Meta Platforms',
            'GOOGL': 'Alphabet', 'GOOG': 'Alphabet', 'JPM': 'JPMorgan',
            'GS': 'Goldman Sachs', 'V': 'Visa', 'MA': 'Mastercard',
            'JNJ': 'Johnson & Johnson', 'WMT': 'Walmart', 'PG': 'Procter & Gamble',
            'KO': 'Coca-Cola', 'DIS': 'Disney', 'NKE': 'Nike', 'BA': 'Boeing',
            'CAT': 'Caterpillar', 'HD': 'Home Depot', 'HON': 'Honeywell',
            'CVX': 'Chevron', 'MRK': 'Merck', 'AMGN': 'Amgen', 'CSCO': 'Cisco',
            'INTC': 'Intel', 'IBM': 'IBM', 'VZ': 'Verizon', 'CRM': 'Salesforce',
            'UNH': 'UnitedHealth', 'AXP': 'American Express', 'TRV': 'Travelers',
            'MMM': '3M', 'DOW': 'Dow Inc', 'MCD': "McDonald's", 'NFLX': 'Netflix',
            'PYPL': 'PayPal', 'UBER': 'Uber', 'COIN': 'Coinbase', 'PLTR': 'Palantir',
            'SHOP': 'Shopify', 'SNOW': 'Snowflake', 'CRWD': 'CrowdStrike',
            'PANW': 'Palo Alto Networks', 'DDOG': 'Datadog', 'NET': 'Cloudflare',
            'RIVN': 'Rivian', 'LCID': 'Lucid', 'ABNB': 'Airbnb', 'SQ': 'Block',
            'AMD': 'AMD', 'QCOM': 'Qualcomm', 'TXN': 'Texas Instruments',
            'LLY': 'Eli Lilly', 'ABBV': 'AbbVie', 'COST': 'Costco',
            'ACN': 'Accenture', 'ADBE': 'Adobe', 'AVGO': 'Broadcom',
            'ORCL': 'Oracle', 'PEP': 'PepsiCo', 'TMO': 'Thermo Fisher',
            'XOM': 'ExxonMobil', 'BAC': 'Bank of America',
        }

    # ---------- public API ----------

    def search_ticker(self,
                      ticker: str,
                      limit: int = 100,
                      hours_back: int = 72,
                      **kwargs) -> List[Dict]:
        """Return deduped list of news articles mentioning the ticker."""
        ticker = ticker.upper().lstrip("$")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        company = self.ticker_to_company.get(ticker, ticker)

        articles: List[Dict] = []

        if self.use_yahoo:
            try:
                articles.extend(self._fetch_yahoo(ticker))
            except Exception as e:
                print(f"  ⚠️  Yahoo RSS failed: {e}")

        if self.use_google:
            try:
                articles.extend(self._fetch_google(ticker, company))
            except Exception as e:
                print(f"  ⚠️  Google News RSS failed: {e}")

        if self.use_finnhub:
            try:
                articles.extend(self._fetch_finnhub(ticker, hours_back))
            except Exception as e:
                print(f"  ⚠️  Finnhub failed: {e}")

        # Filter by cutoff window + dedupe by URL/title
        seen = set()
        filtered: List[Dict] = []
        for art in articles:
            created = art.get("created_utc")
            if isinstance(created, datetime):
                created_cmp = created if created.tzinfo else created.replace(tzinfo=timezone.utc)
                if created_cmp < cutoff:
                    continue
            key = art.get("url") or art.get("id") or art.get("full_text", "")[:100]
            if key in seen:
                continue
            seen.add(key)
            filtered.append(art)

        # Sort newest first, cap at limit
        filtered.sort(
            key=lambda a: a.get("created_utc") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return filtered[:limit]

    # ---------- source: Yahoo Finance RSS ----------

    def _fetch_yahoo(self, ticker: str) -> List[Dict]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return self._parse_rss(r.content, source="yahoo_finance", ticker=ticker)

    # ---------- source: Google News RSS ----------

    def _fetch_google(self, ticker: str, company: str) -> List[Dict]:
        # Use both ticker and company name for better recall
        query = f'"{company}" OR "{ticker}" stock'
        url = (
            f"https://news.google.com/rss/search?q={quote_plus(query)}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        return self._parse_rss(r.content, source="google_news", ticker=ticker)

    # ---------- source: Finnhub ----------

    def _fetch_finnhub(self, ticker: str, hours_back: int) -> List[Dict]:
        end = datetime.now(timezone.utc).date()
        start = (datetime.now(timezone.utc) - timedelta(hours=max(hours_back, 24))).date()
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": self.finnhub_api_key,
        }
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() or []
        out = []
        for item in data:
            ts = item.get("datetime")
            created = (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                if isinstance(ts, (int, float)) else None
            )
            headline = item.get("headline") or ""
            summary = item.get("summary") or ""
            text = (headline + ". " + summary).strip(". ").strip()
            if not text:
                continue
            url_ = item.get("url", "")
            out.append(self._standardize_post({
                "id": f"finnhub_{ticker}_{item.get('id') or hashlib.md5(url_.encode()).hexdigest()[:10]}",
                "text": headline,
                "full_text": text,
                "author": item.get("source", "finnhub"),
                "created_utc": created,
                "url": url_,
                "post_age_hours": self._calculate_post_age_hours(created) if created else 0,
                "has_links": True,
                "is_verified": True,
                "engagement_score": 50,  # News: treat as moderately credible
                "source_sub": "finnhub",
            }))
        return out

    # ---------- RSS parsing ----------

    def _parse_rss(self, content: bytes, source: str, ticker: str) -> List[Dict]:
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return []
        out = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = item.findtext("pubDate") or ""
            try:
                created = parsedate_to_datetime(pub) if pub else None
                if created and created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                created = None

            # Strip HTML tags from description cheaply
            desc_text = _strip_tags(desc)
            text = title if not desc_text else f"{title}. {desc_text}"
            if not text.strip():
                continue

            uid = hashlib.md5((source + link + title).encode()).hexdigest()[:16]
            out.append(self._standardize_post({
                "id": f"{source}_{uid}",
                "text": title,
                "full_text": text,
                "author": source,
                "created_utc": created,
                "url": link,
                "post_age_hours": self._calculate_post_age_hours(created) if created else 0,
                "has_links": True,
                "is_verified": True,
                "engagement_score": 40,
                "source_sub": source,
            }))
        return out

    # ---------- required by ABC ----------

    def _extract_post_data(self, raw_data, **kwargs) -> Dict:
        # News parsing is handled inline in _parse_rss / _fetch_finnhub;
        # this method exists to satisfy the abstract base class.
        return raw_data if isinstance(raw_data, dict) else {}


def _strip_tags(html: str) -> str:
    import re
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text
