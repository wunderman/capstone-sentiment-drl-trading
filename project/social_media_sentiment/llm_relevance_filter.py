"""
LLM-based Relevance Filter
Uses Ollama (local LLM) to intelligently filter social media posts
for stock trading relevance, replacing rigid regex-based filtering.
"""

import requests
import json
import re
from typing import Dict, List, Tuple
from tqdm import tqdm


class LLMRelevanceFilter:
    """
    Filters social media posts using a local LLM (Ollama) to determine
    relevance to stock trading sentiment analysis.
    
    The LLM evaluates each post on three dimensions:
    - Relevance: Is this about stock trading / financial markets?
    - Quality: Is this substantive analysis vs noise/spam?
    - Sentiment Usefulness: Does this contain actionable sentiment?
    """
    
    SYSTEM_PROMPT = """You are a financial relevance classifier for a stock trading sentiment system.
Your job is to evaluate social media posts and determine if they are useful for stock trading sentiment analysis.

You must respond with ONLY a JSON object, no other text.

Evaluate the post on these criteria:
1. RELEVANT: Is this post about stock trading, investing, or financial markets? (true/false)
2. QUALITY: Is this substantive content vs spam/noise/memes? (0-100)
   - 0-20: Spam, ads, pump-and-dump, bot content
   - 21-40: Very low effort (single emoji, "lol", "moon")
   - 41-60: Brief opinion with some substance
   - 61-80: Informed opinion with reasoning or data
   - 81-100: Detailed analysis, DD, or expert commentary
3. SENTIMENT_USEFUL: Does this contain clear sentiment about a stock that could inform trading? (true/false)
4. REASONING: One sentence explaining your decision.

Respond with ONLY this JSON (no markdown, no code fences):
{"relevant": true/false, "quality": 0-100, "sentiment_useful": true/false, "reasoning": "..."}"""

    def __init__(self,
                 model: str = "llama3.2",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 15,
                 min_quality: int = 40):
        """
        Initialize the LLM relevance filter.
        
        Args:
            model: Ollama model name
            base_url: Ollama API endpoint
            timeout: Request timeout per post (seconds)
            min_quality: Minimum quality score to pass (0-100)
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.min_quality = min_quality
        
        self._test_connection()
    
    def _test_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = [m['name'] for m in response.json().get('models', [])]
            if not any(self.model in name for name in models):
                raise ValueError(
                    f"Model '{self.model}' not found. Available: {', '.join(models)}\n"
                    f"Run: ollama pull {self.model}"
                )
            
            print(f"  ✓ LLM filter connected to Ollama ({self.model})")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. Start it with: ollama serve"
            )
    
    def _call_ollama(self, post_text: str) -> str:
        """Send a post to Ollama for evaluation."""
        payload = {
            "model": self.model,
            "prompt": f"{self.SYSTEM_PROMPT}\n\nPOST: {post_text[:500]}",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 150,
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get('response', '')
    
    def _parse_llm_response(self, raw: str) -> Dict:
        """Parse LLM JSON response, with fallback handling."""
        # Strip markdown fences if present
        raw = raw.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        
        # Find JSON object
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    'relevant': bool(data.get('relevant', False)),
                    'quality': int(data.get('quality', 0)),
                    'sentiment_useful': bool(data.get('sentiment_useful', False)),
                    'reasoning': str(data.get('reasoning', '')),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        
        # If LLM response is unparseable, reject the post
        return {
            'relevant': False,
            'quality': 0,
            'sentiment_useful': False,
            'reasoning': 'LLM response unparseable',
        }
    
    def filter_post(self, text: str, **kwargs) -> Dict:
        """
        Filter a single post using the LLM.
        
        Compatible with the same interface as RelevanceFilter.filter_post()
        so it can be used as a drop-in replacement.
        
        Returns:
            Dictionary with filter results matching RelevanceFilter format.
        """
        if not text or len(text.strip()) < 5:
            return {
                'passes_filter': False,
                'relevance_score': 0,
                'quality_score': 0,
                'account_score': 50,
                'confidence_level': 'REJECT',
                'confidence_percentage': 0,
                'relevance_breakdown': {},
                'reasoning': 'Empty or too short',
            }
        
        try:
            raw_response = self._call_ollama(text)
            llm_result = self._parse_llm_response(raw_response)
        except Exception:
            # On timeout/error, reject gracefully
            llm_result = {
                'relevant': False,
                'quality': 0,
                'sentiment_useful': False,
                'reasoning': 'LLM call failed',
            }
        
        # Map LLM result to the standard filter format
        quality_score = llm_result['quality']
        passes = (
            llm_result['relevant']
            and llm_result['sentiment_useful']
            and quality_score >= self.min_quality
        )
        
        # Confidence level mapping
        if quality_score >= 80:
            confidence_level = 'HIGH'
        elif quality_score >= 55:
            confidence_level = 'MEDIUM'
        elif quality_score >= 35:
            confidence_level = 'LOW'
        else:
            confidence_level = 'REJECT'
        
        confidence_pct = quality_score  # LLM quality IS our confidence
        
        return {
            'passes_filter': passes,
            'relevance_score': 10 if llm_result['relevant'] else 0,
            'quality_score': quality_score,
            'account_score': 50,  # LLM doesn't evaluate accounts
            'confidence_level': confidence_level,
            'confidence_percentage': confidence_pct,
            'relevance_breakdown': {
                'llm_relevant': llm_result['relevant'],
                'llm_sentiment_useful': llm_result['sentiment_useful'],
                'llm_quality': quality_score,
            },
            'reasoning': llm_result['reasoning'],
        }
    
    def filter_batch(self, posts: List[Dict], ticker: str = None) -> List[Dict]:
        """
        Filter a batch of posts using the LLM.
        Shows detailed evaluation for each post.
        
        Args:
            posts: List of post dicts with 'text' or 'full_text'
            ticker: Optional ticker for context
            
        Returns:
            List of posts that passed the filter, with filter_results attached.
        """
        filtered = []
        
        print(f"\n{'─'*70}")
        print(f"  LLM RELEVANCE EVALUATION ({len(posts)} posts)")
        print(f"{'─'*70}")
        
        for i, post in enumerate(posts, 1):
            text = post.get('full_text') or post.get('text', '')
            platform = post.get('platform', '?')
            author = post.get('author', 'unknown')
            preview = text[:120].replace('\n', ' ')
            
            result = self.filter_post(text=text)
            
            # Status icon
            status = "✓" if result['passes_filter'] else "✗"
            quality = result['quality_score']
            reasoning = result.get('reasoning', '')
            breakdown = result.get('relevance_breakdown', {})
            relevant = breakdown.get('llm_relevant', False)
            useful = breakdown.get('llm_sentiment_useful', False)
            
            print(f"\n  [{i}/{len(posts)}] @{author} ({platform})")
            print(f"  \"{preview}{'…' if len(text) > 120 else ''}\"")
            print(f"  {status} relevant={relevant}  sentiment_useful={useful}  quality={quality}/100")
            print(f"    → {reasoning}")
            
            if result['passes_filter']:
                post['filter_results'] = result
                filtered.append(post)
        
        print(f"\n{'─'*70}")
        print(f"  RESULT: {len(filtered)}/{len(posts)} posts passed LLM filter")
        print(f"{'─'*70}\n")
        
        return filtered
