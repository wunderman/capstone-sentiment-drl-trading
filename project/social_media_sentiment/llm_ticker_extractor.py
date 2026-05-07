"""
LLM-based Ticker Extraction using Ollama
Uses local Ollama models to intelligently extract stock tickers from text.
"""

import requests
import json
import re
from typing import List, Dict, Optional


class OllamaTickerExtractor:
    """Extract stock tickers using Ollama LLM for intelligent context understanding."""
    
    def __init__(self, 
                 model: str = "llama3.2",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30):
        """
        Initialize Ollama ticker extractor.
        
        Args:
            model: Ollama model name (llama3.2, mistral, gemma2, etc.)
            base_url: Ollama API endpoint
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if not any(self.model in name for name in model_names):
                print(f"⚠️  Model '{self.model}' not found in Ollama.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"Run: ollama pull {self.model}")
                raise ValueError(f"Model {self.model} not available")
            
            print(f"✓ Connected to Ollama - using model: {self.model}")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running?\n"
                "Start it with: ollama serve"
            )
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text using LLM reasoning.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of ticker symbols (uppercase)
        """
        prompt = f"""Extract stock ticker symbols from the following text. Return ONLY a JSON object, nothing else.

RULES:
- Only extract tickers being actively DISCUSSED (not in negative context like "Not like AAPL")
- Valid tickers are 2-5 uppercase letters
- For Indian stocks with .NS/.BO, return just the symbol
- If no tickers, return empty array

TEXT: {text}

Return this exact JSON format (no code, no explanation):
{{"tickers": ["TICKER1", "TICKER2"]}}"""

        try:
            response = self._call_ollama(prompt)
            # Debug: print raw response
            # print(f"DEBUG - Raw LLM response: {response[:200]}")
            tickers = self._parse_response(response)
            return tickers
            
        except Exception as e:
            # Silent fallback to regex
            return self._fallback_regex(text)
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9,
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response to extract ticker list."""
        
        # Try to find JSON in response
        json_match = re.search(r'\{.*?"tickers".*?\[.*?\].*?\}', response, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                tickers = data.get('tickers', [])
                
                # Validate and clean
                valid_tickers = []
                for ticker in tickers:
                    ticker = str(ticker).strip().upper()
                    # Remove common suffixes
                    ticker = ticker.replace('.NS', '').replace('.BO', '')
                    # Validate format (2-5 letters, must be alphanumeric)
                    if re.match(r'^[A-Z]{2,5}$', ticker):
                        valid_tickers.append(ticker)
                
                return valid_tickers
                
            except json.JSONDecodeError:
                pass
        
        # Fallback: Look for array of tickers in response
        # Pattern: ["TICKER1", "TICKER2"] or ['TICKER1', 'TICKER2']
        array_match = re.search(r'\[(.*?)\]', response)
        if array_match:
            content = array_match.group(1)
            # Extract quoted strings
            tickers = re.findall(r'["\']([A-Z]{2,5})["\']', content)
            excluded = {'JSON', 'TEXT', 'LIST', 'NONE', 'NULL', 'TRUE', 'FALSE', 'THE', 'AND', 'FOR', 'NS', 'BO'}
            return [t for t in tickers if t not in excluded]
        
        return []
    
    def _fallback_regex(self, text: str) -> List[str]:
        """Fallback to regex-based extraction if LLM fails."""
        tickers = set()
        
        # $SYMBOL
        dollar_tickers = re.findall(r'\$([A-Z]{2,5})\b', text)
        tickers.update(dollar_tickers)
        
        # SYMBOL.NS or SYMBOL.BO
        indian_tickers = re.findall(r'\b([A-Z]+)\.(NS|BO|BSE|NSE)\b', text)
        tickers.update([t[0] for t in indian_tickers])
        
        return sorted(list(tickers))


def test_ollama_extraction():
    """Test Ollama ticker extraction."""
    print("\n" + "="*70)
    print("  TESTING OLLAMA TICKER EXTRACTION")
    print("="*70)
    
    # Test cases
    test_cases = [
        ("$AAPL and $TSLA are bullish today", ["AAPL", "TSLA"]),
        ("HDFC Bank reports strong earnings", ["HDFC"]),
        ("Unlike AAPL, this company is struggling", []),  # Should NOT extract AAPL
        ("Not Adani or Tata, but smaller companies", []),  # Should NOT extract
        ("US markets and UK indices rise", []),  # Should NOT extract US/UK
        ("HDFC, SBI, ICICI all report profits", ["HDFC", "SBI", "ICICI"]),
    ]
    
    try:
        print("\nInitializing Ollama extractor...")
        extractor = OllamaTickerExtractor()
        
        print("\nRunning tests...\n")
        passed = 0
        failed = 0
        
        for i, (text, expected) in enumerate(test_cases, 1):
            print(f"{i}. Text: {text}")
            result = extractor.extract_tickers(text)
            
            # Show raw response for first test
            if i == 1:
                print(f"   [Debug - testing raw response]")
            
            # Check if result matches expected (order doesn't matter)
            is_correct = set(result) == set(expected)
            status = "✓" if is_correct else "✗"
            
            print(f"   Expected: {expected}")
            print(f"   Got:      {result} {status}")
            
            if is_correct:
                passed += 1
            else:
                failed += 1
            print()
        
        print("="*70)
        print(f"Results: {passed} passed, {failed} failed")
        print("="*70)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  1. ollama serve")
        print("  2. ollama pull llama3.2")


if __name__ == "__main__":
    test_ollama_extraction()
