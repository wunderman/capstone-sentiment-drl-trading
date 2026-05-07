"""
Sentiment Analyzer Module
Uses FinBERT (financial BERT) to analyze sentiment of filtered social media posts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple


class SentimentAnalyzer:
    """Analyzes sentiment of financial text using FinBERT."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer with FinBERT model.
        
        Args:
            model_name: HuggingFace model name (default: FinBERT)
        """
        print(f"Loading sentiment model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        # FinBERT labels: positive, negative, neutral
        self.labels = ['positive', 'negative', 'neutral']
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and label
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().numpy()[0]
        
        # Create result dictionary
        result = {
            'positive': float(predictions[0]),
            'negative': float(predictions[1]),
            'neutral': float(predictions[2]),
        }
        
        # Determine dominant sentiment
        dominant_sentiment = max(result, key=result.get)
        result['label'] = dominant_sentiment
        result['confidence'] = result[dominant_sentiment]
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = result['positive'] - result['negative']
        result['score'] = sentiment_score
        
        return result
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()
            
            # Process each prediction
            for pred in predictions:
                result = {
                    'positive': float(pred[0]),
                    'negative': float(pred[1]),
                    'neutral': float(pred[2]),
                }
                
                dominant_sentiment = max(result, key=result.get)
                result['label'] = dominant_sentiment
                result['confidence'] = result[dominant_sentiment]
                result['score'] = result['positive'] - result['negative']
                
                results.append(result)
        
        return results
    
    def aggregate_sentiment(self, 
                          sentiments: List[Dict], 
                          weights: List[float] = None) -> Dict:
        """
        Aggregate multiple sentiment scores into a single score.
        
        Args:
            sentiments: List of sentiment dictionaries
            weights: Optional weights for each sentiment (e.g., confidence scores)
            
        Returns:
            Aggregated sentiment dictionary
        """
        if not sentiments:
            return {
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'count': 0,
            }
        
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0] * len(sentiments)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted averages
        avg_positive = sum(s['positive'] * w for s, w in zip(sentiments, normalized_weights))
        avg_negative = sum(s['negative'] * w for s, w in zip(sentiments, normalized_weights))
        avg_neutral = sum(s['neutral'] * w for s, w in zip(sentiments, normalized_weights))
        avg_score = sum(s['score'] * w for s, w in zip(sentiments, normalized_weights))
        
        # Determine label
        scores_dict = {
            'positive': avg_positive,
            'negative': avg_negative,
            'neutral': avg_neutral,
        }
        label = max(scores_dict, key=scores_dict.get)
        
        return {
            'score': avg_score,
            'label': label,
            'confidence': scores_dict[label],
            'positive': avg_positive,
            'negative': avg_negative,
            'neutral': avg_neutral,
            'count': len(sentiments),
            'distribution': {
                'positive_count': sum(1 for s in sentiments if s['label'] == 'positive'),
                'negative_count': sum(1 for s in sentiments if s['label'] == 'negative'),
                'neutral_count': sum(1 for s in sentiments if s['label'] == 'neutral'),
            }
        }
    
    def calculate_trade_signal(self, aggregated_sentiment: Dict) -> Dict:
        """
        Convert aggregated sentiment into trade signal.
        
        Args:
            aggregated_sentiment: Output from aggregate_sentiment()
            
        Returns:
            Trade signal dictionary
        """
        score = aggregated_sentiment['score']
        confidence = aggregated_sentiment['confidence']
        count = aggregated_sentiment['count']
        
        # Determine signal strength (0-100)
        signal_strength = min(100, abs(score) * 100)
        
        # Determine action
        if score > 0.3 and confidence > 0.6:
            action = "STRONG_BUY"
            signal_value = 2
        elif score > 0.1:
            action = "BUY"
            signal_value = 1
        elif score < -0.3 and confidence > 0.6:
            action = "STRONG_SELL"
            signal_value = -2
        elif score < -0.1:
            action = "SELL"
            signal_value = -1
        else:
            action = "HOLD"
            signal_value = 0
        
        # Adjust for sample size (more data = more confidence)
        reliability = min(100, (count / 50) * 100)  # 50+ posts = 100% reliability
        
        return {
            'action': action,
            'signal_value': signal_value,
            'signal_strength': signal_strength,
            'reliability': reliability,
            'sentiment_score': score,
            'confidence': confidence,
            'sample_size': count,
            'recommendation': self._generate_recommendation(action, score, confidence, count)
        }
    
    def _generate_recommendation(self, 
                                action: str, 
                                score: float, 
                                confidence: float,
                                count: int) -> str:
        """Generate human-readable recommendation."""
        if count < 10:
            sample_warning = " (WARNING: Low sample size, use caution)"
        else:
            sample_warning = ""
        
        sentiment_desc = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        
        if action == "STRONG_BUY":
            return f"Strong positive sentiment ({score:.2f}). Consider buying.{sample_warning}"
        elif action == "BUY":
            return f"Moderate positive sentiment ({score:.2f}). Buying opportunity.{sample_warning}"
        elif action == "STRONG_SELL":
            return f"Strong negative sentiment ({score:.2f}). Consider selling.{sample_warning}"
        elif action == "SELL":
            return f"Moderate negative sentiment ({score:.2f}). Selling opportunity.{sample_warning}"
        else:
            return f"Neutral sentiment ({score:.2f}). Hold position.{sample_warning}"
