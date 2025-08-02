import os
import logging
import threading
import time
import random
from datetime import datetime
from collections import deque
from typing import Dict, List, Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Sentiment analysis class that handles model loading and inference.
    Uses a simple rule-based approach as fallback when transformers are not available.
    """
    
    def __init__(self, model_name: str = None):
        """Initialize the sentiment analyzer."""
        self.model_name = model_name or "simple-rule-based"
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Try to load the transformer model
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model."""
        try:
            # Try to import and load transformers model
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            logger.info(f"Loading model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model_name = model_name
            self.is_loaded = True
            
            logger.info("Transformer model loaded successfully")
            
        except ImportError:
            logger.warning("Transformers not available, using rule-based approach")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")
            self.is_loaded = False
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict containing sentiment, confidence, and metadata
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': 'Empty text provided'
            }
        
        text = text.strip()
        
        if self.is_loaded:
            return self._analyze_with_transformer(text)
        else:
            return self._analyze_with_rules(text)
    
    def _analyze_with_transformer(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze text using transformer model."""
        try:
            # Tokenize and predict
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the predicted class and confidence
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            
            # Map class to sentiment (adjust based on your model)
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(predicted_class, 'neutral')
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 4),
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Transformer analysis failed: {e}")
            return self._analyze_with_rules(text)
    
    def _analyze_with_rules(self, text: str) -> Dict[str, Union[str, float]]:
        """Simple rule-based sentiment analysis as fallback."""
        text_lower = text.lower()
        
        # Simple positive/negative word lists
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'happy', 'joy', 'pleased', 'satisfied', 'awesome',
            'brilliant', 'perfect', 'outstanding', 'superb', 'marvelous'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'angry', 'sad', 'disappointed', 'frustrated', 'annoyed',
            'disgusted', 'furious', 'upset', 'miserable', 'pathetic'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.6 + (positive_count - negative_count) * 0.1, 0.9)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.6 + (negative_count - positive_count) * 0.1, 0.9)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'model': 'rule-based-fallback'
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        for i, text in enumerate(texts):
            result = self.analyze_text(text)
            result['index'] = i
            result['text'] = text
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Union[str, bool]]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'model_type': 'transformer' if self.is_loaded else 'rule-based',
            'status': 'ready' if self.is_loaded else 'fallback'
        }

# Global analyzer instance
analyzer = SentimentAnalyzer()

def get_analyzer() -> SentimentAnalyzer:
    """Get the global sentiment analyzer instance."""
    return analyzer


class StreamingProcessor:
    """
    Simulates real-time tweet streaming and processing.
    """
    def __init__(self, analyzer_instance: SentimentAnalyzer):
        self.analyzer = analyzer_instance
        self.stream_active = False
        self.processing_active = False
        self.tweets_per_minute = 60  # Default rate
        self.batch_size = 10
        self.processing_interval = 2.0 # seconds

        self.generated_tweets_count = 0
        self.processed_tweets_count = 0
        self.sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        self.recent_tweets = deque(maxlen=50) # Store last 50 processed tweets
        self.sentiment_trends = deque(maxlen=60) # Store sentiment distribution every minute
        self.confidence_trends = deque(maxlen=60) # Store average confidence every minute

        self._stream_thread = None
        self._processing_thread = None
        self._trend_thread = None
        self._lock = threading.Lock()

        self.tweet_templates = [
            "Just watched the new movie, it was {sentiment_adj}! #movie #review",
            "Feeling {sentiment_adj} about the weather today. #weather",
            "This new product is {sentiment_adj}! Highly recommend. #tech #innovation",
            "Traffic is making me feel {sentiment_adj}. #commute",
            "Had a {sentiment_adj} day at work. #worklife",
            "The food at that restaurant was {sentiment_adj}. #food #review",
            "Excited for the weekend, feeling {sentiment_adj}! #weekendvibes",
            "Can't believe how {sentiment_adj} this situation is. #frustration",
            "Learning new things is always {sentiment_adj}. #education",
            "Just finished a {sentiment_adj} book. #reading #books"
        ]
        self.positive_adjectives = ['amazing', 'fantastic', 'great', 'wonderful', 'excellent', 'joyful']
        self.negative_adjectives = ['terrible', 'awful', 'horrible', 'frustrating', 'bad', 'disappointing']
        self.neutral_adjectives = ['okay', 'alright', 'decent', 'average', 'fine', 'so-so']

    def _generate_tweet(self) -> Dict:
        sentiment_type = random.choice(['positive', 'negative', 'neutral'])
        if sentiment_type == 'positive':
            adj = random.choice(self.positive_adjectives)
        elif sentiment_type == 'negative':
            adj = random.choice(self.negative_adjectives)
        else:
            adj = random.choice(self.neutral_adjectives)
        
        text = random.choice(self.tweet_templates).format(sentiment_adj=adj)
        
        return {
            'id': self.generated_tweets_count + 1,
            'text': text,
            'timestamp': datetime.utcnow().isoformat(),
            'user': f'user_{random.randint(1000, 9999)}',
            'hashtags': ['#sentiment', '#ai', '#data']
        }

    def _stream_generator(self):
        logger.info("Stream generator started.")
        while self.stream_active:
            tweet = self._generate_tweet()
            with self._lock:
                self.generated_tweets_count += 1
                # In a real system, this would push to a message queue
                # For simulation, we'll just process it directly or add to a buffer
                self._process_single_tweet(tweet)
            time.sleep(60 / self.tweets_per_minute)
        logger.info("Stream generator stopped.")

    def _process_single_tweet(self, tweet: Dict):
        # This is a simplified direct processing for simulation
        # In a real system, tweets would be pulled from a queue by the processor
        analysis_result = self.analyzer.analyze_text(tweet['text'])
        
        with self._lock:
            self.processed_tweets_count += 1
            sentiment = analysis_result.get('sentiment', 'neutral')
            self.sentiment_counts[sentiment] = self.sentiment_counts.get(sentiment, 0) + 1
            
            processed_tweet = {
                'id': tweet['id'],
                'text': tweet['text'],
                'timestamp': tweet['timestamp'],
                'sentiment': sentiment,
                'confidence': analysis_result.get('confidence', 0.0),
                'model': analysis_result.get('model', 'unknown')
            }
            self.recent_tweets.appendleft(processed_tweet) # Add to the front

    def _processing_loop(self):
        logger.info("Processing loop started.")
        while self.processing_active:
            # In a real system, this would pull from a message queue in batches
            # For this simulation, the stream_generator directly processes
            # So this loop primarily manages the active state and interval
            time.sleep(self.processing_interval)
        logger.info("Processing loop stopped.")

    def _trend_aggregator(self):
        logger.info("Trend aggregator started.")
        while self.stream_active or self.processing_active:
            time.sleep(60) # Aggregate every minute
            with self._lock:
                total_sentiments = sum(self.sentiment_counts.values())
                if total_sentiments > 0:
                    positive_percent = (self.sentiment_counts['positive'] / total_sentiments) * 100
                    negative_percent = (self.sentiment_counts['negative'] / total_sentiments) * 100
                    neutral_percent = (self.sentiment_counts['neutral'] / total_sentiments) * 100
                else:
                    positive_percent, negative_percent, neutral_percent = 0, 0, 0

                avg_confidence = 0.0
                if self.processed_tweets_count > 0:
                    # This is a simplified average, for real avg, sum confidences
                    avg_confidence = sum(t['confidence'] for t in self.recent_tweets) / len(self.recent_tweets) if self.recent_tweets else 0.0

                self.sentiment_trends.appendleft({
                    'timestamp': datetime.utcnow().isoformat(),
                    'positive': round(positive_percent, 2),
                    'negative': round(negative_percent, 2),
                    'neutral': round(neutral_percent, 2)
                })
                self.confidence_trends.appendleft({
                    'timestamp': datetime.utcnow().isoformat(),
                    'average_confidence': round(avg_confidence, 2)
                })
                # Reset counts for next minute
                self.sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        logger.info("Trend aggregator stopped.")

    def start_streaming(self, tweets_per_minute: int = 60, batch_size: int = 10, processing_interval: float = 2.0):
        with self._lock:
            if self.stream_active or self.processing_active:
                return False, "Streaming system already active."
            
            self.tweets_per_minute = tweets_per_minute
            self.batch_size = batch_size
            self.processing_interval = processing_interval
            self.stream_active = True
            self.processing_active = True
            self.generated_tweets_count = 0
            self.processed_tweets_count = 0
            self.sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            self.recent_tweets.clear()
            self.sentiment_trends.clear()
            self.confidence_trends.clear()

            self._stream_thread = threading.Thread(target=self._stream_generator)
            self._processing_thread = threading.Thread(target=self._processing_loop)
            self._trend_thread = threading.Thread(target=self._trend_aggregator)

            self._stream_thread.start()
            self._processing_thread.start()
            self._trend_thread.start()
            logger.info(f"Streaming system started with {tweets_per_minute} tweets/min, batch size {batch_size}")
            return True, "Streaming system started successfully."

    def stop_streaming(self):
        with self._lock:
            if not self.stream_active and not self.processing_active:
                return False, "Streaming system not active."
            self.stream_active = False
            self.processing_active = False
            
            # Wait for threads to finish (with a timeout)
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=5)
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5)
            if self._trend_thread and self._trend_thread.is_alive():
                self._trend_thread.join(timeout=5)

            logger.info("Streaming system stopped.")
            return True, "Streaming system stopped successfully."

    def get_status(self) -> Dict:
        with self._lock:
            return {
                'stream_active': self.stream_active,
                'processing_active': self.processing_active,
                'tweets_per_minute': self.tweets_per_minute,
                'batch_size': self.batch_size,
                'processing_interval': self.processing_interval,
                'generated_tweets_count': self.generated_tweets_count,
                'processed_tweets_count': self.processed_tweets_count,
                'current_sentiment_distribution': {
                    'positive': self.sentiment_counts['positive'],
                    'negative': self.sentiment_counts['negative'],
                    'neutral': self.sentiment_counts['neutral']
                },
                'recent_tweets': list(self.recent_tweets),
                'sentiment_trends': list(self.sentiment_trends),
                'confidence_trends': list(self.confidence_trends)
            }

# Global streaming processor instance
streaming_processor = StreamingProcessor(analyzer)

def get_streaming_processor() -> StreamingProcessor:
    """Get the global streaming processor instance."""
    return streaming_processor


