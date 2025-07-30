"""
Real-time Data Streaming System for Social Media Sentiment Analysis
Simulates Twitter streaming and provides real-time data processing capabilities
"""
import json
import time
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from queue import Queue, Empty
import uuid
from dataclasses import dataclass, asdict
from collections import deque
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamedTweet:
    """Data class representing a streamed tweet."""
    id: str
    text: str
    timestamp: str
    user_id: str
    username: str
    hashtags: List[str]
    mentions: List[str]
    retweet_count: int
    like_count: int
    language: str = "en"
    location: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

class TwitterStreamSimulator:
    """
    Simulates Twitter streaming API for real-time sentiment analysis.
    Generates realistic tweet data for testing and demonstration purposes.
    """
    
    def __init__(self, tweets_per_minute: int = 100):
        """
        Initialize the Twitter stream simulator.
        
        Args:
            tweets_per_minute (int): Number of tweets to generate per minute
        """
        self.tweets_per_minute = tweets_per_minute
        self.is_streaming = False
        self.stream_thread = None
        self.data_queue = Queue(maxsize=1000)
        
        # Sample tweet templates for different sentiments
        self.positive_templates = [
            "I absolutely love {product}! Best purchase ever! 😍",
            "Amazing experience with {service}! Highly recommend! ⭐⭐⭐⭐⭐",
            "Just had the best {experience} today! So happy! 😊",
            "Fantastic {product}! Worth every penny! 💯",
            "Great job {company}! Keep up the excellent work! 👏",
            "Perfect {service}! Exactly what I needed! ✨",
            "Outstanding {product}! Exceeded my expectations! 🚀",
            "Loving the new {feature}! So convenient! ❤️",
            "Brilliant {service}! Customer service was amazing! 🌟",
            "Incredible {experience}! Will definitely come back! 🎉"
        ]
        
        self.negative_templates = [
            "Terrible experience with {service}. Very disappointed. 😞",
            "Worst {product} I've ever bought. Complete waste of money. 😡",
            "Awful {service}! Never using them again. 👎",
            "Horrible {experience}. Staff was rude and unhelpful. 😠",
            "Disappointing {product}. Not worth the hype. 😒",
            "Bad {service}. Long wait times and poor quality. ⏰",
            "Frustrated with {company}. They don't care about customers. 😤",
            "Regret buying {product}. Should have read reviews first. 😔",
            "Unacceptable {service}. Demanding a refund! 💸",
            "Pathetic {experience}. Will never recommend to anyone. 🚫"
        ]
        
        self.neutral_templates = [
            "Used {service} today. It was okay, nothing special.",
            "{Product} is decent. Does what it's supposed to do.",
            "Had {experience} yesterday. Average, I guess.",
            "{Service} is fine. Not great, not terrible.",
            "Tried {product}. It's alright for the price.",
            "{Experience} was standard. Nothing to complain about.",
            "Using {service} now. Works as expected.",
            "{Product} is okay. Could be better, could be worse.",
            "Regular {experience}. Pretty much what I expected.",
            "{Service} is functional. Gets the job done."
        ]
        
        # Sample entities for templates
        self.products = ["iPhone", "Tesla", "Netflix", "Spotify", "Amazon Prime", "MacBook", "AirPods", "PlayStation", "Nintendo Switch", "Samsung Galaxy"]
        self.services = ["Uber", "DoorDash", "customer service", "tech support", "delivery service", "banking app", "streaming service", "online shopping", "food delivery", "ride sharing"]
        self.companies = ["@Apple", "@Tesla", "@Netflix", "@Amazon", "@Google", "@Microsoft", "@Meta", "@Twitter", "@Samsung", "@Sony"]
        self.experiences = ["shopping experience", "dining experience", "travel experience", "movie experience", "concert experience", "hotel stay", "flight experience", "vacation", "weekend trip", "business meeting"]
        
        # Hashtag pools
        self.positive_hashtags = ["#love", "#amazing", "#perfect", "#awesome", "#great", "#excellent", "#fantastic", "#wonderful", "#brilliant", "#outstanding"]
        self.negative_hashtags = ["#terrible", "#awful", "#worst", "#disappointed", "#frustrated", "#angry", "#horrible", "#pathetic", "#unacceptable", "#regret"]
        self.neutral_hashtags = ["#okay", "#average", "#standard", "#regular", "#normal", "#fine", "#decent", "#alright", "#fair", "#moderate"]
        
        # User pools
        self.usernames = [f"user{i:04d}" for i in range(1, 1001)]
        self.locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA"]
        
        # Statistics tracking
        self.total_generated = 0
        self.sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        
    def _generate_tweet(self) -> StreamedTweet:
        """Generate a single realistic tweet."""
        # Randomly choose sentiment (weighted towards positive for realism)
        sentiment_weights = [0.4, 0.3, 0.3]  # positive, negative, neutral
        sentiment = random.choices(["positive", "negative", "neutral"], weights=sentiment_weights)[0]
        
        # Select template and entities based on sentiment
        if sentiment == "positive":
            template = random.choice(self.positive_templates)
            hashtags = random.sample(self.positive_hashtags, random.randint(0, 2))
        elif sentiment == "negative":
            template = random.choice(self.negative_templates)
            hashtags = random.sample(self.negative_hashtags, random.randint(0, 2))
        else:
            template = random.choice(self.neutral_templates)
            hashtags = random.sample(self.neutral_hashtags, random.randint(0, 1))
        
        # Fill template with random entities
        entities = {
            "product": random.choice(self.products),
            "service": random.choice(self.services),
            "company": random.choice(self.companies),
            "experience": random.choice(self.experiences),
            "feature": "new feature"
        }
        
        # Generate tweet text - handle missing keys gracefully
        try:
            tweet_text = template.format(**entities)
        except KeyError as e:
            # If template has a key we don't have, use a fallback
            missing_key = str(e).strip("'")
            entities[missing_key] = random.choice(self.products)  # Use product as fallback
            tweet_text = template.format(**entities)
        
        # Add hashtags
        if hashtags:
            tweet_text += " " + " ".join(hashtags)
        
        # Extract mentions and hashtags
        mentions = [company for company in self.companies if company in tweet_text]
        hashtags_in_text = [tag for tag in hashtags]
        
        # Generate tweet metadata
        tweet = StreamedTweet(
            id=str(uuid.uuid4()),
            text=tweet_text,
            timestamp=datetime.utcnow().isoformat() + "Z",
            user_id=f"user_{random.randint(1, 10000)}",
            username=random.choice(self.usernames),
            hashtags=hashtags_in_text,
            mentions=mentions,
            retweet_count=random.randint(0, 1000),
            like_count=random.randint(0, 5000),
            language="en",
            location=random.choice(self.locations) if random.random() < 0.3 else None
        )
        
        # Update statistics
        self.total_generated += 1
        self.sentiment_counts[sentiment] += 1
        
        return tweet
    
    def _stream_worker(self):
        """Worker thread for generating streaming data."""
        interval = 60.0 / self.tweets_per_minute  # seconds between tweets
        
        logger.info(f"Starting stream worker with {self.tweets_per_minute} tweets/minute")
        
        while self.is_streaming:
            try:
                tweet = self._generate_tweet()
                
                # Add to queue (non-blocking)
                try:
                    self.data_queue.put(tweet, timeout=1.0)
                except:
                    logger.warning("Data queue is full, dropping tweet")
                
                # Wait for next tweet
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in stream worker: {str(e)}")
                time.sleep(1.0)
    
    def start_streaming(self):
        """Start the Twitter stream simulation."""
        if self.is_streaming:
            logger.warning("Stream is already running")
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        logger.info("Twitter stream simulation started")
    
    def stop_streaming(self):
        """Stop the Twitter stream simulation."""
        if not self.is_streaming:
            logger.warning("Stream is not running")
            return
        
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5.0)
        
        logger.info("Twitter stream simulation stopped")
    
    def get_tweets(self, max_count: int = 10) -> List[StreamedTweet]:
        """
        Get tweets from the stream queue.
        
        Args:
            max_count (int): Maximum number of tweets to retrieve
            
        Returns:
            List[StreamedTweet]: List of tweets
        """
        tweets = []
        
        for _ in range(max_count):
            try:
                tweet = self.data_queue.get_nowait()
                tweets.append(tweet)
            except Empty:
                break
        
        return tweets
    
    def get_stream_stats(self) -> Dict:
        """Get streaming statistics."""
        return {
            "is_streaming": self.is_streaming,
            "tweets_per_minute": self.tweets_per_minute,
            "total_generated": self.total_generated,
            "sentiment_distribution": self.sentiment_counts.copy(),
            "queue_size": self.data_queue.qsize(),
            "queue_max_size": self.data_queue.maxsize
        }

class RealTimeProcessor:
    """
    Real-time data processor that consumes streaming data and applies sentiment analysis.
    """
    
    def __init__(self, sentiment_analyzer, batch_size: int = 10, processing_interval: float = 5.0):
        """
        Initialize the real-time processor.
        
        Args:
            sentiment_analyzer: Sentiment analysis model instance
            batch_size (int): Number of tweets to process in each batch
            processing_interval (float): Seconds between processing batches
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        
        self.is_processing = False
        self.processing_thread = None
        
        # Data storage
        self.processed_tweets = deque(maxlen=1000)  # Keep last 1000 processed tweets
        self.processing_stats = {
            "total_processed": 0,
            "processing_rate": 0.0,
            "average_confidence": 0.0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
        }
        
        # Stream source
        self.stream_source = None
        
    def set_stream_source(self, stream_source):
        """Set the data stream source."""
        self.stream_source = stream_source
    
    def _processing_worker(self):
        """Worker thread for processing streaming data."""
        logger.info("Starting real-time processing worker")
        
        while self.is_processing:
            try:
                if not self.stream_source:
                    time.sleep(self.processing_interval)
                    continue
                
                # Get batch of tweets from stream
                tweets = self.stream_source.get_tweets(self.batch_size)
                
                if not tweets:
                    time.sleep(self.processing_interval)
                    continue
                
                # Extract texts for sentiment analysis
                texts = [tweet.text for tweet in tweets]
                
                # Process sentiment analysis
                start_time = time.time()
                sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
                processing_time = time.time() - start_time
                
                # Combine tweets with sentiment results
                processed_batch = []
                total_confidence = 0
                
                for tweet, sentiment_result in zip(tweets, sentiment_results):
                    processed_tweet = {
                        "tweet_data": tweet.to_dict(),
                        "sentiment_analysis": sentiment_result,
                        "processing_timestamp": datetime.utcnow().isoformat(),
                        "processing_time_ms": round((processing_time / len(tweets)) * 1000, 2)
                    }
                    
                    processed_batch.append(processed_tweet)
                    self.processed_tweets.append(processed_tweet)
                    
                    # Update statistics
                    sentiment = sentiment_result["sentiment"]
                    confidence = sentiment_result["confidence"]
                    
                    self.processing_stats["sentiment_distribution"][sentiment] += 1
                    total_confidence += confidence
                
                # Update processing statistics
                self.processing_stats["total_processed"] += len(tweets)
                self.processing_stats["processing_rate"] = len(tweets) / processing_time if processing_time > 0 else 0
                
                if len(tweets) > 0:
                    avg_confidence = total_confidence / len(tweets)
                    # Moving average for overall confidence
                    if self.processing_stats["average_confidence"] == 0:
                        self.processing_stats["average_confidence"] = avg_confidence
                    else:
                        self.processing_stats["average_confidence"] = (
                            self.processing_stats["average_confidence"] * 0.9 + avg_confidence * 0.1
                        )
                
                logger.info(f"Processed batch of {len(tweets)} tweets in {processing_time:.2f}s")
                
                # Wait before next batch
                time.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in processing worker: {str(e)}")
                time.sleep(self.processing_interval)
    
    def start_processing(self):
        """Start real-time processing."""
        if self.is_processing:
            logger.warning("Processing is already running")
            return
        
        if not self.stream_source:
            logger.error("No stream source set")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        logger.info("Real-time processing started")
    
    def stop_processing(self):
        """Stop real-time processing."""
        if not self.is_processing:
            logger.warning("Processing is not running")
            return
        
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Real-time processing stopped")
    
    def get_recent_tweets(self, count: int = 50) -> List[Dict]:
        """
        Get recently processed tweets.
        
        Args:
            count (int): Number of recent tweets to return
            
        Returns:
            List[Dict]: List of processed tweets
        """
        return list(self.processed_tweets)[-count:]
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        stats["is_processing"] = self.is_processing
        stats["queue_size"] = len(self.processed_tweets)
        stats["processing_interval"] = self.processing_interval
        stats["batch_size"] = self.batch_size
        
        return stats
    
    def get_sentiment_trends(self, time_window_minutes: int = 60) -> Dict:
        """
        Get sentiment trends over a time window.
        
        Args:
            time_window_minutes (int): Time window in minutes
            
        Returns:
            Dict: Sentiment trends data
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        recent_tweets = []
        for tweet in self.processed_tweets:
            tweet_time = datetime.fromisoformat(tweet["processing_timestamp"].replace("Z", "+00:00"))
            if tweet_time >= cutoff_time:
                recent_tweets.append(tweet)
        
        # Calculate trends
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_confidence = 0
        
        for tweet in recent_tweets:
            sentiment = tweet["sentiment_analysis"]["sentiment"]
            confidence = tweet["sentiment_analysis"]["confidence"]
            
            sentiment_counts[sentiment] += 1
            total_confidence += confidence
        
        total_tweets = len(recent_tweets)
        avg_confidence = total_confidence / total_tweets if total_tweets > 0 else 0
        
        # Calculate percentages
        sentiment_percentages = {}
        for sentiment, count in sentiment_counts.items():
            sentiment_percentages[sentiment] = (count / total_tweets * 100) if total_tweets > 0 else 0
        
        return {
            "time_window_minutes": time_window_minutes,
            "total_tweets": total_tweets,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "average_confidence": round(avg_confidence, 4),
            "timestamp": datetime.utcnow().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the streaming system
    from src.models.sentiment_model import create_sentiment_analyzer
    
    print("=== Testing Real-time Data Streaming System ===")
    
    # Create components
    stream_simulator = TwitterStreamSimulator(tweets_per_minute=60)  # 1 tweet per second
    sentiment_analyzer = create_sentiment_analyzer("twitter-roberta")
    processor = RealTimeProcessor(sentiment_analyzer, batch_size=5, processing_interval=3.0)
    
    # Connect components
    processor.set_stream_source(stream_simulator)
    
    # Start streaming and processing
    stream_simulator.start_streaming()
    processor.start_processing()
    
    try:
        # Let it run for a bit
        print("Running for 30 seconds...")
        time.sleep(30)
        
        # Get statistics
        stream_stats = stream_simulator.get_stream_stats()
        processing_stats = processor.get_processing_stats()
        trends = processor.get_sentiment_trends(time_window_minutes=5)
        
        print("\n=== Stream Statistics ===")
        for key, value in stream_stats.items():
            print(f"{key}: {value}")
        
        print("\n=== Processing Statistics ===")
        for key, value in processing_stats.items():
            print(f"{key}: {value}")
        
        print("\n=== Sentiment Trends (Last 5 minutes) ===")
        for key, value in trends.items():
            print(f"{key}: {value}")
        
        # Get recent tweets
        recent_tweets = processor.get_recent_tweets(5)
        print(f"\n=== Recent Processed Tweets ({len(recent_tweets)}) ===")
        for i, tweet in enumerate(recent_tweets[-3:], 1):
            print(f"{i}. {tweet['tweet_data']['text']}")
            print(f"   Sentiment: {tweet['sentiment_analysis']['sentiment']} (confidence: {tweet['sentiment_analysis']['confidence']})")
            print()
        
    finally:
        # Stop components
        processor.stop_processing()
        stream_simulator.stop_streaming()
        print("System stopped.")

