import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.models.data_stream import TwitterStreamSimulator, RealTimeProcessor
from src.models.sentiment_model import create_sentiment_analyzer
import time

def test_streaming_system():
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
        print("Running for 15 seconds...")
        time.sleep(15)
        
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
        recent_tweets = processor.get_recent_tweets(3)
        print(f"\n=== Recent Processed Tweets ({len(recent_tweets)}) ===")
        for i, tweet in enumerate(recent_tweets, 1):
            print(f"{i}. {tweet['tweet_data']['text']}")
            print(f"   Sentiment: {tweet['sentiment_analysis']['sentiment']} (confidence: {tweet['sentiment_analysis']['confidence']})")
            print()
        
    finally:
        # Stop components
        processor.stop_processing()
        stream_simulator.stop_streaming()
        print("System stopped.")

if __name__ == "__main__":
    test_streaming_system()

