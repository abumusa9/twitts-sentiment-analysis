"""
Streaming API Routes for Real-time Data Processing
Provides endpoints for managing and monitoring real-time sentiment analysis
"""
from flask import Blueprint, request, jsonify, Response
from flask_cors import cross_origin
import logging
import json
import time
from datetime import datetime
from typing import Dict, List
import threading

# Import streaming components
from src.models.data_stream import TwitterStreamSimulator, RealTimeProcessor
from src.models.sentiment_model import create_sentiment_analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
streaming_bp = Blueprint('streaming', __name__, url_prefix='/api/streaming')

# Global instances
stream_simulator = None
real_time_processor = None
sentiment_analyzer = None

def get_streaming_components():
    """Get or create global streaming components."""
    global stream_simulator, real_time_processor, sentiment_analyzer
    
    if sentiment_analyzer is None:
        logger.info("Initializing sentiment analyzer for streaming...")
        sentiment_analyzer = create_sentiment_analyzer("twitter-roberta")
    
    if stream_simulator is None:
        logger.info("Initializing Twitter stream simulator...")
        stream_simulator = TwitterStreamSimulator(tweets_per_minute=120)  # 2 tweets per second
    
    if real_time_processor is None:
        logger.info("Initializing real-time processor...")
        real_time_processor = RealTimeProcessor(
            sentiment_analyzer=sentiment_analyzer,
            batch_size=10,
            processing_interval=2.0
        )
        real_time_processor.set_stream_source(stream_simulator)
    
    return stream_simulator, real_time_processor, sentiment_analyzer

@streaming_bp.route('/status', methods=['GET'])
@cross_origin()
def get_streaming_status():
    """Get the current status of the streaming system."""
    try:
        simulator, processor, analyzer = get_streaming_components()
        
        stream_stats = simulator.get_stream_stats()
        processing_stats = processor.get_processing_stats()
        
        return jsonify({
            'success': True,
            'streaming_status': {
                'stream_active': stream_stats['is_streaming'],
                'processing_active': processing_stats['is_processing'],
                'tweets_per_minute': stream_stats['tweets_per_minute'],
                'total_generated': stream_stats['total_generated'],
                'total_processed': processing_stats['total_processed'],
                'queue_size': stream_stats['queue_size'],
                'processing_rate': processing_stats['processing_rate']
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting streaming status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@streaming_bp.route('/start', methods=['POST'])
@cross_origin()
def start_streaming():
    """Start the real-time streaming and processing system."""
    try:
        data = request.get_json() or {}
        
        # Get optional parameters
        tweets_per_minute = data.get('tweets_per_minute', 120)
        batch_size = data.get('batch_size', 10)
        processing_interval = data.get('processing_interval', 2.0)
        
        # Validate parameters
        if tweets_per_minute < 1 or tweets_per_minute > 1000:
            return jsonify({
                'success': False,
                'error': 'tweets_per_minute must be between 1 and 1000'
            }), 400
        
        if batch_size < 1 or batch_size > 100:
            return jsonify({
                'success': False,
                'error': 'batch_size must be between 1 and 100'
            }), 400
        
        if processing_interval < 0.1 or processing_interval > 60:
            return jsonify({
                'success': False,
                'error': 'processing_interval must be between 0.1 and 60 seconds'
            }), 400
        
        # Get components
        simulator, processor, analyzer = get_streaming_components()
        
        # Update configuration if needed
        if simulator.tweets_per_minute != tweets_per_minute:
            # Stop current simulator if running
            if simulator.is_streaming:
                simulator.stop_streaming()
            
            # Create new simulator with updated rate
            global stream_simulator
            stream_simulator = TwitterStreamSimulator(tweets_per_minute=tweets_per_minute)
            simulator = stream_simulator
            processor.set_stream_source(simulator)
        
        # Update processor configuration
        processor.batch_size = batch_size
        processor.processing_interval = processing_interval
        
        # Start components
        if not simulator.is_streaming:
            simulator.start_streaming()
        
        if not processor.is_processing:
            processor.start_processing()
        
        return jsonify({
            'success': True,
            'message': 'Streaming system started successfully',
            'configuration': {
                'tweets_per_minute': tweets_per_minute,
                'batch_size': batch_size,
                'processing_interval': processing_interval
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error starting streaming: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@streaming_bp.route('/stop', methods=['POST'])
@cross_origin()
def stop_streaming():
    """Stop the real-time streaming and processing system."""
    try:
        simulator, processor, analyzer = get_streaming_components()
        
        # Stop components
        if processor.is_processing:
            processor.stop_processing()
        
        if simulator.is_streaming:
            simulator.stop_streaming()
        
        return jsonify({
            'success': True,
            'message': 'Streaming system stopped successfully',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error stopping streaming: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@streaming_bp.route('/stats', methods=['GET'])
@cross_origin()
def get_streaming_stats():
    """Get detailed streaming statistics."""
    try:
        simulator, processor, analyzer = get_streaming_components()
        
        stream_stats = simulator.get_stream_stats()
        processing_stats = processor.get_processing_stats()
        
        return jsonify({
            'success': True,
            'statistics': {
                'stream_stats': stream_stats,
                'processing_stats': processing_stats,
                'system_info': {
                    'model_name': analyzer.model_name,
                    'device': str(analyzer.device),
                    'total_model_processed': analyzer.processed_count
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting streaming stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@streaming_bp.route('/recent-tweets', methods=['GET'])
@cross_origin()
def get_recent_tweets():
    """Get recently processed tweets."""
    try:
        count = request.args.get('count', 50, type=int)
        
        if count < 1 or count > 500:
            return jsonify({
                'success': False,
                'error': 'count must be between 1 and 500'
            }), 400
        
        simulator, processor, analyzer = get_streaming_components()
        
        recent_tweets = processor.get_recent_tweets(count)
        
        return jsonify({
            'success': True,
            'tweets': recent_tweets,
            'count': len(recent_tweets),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting recent tweets: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@streaming_bp.route('/trends', methods=['GET'])
@cross_origin()
def get_sentiment_trends():
    """Get sentiment trends over a time window."""
    try:
        time_window = request.args.get('time_window_minutes', 60, type=int)
        
        if time_window < 1 or time_window > 1440:  # Max 24 hours
            return jsonify({
                'success': False,
                'error': 'time_window_minutes must be between 1 and 1440'
            }), 400
        
        simulator, processor, analyzer = get_streaming_components()
        
        trends = processor.get_sentiment_trends(time_window_minutes=time_window)
        
        return jsonify({
            'success': True,
            'trends': trends,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting sentiment trends: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@streaming_bp.route('/live-feed', methods=['GET'])
@cross_origin()
def live_feed():
    """Server-Sent Events endpoint for live tweet feed."""
    def generate_events():
        """Generate server-sent events for live updates."""
        try:
            simulator, processor, analyzer = get_streaming_components()
            
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connection', 'message': 'Connected to live feed', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            
            last_processed_count = 0
            
            while True:
                try:
                    # Get recent tweets
                    recent_tweets = processor.get_recent_tweets(10)
                    current_processed_count = processor.processing_stats['total_processed']
                    
                    # Send new tweets if any
                    if current_processed_count > last_processed_count and recent_tweets:
                        new_tweets = recent_tweets[-(current_processed_count - last_processed_count):]
                        
                        for tweet in new_tweets:
                            event_data = {
                                'type': 'new_tweet',
                                'data': tweet,
                                'timestamp': datetime.utcnow().isoformat()
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                        
                        last_processed_count = current_processed_count
                    
                    # Send periodic stats update
                    if int(time.time()) % 10 == 0:  # Every 10 seconds
                        stats = processor.get_processing_stats()
                        event_data = {
                            'type': 'stats_update',
                            'data': stats,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                    
                    time.sleep(1)  # Check every second
                    
                except Exception as e:
                    error_data = {
                        'type': 'error',
                        'message': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break
                    
        except Exception as e:
            logger.error(f"Error in live feed generator: {str(e)}")
            error_data = {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(
        generate_events(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )

@streaming_bp.route('/simulate-burst', methods=['POST'])
@cross_origin()
def simulate_burst():
    """Simulate a burst of tweets for testing high-volume scenarios."""
    try:
        data = request.get_json() or {}
        
        burst_count = data.get('burst_count', 50)
        burst_duration = data.get('burst_duration_seconds', 10)
        
        if burst_count < 1 or burst_count > 1000:
            return jsonify({
                'success': False,
                'error': 'burst_count must be between 1 and 1000'
            }), 400
        
        if burst_duration < 1 or burst_duration > 300:
            return jsonify({
                'success': False,
                'error': 'burst_duration_seconds must be between 1 and 300'
            }), 400
        
        simulator, processor, analyzer = get_streaming_components()
        
        # Calculate burst rate
        original_rate = simulator.tweets_per_minute
        burst_rate = int((burst_count / burst_duration) * 60)
        
        def burst_worker():
            """Worker function to handle burst simulation."""
            try:
                # Temporarily increase tweet rate
                simulator.tweets_per_minute = burst_rate
                logger.info(f"Starting burst simulation: {burst_count} tweets over {burst_duration}s")
                
                # Wait for burst duration
                time.sleep(burst_duration)
                
                # Restore original rate
                simulator.tweets_per_minute = original_rate
                logger.info("Burst simulation completed")
                
            except Exception as e:
                logger.error(f"Error in burst simulation: {str(e)}")
                simulator.tweets_per_minute = original_rate
        
        # Start burst in background thread
        burst_thread = threading.Thread(target=burst_worker, daemon=True)
        burst_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Burst simulation started: {burst_count} tweets over {burst_duration} seconds',
            'burst_config': {
                'burst_count': burst_count,
                'burst_duration_seconds': burst_duration,
                'burst_rate_per_minute': burst_rate,
                'original_rate_per_minute': original_rate
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error simulating burst: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@streaming_bp.route('/health', methods=['GET'])
@cross_origin()
def streaming_health_check():
    """Health check for streaming system."""
    try:
        simulator, processor, analyzer = get_streaming_components()
        
        health_status = {
            'stream_simulator': 'healthy' if simulator else 'not_initialized',
            'processor': 'healthy' if processor else 'not_initialized',
            'sentiment_analyzer': 'healthy' if analyzer else 'not_initialized',
            'stream_active': simulator.is_streaming if simulator else False,
            'processing_active': processor.is_processing if processor else False
        }
        
        overall_health = all(status == 'healthy' for status in [
            health_status['stream_simulator'],
            health_status['processor'],
            health_status['sentiment_analyzer']
        ])
        
        return jsonify({
            'success': True,
            'health': 'healthy' if overall_health else 'degraded',
            'components': health_status,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Streaming health check failed: {str(e)}")
        return jsonify({
            'success': False,
            'health': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Error handlers
@streaming_bp.errorhandler(404)
def streaming_not_found(error):
    return jsonify({
        'success': False,
        'error': 'Streaming endpoint not found',
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@streaming_bp.errorhandler(405)
def streaming_method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed for streaming endpoint',
        'timestamp': datetime.utcnow().isoformat()
    }), 405

@streaming_bp.errorhandler(500)
def streaming_internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error in streaming system',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

