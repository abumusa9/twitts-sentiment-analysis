from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging

# Import our custom modules
from models import get_analyzer, get_streaming_processor
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    """Application factory pattern."""
    app = Flask(__name__, static_folder='static', static_url_path='')
    
    # Load configuration
    config_name = config_name or os.environ.get('FLASK_ENV', 'production')
    app.config.from_object(config[config_name])
    
    # Initialize CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Initialize sentiment analyzer and streaming processor
    analyzer = get_analyzer()
    streaming_processor = get_streaming_processor()
    
    @app.route('/')
    def index():
        """Serve the main dashboard"""
        try:
            return send_from_directory(app.static_folder, 'index.html')
        except Exception as e:
            logger.error(f"Failed to serve index.html: {e}")
            return jsonify({
                'message': 'Sentiment Analysis Platform',
                'status': 'Frontend not built yet. Please build the React app first.',
                'api_endpoints': {
                    'health': '/api/health',
                    'analyze': '/api/sentiment/analyze',
                    'batch': '/api/sentiment/batch',
                    'model_info': '/api/model/info',
                    'streaming_start': '/api/streaming/start',
                    'streaming_stop': '/api/streaming/stop',
                    'streaming_status': '/api/streaming/status',
                    'streaming_recent': '/api/streaming/recent-tweets',
                    'streaming_trends': '/api/streaming/trends'
                }
            })

    @app.route('/api/health')
    def health():
        """Health check endpoint"""
        model_info = analyzer.get_model_info()
        streaming_status = streaming_processor.get_status()
        return jsonify({
            'status': 'healthy',
            'message': 'Sentiment Analysis Platform is running',
            'timestamp': datetime.utcnow().isoformat(),
            'version': app.config['APP_VERSION'],
            'model': model_info,
            'streaming': {
                'active': streaming_status['stream_active'],
                'processed_tweets': streaming_status['processed_tweets_count']
            }
        })

    @app.route('/api/model/info')
    def model_info():
        """Get model information"""
        return jsonify(analyzer.get_model_info())

    @app.route('/api/sentiment/analyze', methods=['POST'])
    def analyze_sentiment():
        """Analyze sentiment of provided text"""
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({
                    'error': 'No text provided',
                    'message': 'Please provide text in the request body'
                }), 400
            
            text = data.get('text', '').strip()
            if not text:
                return jsonify({
                    'error': 'Empty text provided',
                    'message': 'Text cannot be empty'
                }), 400
            
            # Analyze sentiment using our model
            result = analyzer.analyze_text(text)
            result.update({
                'text': text,
                'timestamp': datetime.utcnow().isoformat(),
                'success': True
            })
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return jsonify({
                'error': 'Analysis failed',
                'message': str(e),
                'success': False
            }), 500

    @app.route('/api/sentiment/batch', methods=['POST'])
    def analyze_batch():
        """Analyze sentiment for multiple texts"""
        try:
            data = request.get_json()
            if not data or 'texts' not in data:
                return jsonify({
                    'error': 'No texts provided',
                    'message': 'Please provide texts array in the request body'
                }), 400
            
            texts = data.get('texts', [])
            if not isinstance(texts, list) or len(texts) == 0:
                return jsonify({
                    'error': 'Invalid texts format',
                    'message': 'Texts must be a non-empty array'
                }), 400
            
            # Limit batch size
            max_batch_size = 100
            if len(texts) > max_batch_size:
                return jsonify({
                    'error': 'Batch too large',
                    'message': f'Maximum batch size is {max_batch_size}'
                }), 400
            
            # Analyze all texts
            results = analyzer.analyze_batch(texts)
            
            return jsonify({
                'results': results,
                'total_processed': len(results),
                'timestamp': datetime.utcnow().isoformat(),
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return jsonify({
                'error': 'Batch analysis failed',
                'message': str(e),
                'success': False
            }), 500

    # Streaming API endpoints
    @app.route('/api/streaming/start', methods=['POST'])
    def start_streaming():
        """Start the streaming system"""
        try:
            data = request.get_json() or {}
            tweets_per_minute = data.get('tweets_per_minute', 60)
            batch_size = data.get('batch_size', 10)
            processing_interval = data.get('processing_interval', 2.0)
            
            # Validate parameters
            if not (1 <= tweets_per_minute <= 1000):
                return jsonify({
                    'error': 'Invalid tweets_per_minute',
                    'message': 'tweets_per_minute must be between 1 and 1000'
                }), 400
            
            if not (1 <= batch_size <= 100):
                return jsonify({
                    'error': 'Invalid batch_size',
                    'message': 'batch_size must be between 1 and 100'
                }), 400
            
            if not (0.1 <= processing_interval <= 60):
                return jsonify({
                    'error': 'Invalid processing_interval',
                    'message': 'processing_interval must be between 0.1 and 60 seconds'
                }), 400
            
            success, message = streaming_processor.start_streaming(
                tweets_per_minute=tweets_per_minute,
                batch_size=batch_size,
                processing_interval=processing_interval
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': message,
                    'configuration': {
                        'tweets_per_minute': tweets_per_minute,
                        'batch_size': batch_size,
                        'processing_interval': processing_interval
                    },
                    'timestamp': datetime.utcnow().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to start streaming',
                    'message': message
                }), 400
                
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to start streaming',
                'message': str(e)
            }), 500

    @app.route('/api/streaming/stop', methods=['POST'])
    def stop_streaming():
        """Stop the streaming system"""
        try:
            success, message = streaming_processor.stop_streaming()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': message,
                    'timestamp': datetime.utcnow().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to stop streaming',
                    'message': message
                }), 400
                
        except Exception as e:
            logger.error(f"Failed to stop streaming: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to stop streaming',
                'message': str(e)
            }), 500

    @app.route('/api/streaming/status')
    def streaming_status():
        """Get streaming system status"""
        try:
            status = streaming_processor.get_status()
            return jsonify({
                'success': True,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to get streaming status: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get status',
                'message': str(e)
            }), 500

    @app.route('/api/streaming/recent-tweets')
    def get_recent_tweets():
        """Get recently processed tweets"""
        try:
            limit = request.args.get('limit', 20, type=int)
            limit = min(max(limit, 1), 100)  # Clamp between 1 and 100
            
            status = streaming_processor.get_status()
            recent_tweets = status['recent_tweets'][:limit]
            
            return jsonify({
                'success': True,
                'tweets': recent_tweets,
                'total_available': len(status['recent_tweets']),
                'limit': limit,
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to get recent tweets: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get recent tweets',
                'message': str(e)
            }), 500

    @app.route('/api/streaming/trends')
    def get_sentiment_trends():
        """Get sentiment trends over time"""
        try:
            limit = request.args.get('limit', 20, type=int)
            limit = min(max(limit, 1), 100)  # Clamp between 1 and 100
            
            status = streaming_processor.get_status()
            sentiment_trends = list(status['sentiment_trends'])[:limit]
            confidence_trends = list(status['confidence_trends'])[:limit]
            
            return jsonify({
                'success': True,
                'sentiment_trends': sentiment_trends,
                'confidence_trends': confidence_trends,
                'current_distribution': status['current_sentiment_distribution'],
                'total_processed': status['processed_tweets_count'],
                'limit': limit,
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to get trends: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get trends',
                'message': str(e)
            }), 500

    @app.route('/api/streaming/live-feed')
    def live_feed():
        """Server-sent events endpoint for live updates"""
        def generate():
            while True:
                try:
                    status = streaming_processor.get_status()
                    if not status['stream_active']:
                        break
                    
                    # Send current status as SSE
                    data = {
                        'processed_count': status['processed_tweets_count'],
                        'sentiment_distribution': status['current_sentiment_distribution'],
                        'recent_tweets': status['recent_tweets'][:5],  # Last 5 tweets
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # Wait before next update
                    import time
                    time.sleep(3)  # Update every 3 seconds
                    
                except Exception as e:
                    logger.error(f"Live feed error: {e}")
                    break
        
        return Response(generate(), mimetype='text/plain')

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found',
            'available_endpoints': [
                '/api/health',
                '/api/sentiment/analyze',
                '/api/sentiment/batch',
                '/api/model/info',
                '/api/streaming/start',
                '/api/streaming/stop',
                '/api/streaming/status',
                '/api/streaming/recent-tweets',
                '/api/streaming/trends',
                '/api/streaming/live-feed'
            ]
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle request too large errors"""
        return jsonify({
            'error': 'Request too large',
            'message': 'The request payload is too large'
        }), 413

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Starting Sentiment Analysis Platform on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

