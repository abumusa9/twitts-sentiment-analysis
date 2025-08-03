import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from src.routes.user import user_bp
from src.routes.sentiment_api import sentiment_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app)

from src.routes.streaming_api import streaming_bp

app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(sentiment_bp)
app.register_blueprint(streaming_bp)

# uncomment if you need to use database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging
import threading
import time
import random
from collections import deque
from typing import Dict, List, Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    APP_NAME = 'Sentiment Analysis Platform'
    APP_VERSION = '1.0.0'
    CORS_ORIGINS = ['*']
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# Sentiment Analyzer
class SentimentAnalyzer:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "simple-rule-based"
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self):
        try:
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
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
            
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
        text_lower = text.lower()
        
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
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
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
        results = []
        for i, text in enumerate(texts):
            result = self.analyze_text(text)
            result['index'] = i
            result['text'] = text
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Union[str, bool]]:
        return {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'model_type': 'transformer' if self.is_loaded else 'rule-based',
            'status': 'ready' if self.is_loaded else 'fallback'
        }

# Streaming Processor
class StreamingProcessor:
    def __init__(self, analyzer_instance: SentimentAnalyzer):
        self.analyzer = analyzer_instance
        self.stream_active = False
        self.processing_active = False
        self.tweets_per_minute = 60
        self.batch_size = 10
        self.processing_interval = 2.0

        self.generated_tweets_count = 0
        self.processed_tweets_count = 0
        self.sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        self.recent_tweets = deque(maxlen=50)
        self.sentiment_trends = deque(maxlen=60)
        self.confidence_trends = deque(maxlen=60)

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
                self._process_single_tweet(tweet)
            time.sleep(60 / self.tweets_per_minute)
        logger.info("Stream generator stopped.")

    def _process_single_tweet(self, tweet: Dict):
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
            self.recent_tweets.appendleft(processed_tweet)

    def _processing_loop(self):
        logger.info("Processing loop started.")
        while self.processing_active:
            time.sleep(self.processing_interval)
        logger.info("Processing loop stopped.")

    def _trend_aggregator(self):
        logger.info("Trend aggregator started.")
        while self.stream_active or self.processing_active:
            time.sleep(60)
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

# Initialize global instances
analyzer = SentimentAnalyzer()
streaming_processor = StreamingProcessor(analyzer)

# Flask App
app = Flask(__name__, static_folder='static', static_url_path='')
app.config.from_object(Config)
CORS(app, origins=app.config['CORS_ORIGINS'])

@app.route('/')
def index():
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
    return jsonify(analyzer.get_model_info())

@app.route('/api/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
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
        
        max_batch_size = 100
        if len(texts) > max_batch_size:
            return jsonify({
                'error': 'Batch too large',
                'message': f'Maximum batch size is {max_batch_size}'
            }), 400
        
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

@app.route('/api/streaming/start', methods=['POST'])
def start_streaming():
    try:
        data = request.get_json() or {}
        tweets_per_minute = data.get('tweets_per_minute', 60)
        batch_size = data.get('batch_size', 10)
        processing_interval = data.get('processing_interval', 2.0)
        
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
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(max(limit, 1), 100)
        
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
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(max(limit, 1), 100)
        
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
    def generate():
        while True:
            try:
                status = streaming_processor.get_status()
                if not status['stream_active']:
                    break
                
                data = {
                    'processed_count': status['processed_tweets_count'],
                    'sentiment_distribution': status['current_sentiment_distribution'],
                    'recent_tweets': status['recent_tweets'][:5],
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Live feed error: {e}")
                break
    
    return Response(generate(), mimetype='text/plain')

@app.errorhandler(404)
def not_found(error):
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
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'Request too large',
        'message': 'The request payload is too large'
    }), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Starting Sentiment Analysis Platform on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
    app.run(host='0.0.0.0', port=5000, debug=True)
