"""
Sentiment Analysis API Routes
Provides REST endpoints for sentiment analysis functionality
"""
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import logging
import time
from datetime import datetime
from typing import Dict, List
import json

# Import our sentiment model
from src.models.sentiment_model import create_sentiment_analyzer, SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Blueprint
sentiment_bp = Blueprint('sentiment', __name__, url_prefix='/api/sentiment')

# Global sentiment analyzer instance
sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the global sentiment analyzer instance."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        logger.info("Initializing sentiment analyzer...")
        sentiment_analyzer = create_sentiment_analyzer("twitter-roberta")
        logger.info("Sentiment analyzer initialized successfully")
    return sentiment_analyzer

@sentiment_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint."""
    try:
        analyzer = get_sentiment_analyzer()
        model_info = analyzer.get_model_info()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_info': {
                'model_name': model_info['model_name'],
                'device': model_info['device'],
                'processed_count': model_info['processed_count']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@sentiment_bp.route('/analyze', methods=['POST'])
@cross_origin()
def analyze_sentiment():
    """
    Analyze sentiment for a single text or batch of texts.
    
    Request body:
    {
        "text": "single text to analyze"
    }
    OR
    {
        "texts": ["text1", "text2", "text3"]
    }
    
    Response:
    {
        "success": true,
        "result": {...} or "results": [...]
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Get analyzer
        analyzer = get_sentiment_analyzer()
        
        # Check if single text or batch
        if 'text' in data:
            # Single text analysis
            text = data['text']
            
            if not text or not isinstance(text, str):
                return jsonify({
                    'success': False,
                    'error': 'Text must be a non-empty string'
                }), 400
            
            start_time = time.time()
            result = analyzer.analyze_sentiment(text)
            processing_time = time.time() - start_time
            
            return jsonify({
                'success': True,
                'result': result,
                'processing_time_ms': round(processing_time * 1000, 2),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        elif 'texts' in data:
            # Batch analysis
            texts = data['texts']
            
            if not texts or not isinstance(texts, list):
                return jsonify({
                    'success': False,
                    'error': 'Texts must be a non-empty list'
                }), 400
            
            if len(texts) > 100:  # Limit batch size
                return jsonify({
                    'success': False,
                    'error': 'Batch size cannot exceed 100 texts'
                }), 400
            
            # Validate all texts
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    return jsonify({
                        'success': False,
                        'error': f'Text at index {i} must be a string'
                    }), 400
            
            start_time = time.time()
            results = analyzer.analyze_batch(texts)
            processing_time = time.time() - start_time
            
            # Calculate batch statistics
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_confidence = 0
            
            for result in results:
                sentiment_counts[result['sentiment']] += 1
                total_confidence += result['confidence']
            
            avg_confidence = total_confidence / len(results) if results else 0
            
            return jsonify({
                'success': True,
                'results': results,
                'batch_stats': {
                    'total_texts': len(texts),
                    'sentiment_distribution': sentiment_counts,
                    'average_confidence': round(avg_confidence, 4)
                },
                'processing_time_ms': round(processing_time * 1000, 2),
                'timestamp': datetime.utcnow().isoformat()
            }), 200
            
        else:
            return jsonify({
                'success': False,
                'error': 'Request must contain either "text" or "texts" field'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@sentiment_bp.route('/analyze/stream', methods=['POST'])
@cross_origin()
def analyze_sentiment_stream():
    """
    Analyze sentiment for streaming data (simulates real-time processing).
    
    Request body:
    {
        "data": [
            {"id": "tweet1", "text": "I love this!", "timestamp": "2024-01-01T12:00:00Z"},
            {"id": "tweet2", "text": "This is bad", "timestamp": "2024-01-01T12:01:00Z"}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({
                'success': False,
                'error': 'Request must contain "data" field with list of items'
            }), 400
        
        stream_data = data['data']
        
        if not isinstance(stream_data, list):
            return jsonify({
                'success': False,
                'error': 'Data must be a list'
            }), 400
        
        if len(stream_data) > 50:  # Limit for streaming
            return jsonify({
                'success': False,
                'error': 'Stream batch size cannot exceed 50 items'
            }), 400
        
        # Get analyzer
        analyzer = get_sentiment_analyzer()
        
        # Process streaming data
        results = []
        texts = []
        
        # Extract texts and validate data
        for i, item in enumerate(stream_data):
            if not isinstance(item, dict):
                return jsonify({
                    'success': False,
                    'error': f'Item at index {i} must be a dictionary'
                }), 400
            
            if 'text' not in item:
                return jsonify({
                    'success': False,
                    'error': f'Item at index {i} must contain "text" field'
                }), 400
            
            texts.append(item['text'])
        
        start_time = time.time()
        
        # Analyze sentiments
        sentiment_results = analyzer.analyze_batch(texts)
        
        # Combine with original data
        for i, (original_item, sentiment_result) in enumerate(zip(stream_data, sentiment_results)):
            result = {
                'id': original_item.get('id', f'item_{i}'),
                'original_text': original_item['text'],
                'timestamp': original_item.get('timestamp', datetime.utcnow().isoformat()),
                'sentiment_analysis': {
                    'sentiment': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence'],
                    'scores': sentiment_result['scores']
                },
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            results.append(result)
        
        processing_time = time.time() - start_time
        
        # Calculate stream statistics
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        high_confidence_count = 0
        
        for result in results:
            sentiment = result['sentiment_analysis']['sentiment']
            confidence = result['sentiment_analysis']['confidence']
            
            sentiment_counts[sentiment] += 1
            if confidence > 0.8:
                high_confidence_count += 1
        
        return jsonify({
            'success': True,
            'results': results,
            'stream_stats': {
                'total_items': len(results),
                'sentiment_distribution': sentiment_counts,
                'high_confidence_predictions': high_confidence_count,
                'processing_rate_items_per_second': round(len(results) / processing_time, 2) if processing_time > 0 else 0
            },
            'processing_time_ms': round(processing_time * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in stream sentiment analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@sentiment_bp.route('/model/info', methods=['GET'])
@cross_origin()
def get_model_info():
    """Get detailed information about the current sentiment analysis model."""
    try:
        analyzer = get_sentiment_analyzer()
        model_info = analyzer.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@sentiment_bp.route('/model/switch', methods=['POST'])
@cross_origin()
def switch_model():
    """
    Switch to a different sentiment analysis model.
    
    Request body:
    {
        "model_type": "twitter-roberta" | "bertweet" | "multilingual-bert"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'model_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Request must contain "model_type" field'
            }), 400
        
        model_type = data['model_type']
        valid_models = ["twitter-roberta", "bertweet", "multilingual-bert", "twitter-xlm-roberta"]
        
        if model_type not in valid_models:
            return jsonify({
                'success': False,
                'error': f'Invalid model type. Valid options: {valid_models}'
            }), 400
        
        # Create new analyzer with specified model
        global sentiment_analyzer
        logger.info(f"Switching to model: {model_type}")
        
        start_time = time.time()
        sentiment_analyzer = create_sentiment_analyzer(model_type)
        load_time = time.time() - start_time
        
        model_info = sentiment_analyzer.get_model_info()
        
        return jsonify({
            'success': True,
            'message': f'Successfully switched to {model_type}',
            'model_info': model_info,
            'load_time_ms': round(load_time * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@sentiment_bp.route('/stats', methods=['GET'])
@cross_origin()
def get_stats():
    """Get processing statistics."""
    try:
        analyzer = get_sentiment_analyzer()
        model_info = analyzer.get_model_info()
        
        stats = {
            'total_processed': model_info['processed_count'],
            'model_name': model_info['model_name'],
            'device': model_info['device'],
            'model_size_mb': model_info['model_size_mb'],
            'uptime_info': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Error handlers
@sentiment_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@sentiment_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'timestamp': datetime.utcnow().isoformat()
    }), 405

@sentiment_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

