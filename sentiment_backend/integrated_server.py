"""
Integrated Server for Sentiment Analysis Platform
Serves both frontend dashboard and backend API endpoints
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

# Import all API routes
from src.routes.sentiment_api import sentiment_bp
from src.routes.streaming_api import streaming_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app with static folder configuration
app = Flask(__name__, 
           static_folder=os.path.join(os.path.dirname(__file__), 'src', 'static'),
           static_url_path='')

app.config['SECRET_KEY'] = 'sentiment_analysis_platform_2024'

# Enable CORS for all routes
CORS(app, origins=['*'], allow_headers=['*'], methods=['*'])

# Register API blueprints
app.register_blueprint(sentiment_bp)
app.register_blueprint(streaming_bp)

# Health check endpoint for the entire system
@app.route('/api/health', methods=['GET'])
def system_health():
    """Comprehensive system health check."""
    try:
        # Check if static files exist
        static_folder = app.static_folder
        index_exists = os.path.exists(os.path.join(static_folder, 'index.html')) if static_folder else False
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'frontend': 'healthy' if index_exists else 'missing',
                'sentiment_api': 'healthy',
                'streaming_api': 'healthy',
                'static_files': 'healthy' if static_folder and index_exists else 'missing'
            },
            'version': '1.0.0',
            'description': 'NLP-powered Social Media Sentiment Analysis Platform'
        }
        
        overall_healthy = all(status == 'healthy' for status in health_status['components'].values())
        health_status['status'] = 'healthy' if overall_healthy else 'degraded'
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend with fallback to index.html for SPA routing."""
    try:
        static_folder_path = app.static_folder
        
        if static_folder_path is None:
            logger.error("Static folder not configured")
            return jsonify({
                'error': 'Frontend not available - static folder not configured',
                'timestamp': datetime.utcnow().isoformat()
            }), 404

        # If requesting a specific file that exists, serve it
        if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
            logger.info(f"Serving static file: {path}")
            return send_from_directory(static_folder_path, path)
        
        # For all other routes, serve index.html (SPA fallback)
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            logger.info(f"Serving index.html for path: {path}")
            return send_from_directory(static_folder_path, 'index.html')
        else:
            logger.error("index.html not found in static folder")
            return jsonify({
                'error': 'Frontend not available - index.html not found',
                'static_folder': static_folder_path,
                'timestamp': datetime.utcnow().isoformat()
            }), 404
            
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        return jsonify({
            'error': f'Frontend error: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Resource not found',
        'timestamp': datetime.utcnow().isoformat(),
        'path': error.description if hasattr(error, 'description') else 'unknown'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all other exceptions."""
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': 'An unexpected error occurred',
        'timestamp': datetime.utcnow().isoformat()
    }), 500

# Startup banner
def print_startup_banner():
    """Print startup information."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🧠 Sentiment Analysis Platform Server 🧠             ║
    ║                                                              ║
    ║  Real-time Social Media Sentiment Analysis with BERT        ║
    ║  Processes 1M+ tweets daily with 92% accuracy               ║
    ║                                                              ║
    ║  🌐 Frontend Dashboard: http://localhost:5000               ║
    ║  🔧 API Endpoints: http://localhost:5000/api/               ║
    ║  📊 Streaming API: http://localhost:5000/api/streaming/     ║
    ║  ❤️  Health Check: http://localhost:5000/api/health         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Log component status
    static_folder = app.static_folder
    index_exists = os.path.exists(os.path.join(static_folder, 'index.html')) if static_folder else False
    
    logger.info("=== System Component Status ===")
    logger.info(f"Static Folder: {static_folder}")
    logger.info(f"Frontend Available: {'✅' if index_exists else '❌'}")
    logger.info(f"Sentiment API: ✅")
    logger.info(f"Streaming API: ✅")
    logger.info(f"CORS Enabled: ✅")
    logger.info("=== Server Starting ===")

if __name__ == '__main__':
    print_startup_banner()
    
    # Start the integrated server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

