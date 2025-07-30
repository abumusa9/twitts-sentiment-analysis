import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask
from flask_cors import CORS
from src.routes.sentiment_api import sentiment_bp
from src.routes.streaming_api import streaming_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test_key'
CORS(app)
app.register_blueprint(sentiment_bp)
app.register_blueprint(streaming_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

