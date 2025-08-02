"""
Configuration settings for the sentiment analysis Flask application.
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Application settings
    APP_NAME = 'Sentiment Analysis Platform'
    APP_VERSION = '1.0.0'
    
    # CORS settings
    CORS_ORIGINS = ['*']  # Allow all origins for now
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Model settings
    MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    MODEL_CACHE_DIR = './model_cache'
    
    # Rate limiting (if implemented)
    RATELIMIT_STORAGE_URL = 'memory://'
    
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # Use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Stricter CORS in production (customize as needed)
    # CORS_ORIGINS = ['https://yourdomain.com']

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

