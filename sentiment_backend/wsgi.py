"""
WSGI entry point for the sentiment analysis application.
This file is used by Gunicorn to serve the Flask application.
"""

from app import app

if __name__ == "__main__":
    app.run()

