# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY sentiment_backend /app/sentiment_backend
COPY app.py /app/

# Copy and install requirements
COPY sentiment_backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and build React frontend
COPY sentiment_dashboard /app/sentiment_dashboard
WORKDIR /app/sentiment_dashboard
RUN npm install && npm run build

# Move build to Flask static folder
RUN mkdir -p /app/sentiment_backend/src/static && \
    cp -r dist/* /app/sentiment_backend/src/static

# Set workdir back to backend
WORKDIR /app

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Start Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
