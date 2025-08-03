# flyio_dockerfile content
# Use a lightweight Python base image
FROM python:3.11-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies if any (e.g., for Pillow, if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
# This step is crucial for Docker layer caching. If requirements.txt doesn't change,
# this layer will be cached, speeding up subsequent builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Change to the backend source directory
WORKDIR $APP_HOME/sentiment_backend/src

# Command to run the application using Gunicorn
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080"]