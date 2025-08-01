#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Build frontend if needed
if [ -d "sentiment_dashboard" ]; then
    cd sentiment_dashboard
    npm install
    npm run build
    # Copy built files to Flask static directory
    mkdir -p ../sentiment_backend/static
    cp -r dist/* ../sentiment_backend/static/ 2>/dev/null || true
    cd ..
fi