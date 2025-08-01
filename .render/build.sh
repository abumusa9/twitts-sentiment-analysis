#!/bin/bash

# Exit on errors
set -e

echo "Installing frontend dependencies..."
cd sentiment_dashboard
npm install

echo "Building React app..."
npm run build

echo "Moving frontend build to backend static folder..."
rm -rf ../sentiment_backend/src/static
mv build ../sentiment_backend/src/static

echo "Installing backend dependencies..."
cd ../sentiment_backend
pip install -r requirements.txt

