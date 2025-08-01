#!/bin/bash

# Exit if any command fails
set -e

echo "Installing frontend dependencies..."
cd sentiment_dashboard
npm install

echo "Building React frontend..."
npm run build

echo "Copying frontend build to Flask static folder..."
rm -rf ../sentiment_backend/src/static
mkdir -p ../sentiment_backend/src/static
cp -r dist/* ../sentiment_backend/src/static/


echo "Installing backend dependencies..."
cd ../
pip install -r requirements.txt
