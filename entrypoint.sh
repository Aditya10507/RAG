#!/bin/bash
set -e

# If there's a PDF in /data and no index yet, build the index
if [ -d "data" ] && [ -n "$(find data -maxdepth 1 -name '*.pdf' -print -quit)" ] && [ ! -f "db/index.faiss" ]; then
    echo "PDF found in data/ — building FAISS index..."
    python ingest.py
    echo "Index built successfully"
elif [ ! -f "db/index.faiss" ]; then
    echo "No index found. Upload a PDF to the 'data' folder and the Space will rebuild on restart."
fi

# Start the web server
echo "Starting AI Assistant..."
exec gunicorn --bind 0.0.0.0:7860 --workers 2 --timeout 120 app:app
