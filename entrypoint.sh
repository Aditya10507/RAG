#!/bin/bash
set -e

# Ensure data/ directory exists (HF Spaces removes it since it's in .gitignore)
mkdir -p data

# If there's a PDF in /data and no index yet, build the index
if [ -n "$(find data -maxdepth 1 -name '*.pdf' -print -quit)" ] && [ ! -f "db/index.faiss" ]; then
    echo "PDF found in data/ - building FAISS index..."
    python ingest.py
    echo "Index built successfully"
elif [ ! -f "db/index.faiss" ]; then
    echo "No index found. Upload a PDF to the 'data' folder through the HF Files tab, then restart the Space."
fi

# Start the web server
echo "Starting AI Assistant..."
exec gunicorn --bind 0.0.0.0:7860 --workers 2 --timeout 120 app:app
