#!/bin/bash
set -e

APP_STORAGE_DIR="${APP_STORAGE_DIR:-.}"
DATA_DIR="${DATA_DIR:-${APP_STORAGE_DIR}/data}"
DB_DIR="${DB_DIR:-${APP_STORAGE_DIR}/db}"
export APP_STORAGE_DIR DATA_DIR DB_DIR

# Ensure storage directories exist.
mkdir -p "$DATA_DIR" "$DB_DIR"

# If there are stored PDFs and no document memory yet, build it at startup.
if [ -n "$(find "$DATA_DIR" -maxdepth 1 -name '*.pdf' -print -quit)" ] && [ ! -f "$DB_DIR/index.faiss" ]; then
    echo "PDF found in $DATA_DIR - building document memory..."
    python ingest.py
    echo "Document memory built successfully"
elif [ ! -f "$DB_DIR/index.faiss" ]; then
    echo "No document memory found yet. General chat will work; upload PDFs in the app for document answers."
fi

# Start the web server
echo "Starting AI Assistant..."
exec gunicorn --bind "0.0.0.0:${PORT:-7860}" --workers 1 --threads 2 --timeout 180 app:app
