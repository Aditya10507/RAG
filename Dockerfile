FROM python:3.11-slim

# Create non-root user with UID 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR $HOME/app

# Copy requirements first for better Docker layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application
COPY --chown=user . .

# Expose the port HF Spaces expects
EXPOSE 7860

# Run with gunicorn for production-grade serving
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
