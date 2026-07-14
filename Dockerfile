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
# The Space runs on CPU hardware. Install the CPU wheel explicitly before
# sentence-transformers so pip does not pull multi-gigabyte CUDA libraries.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers and cross-encoder models so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the rest of the application
COPY --chown=user . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose the port HF Spaces expects
EXPOSE 7860

# Run entrypoint that auto-builds the index if needed, then starts the server
CMD ["./entrypoint.sh"]
