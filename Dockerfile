FROM python:3.11-slim

# Create non-root user with UID 1000 (required by HF Spaces)
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    ORT_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR $HOME/app

# Copy requirements first for better Docker layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache compact ONNX retrieval models during the build so runtime startup does
# not depend on downloading model weights.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; list(TextEmbedding(model_name='BAAI/bge-small-en-v1.5', threads=1).embed(['warmup'])); list(TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2', threads=1).rerank('warmup', ['warmup']))"

# Copy the rest of the application
COPY --chown=user . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

RUN chown -R user:user /home/user
USER user

# Render supplies PORT at runtime; 7860 remains the local/HF fallback.
EXPOSE 7860

# Run entrypoint that auto-builds the index if needed, then starts the server
CMD ["./entrypoint.sh"]
