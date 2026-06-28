---
title: RAG AI Assistant
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Personal AI Assistant (RAG + Groq API)

An AI-powered personal assistant built with a Retrieval-Augmented Generation
(RAG) pipeline. It reads your PDFs, retrieves relevant context with hybrid
search, and generates answers through the Groq API.

## Features

- Document understanding from PDFs with recursive semantic chunking.
- PDF upload from the web UI into the `data/` folder.
- Search index rebuild from the web UI after adding documents.
- Hybrid retrieval with dense FAISS search plus sparse BM25 search.
- Reciprocal Rank Fusion (RRF) to combine dense and keyword rankings.
- Cross-encoder re-ranking before context is sent to the LLM.
- Conversational memory for recent turns.
- Source tracking when the index contains file/page metadata.
- Groq API generation with configurable model via `GROQ_MODEL`.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | Groq API, default `qwen/qwen3.6-27b` |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Sparse Retrieval | BM25 (`rank-bm25`) |
| Dense Retrieval | FAISS |
| Re-Ranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| Fusion Strategy | Reciprocal Rank Fusion (RRF) |
| Web Framework | Flask + Gunicorn |

## RAG Pipeline

```text
User Query
  -> Hybrid Retrieval
     -> Dense FAISS semantic search
     -> Sparse BM25 keyword search
     -> RRF fusion
  -> Cross-Encoder Re-Ranking
  -> Context Assembly with source metadata
  -> Groq API Generation
  -> Answer + source citations
```

## Project Structure

```text
rag-assistant/
|-- app.py              # Flask web server
|-- query.py            # RAG pipeline and Groq generation
|-- ingest.py           # PDF processing, chunking, embeddings, FAISS index
|-- templates/
|   `-- index.html      # Chat UI
|-- Dockerfile          # Hugging Face Spaces container definition
|-- entrypoint.sh       # Container startup script
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Groq

Create a `.env` file from `.env.example` and add your Groq API key:

```bash
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=qwen/qwen3.6-27b
PORT=7860
```

You can get a Groq API key from the Groq Console.

### 4. Add documents

You can add PDFs in either of these ways:

- Start the app and use the upload button in the header.
- Manually create a `data/` directory and place PDF files inside it.

Uploaded files are saved to `data/`.

### 5. Build or rebuild the search index

After adding PDFs, use the rebuild index button in the web app header.

You can also run the same ingestion process manually:

```bash
python ingest.py
```

This chunks your PDFs, generates embeddings, and builds the FAISS index in
`db/`.

### 6. Run the assistant

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | none | Groq API key used for LLM generation |
| `GROQ_MODEL` | No | `qwen/qwen3.6-27b` | Groq model ID |
| `PORT` | No | `7860` | Flask server port |
| `MAX_UPLOAD_MB` | No | `25` | Maximum PDF upload size in MB |

## Deploy to Hugging Face Spaces

1. Push this repo to GitHub.
2. Create a new Hugging Face Space with Docker as the SDK.
3. Connect your GitHub repo.
4. Add `GROQ_API_KEY` as a Space secret.
5. Upload PDFs to the Space's persistent storage or include an ingestion step.

The Docker image pre-downloads the embedding and re-ranking models, builds the
index when PDFs are present and no FAISS index exists, and starts the Flask app
with Gunicorn.

## Notes

- The LLM depends on Groq API for answer generation.
- Retrieval still uses local FAISS, BM25, sentence-transformer embeddings, and
  cross-encoder re-ranking.
- If you regenerate `db/chunks.json` with the current `ingest.py`, answers can
  include source file and page citations.
