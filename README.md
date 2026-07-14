---
title: RAG AI Assistant
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Personal AI Assistant (RAG + Groq API)

[Live application](https://adi080122-local-rag-voice-ai-assistant.hf.space/)

An AI-powered personal assistant built with a Retrieval-Augmented Generation
(RAG) pipeline. It reads your PDFs, retrieves relevant context with hybrid
search, and generates answers through the Groq API.

## Features

- Document understanding from PDFs with recursive semantic chunking.
- Message-bound PDF attachments from the chat composer into app storage.
- Automatic document indexing after PDF upload.
- Hybrid retrieval with dense FAISS search plus sparse BM25 search.
- Reciprocal Rank Fusion (RRF) to combine dense and keyword rankings.
- Cross-encoder re-ranking before context is sent to the LLM.
- Stored chat history, including the PDFs associated with each chat turn.
- Source tracking when the index contains file/page metadata.
- Groq API generation with configurable model via `GROQ_MODEL`.
- General chat fallback before any PDFs are uploaded or indexed.
- Configurable storage paths for PDFs, document memory, and chat history.

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

### 4. Chat and add documents

You can chat immediately, even before uploading PDFs.

You can add PDFs in either of these ways:

- Start the app and attach PDFs from the message composer. Selected files are
  shown on the outgoing message, indexed, and then used to answer that message.
- Manually create a `data/` directory and place PDF files inside it.

Uploaded files are saved to app storage and indexed automatically. Once upload
finishes, you can ask questions about those documents in the same chat.

### 5. Refresh document memory

Manual refresh is normally not needed after UI uploads, because the app indexes
PDFs automatically. Use the refresh memory button only after manually changing
files in the storage folder.

You can also run the same ingestion process manually:

```bash
python ingest.py
```

This chunks your PDFs, generates embeddings, and builds the FAISS index in the
configured `DB_DIR`.

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
| `APP_STORAGE_DIR` | No | `.` | Base directory for app storage |
| `DATA_DIR` | No | `<APP_STORAGE_DIR>/data` | PDF upload storage directory |
| `DB_DIR` | No | `<APP_STORAGE_DIR>/db` | FAISS/chunk storage directory |
| `CHAT_HISTORY_FILE` | No | `<APP_STORAGE_DIR>/chat_history.json` | Stored chat history file |

## Deploy to Hugging Face Spaces

1. Push this repo to GitHub.
2. Create a new Hugging Face Space with Docker as the SDK.
3. Connect your GitHub repo.
4. Add `GROQ_API_KEY` as a Space secret.
5. Optionally set `APP_STORAGE_DIR=/data` if you add Hugging Face persistent
   storage. Without persistent storage, uploads and chat history last for the
   running container session but may disappear after rebuilds or restarts.

The Docker image pre-downloads the embedding and re-ranking models, rebuilds
document memory when stored PDFs are present and no FAISS index exists, and
starts the Flask app with Gunicorn.

## Notes

- The LLM depends on Groq API for answer generation.
- Retrieval still uses local FAISS, BM25, sentence-transformer embeddings, and
  cross-encoder re-ranking.
- If no FAISS index exists yet, the app still works as a general Groq-powered
  assistant. Upload PDFs to automatically enable document-grounded answers with
  source citations.
- If you regenerate `db/chunks.json` with the current `ingest.py`, answers can
  include source file and page citations.
