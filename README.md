# Archive AI — RAG Document Assistant

[![Live frontend](https://img.shields.io/badge/Live-Vercel-000000?logo=vercel)](https://archive-ai-rag.vercel.app)
[![API health](https://img.shields.io/badge/API-Render-46E3B7?logo=render)](https://rag-t7t1.onrender.com/api/health)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)

Archive AI is a production-deployed Retrieval-Augmented Generation (RAG)
assistant for PDF documents. Users attach PDFs directly to a chat message, the
backend indexes their contents, and Groq generates grounded answers with source
file and page citations.

## Live Project

| Resource | Link |
|---|---|
| Production frontend | [archive-ai-rag.vercel.app](https://archive-ai-rag.vercel.app) |
| Render backend | [rag-t7t1.onrender.com](https://rag-t7t1.onrender.com) |
| API health check | [rag-t7t1.onrender.com/api/health](https://rag-t7t1.onrender.com/api/health) |
| GitHub repository | [github.com/Aditya10507/RAG](https://github.com/Aditya10507/RAG) |

The Render free service may need a short cold-start period after inactivity.
The Vercel frontend remains available while the API wakes up.

## Latest Updates

- Rebuilt the interface as a professional, responsive document workspace.
- Added PDF attachment cards to the exact user message that submitted them.
- Added an expandable/collapsible desktop sidebar and mobile layout.
- Scoped retrieval to the documents attached to each message.
- Added document deletion with automatic index rebuilding.
- Moved per-browser chat history and PDF recovery data to IndexedDB.
- Added automatic PDF restoration after an ephemeral backend restart.
- Reduced response latency with a compact FastEmbed/FAISS pipeline and a
  low-latency Groq model preference.
- Split deployment: static frontend on Vercel and the RAG API on Render.
- Added production CORS rules, deployment ignore files, and secret-safe builds.
- Removed obsolete server-history endpoints, duplicate environment loading, and
  unreachable legacy index compatibility code.

## Features

- Multiple PDF uploads from the chat composer.
- Recursive semantic chunking with page-level metadata.
- Dense FAISS and sparse BM25 retrieval combined with Reciprocal Rank Fusion.
- Cross-encoder re-ranking for multi-document searches.
- Direct document summaries with broader source coverage.
- Groq generation with a preferred fast model and configurable fallback.
- Grounded answers with source filename and page citations.
- Browser-local chat and PDF persistence through IndexedDB.
- Automatic document recovery when Render's ephemeral storage resets.
- Responsive, accessible, non-animated interface.

## Architecture

```text
Browser / Vercel
  ├─ Professional chat interface
  ├─ Message-bound PDF attachments
  └─ IndexedDB: chat turns + recoverable PDF blobs
              │ HTTPS / JSON / multipart PDF
              ▼
Flask API / Render
  ├─ PDF validation and temporary storage
  ├─ pypdf extraction and semantic chunking
  ├─ FastEmbed ONNX embeddings
  ├─ FAISS dense search + BM25 sparse search
  ├─ Reciprocal Rank Fusion + cross-encoder re-ranking
  └─ Groq generation
              │
              ▼
Grounded answer + source citations
```

The Groq key and all model calls remain on Render. Vercel receives only the
static HTML/CSS/JavaScript frontend. IndexedDB is used instead of SQLite for
temporary user data because it keeps each visitor's chats and PDF recovery
blobs on their own device; a SQLite file on a free ephemeral container would
be shared and removed during restarts.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Device storage | IndexedDB |
| Backend | Python 3.11, Flask, Gunicorn |
| PDF extraction | pypdf |
| Embeddings | FastEmbed ONNX (`BAAI/bge-small-en-v1.5`) |
| Dense retrieval | FAISS cosine similarity |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion |
| Re-ranking | FastEmbed cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Groq API |
| Hosting | Vercel frontend + Render backend |

## Project Structure

```text
rag-assistant/
├── app.py                       # Flask routes, uploads, CORS, health checks
├── ingest.py                    # PDF extraction, chunking, embeddings, FAISS
├── query.py                     # Retrieval, re-ranking, prompts, Groq calls
├── templates/
│   └── index.html               # Complete responsive chat frontend
├── scripts/
│   └── build-frontend.mjs       # Creates the static Vercel build in dist/
├── Dockerfile                   # Production backend image
├── entrypoint.sh                # Storage setup and Gunicorn startup
├── render.yaml                  # Render Blueprint
├── vercel.json                  # Static frontend build configuration
├── package.json                 # Frontend build command
├── requirements.txt             # Python dependencies
├── .env.example                 # Safe environment template
├── .dockerignore                # Minimal backend build context
├── .vercelignore                # Prevents backend/private files from upload
└── README.md
```

Generated and private content is intentionally excluded from Git:

- `.env` — local secrets.
- `data/` — uploaded PDFs.
- `db/` — generated FAISS index, chunks, and manifest.
- `dist/` — generated Vercel frontend.
- `.vercel/`, virtual environments, caches, and bytecode.

## Local Setup

Requirements: Python 3.11+, Node.js 20+, and a Groq API key.

```bash
git clone https://github.com/Aditya10507/RAG.git
cd RAG
python -m venv venv
```

Activate the environment:

```powershell
# Windows PowerShell
venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source venv/bin/activate
```

Install dependencies and create the local environment file:

```bash
pip install -r requirements.txt
cp .env.example .env
```

On Windows without `cp`, use `Copy-Item .env.example .env`. Replace the
placeholder in `.env` with your Groq API key, then start the application:

```bash
python app.py
```

Open `http://localhost:7860`. PDFs uploaded through the UI are indexed
automatically. To index PDFs placed manually in `data/`, run `python ingest.py`.

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Server-side Groq credential |
| `GROQ_LOW_LATENCY_MODEL` | No | `openai/gpt-oss-20b` | Preferred fast model |
| `GROQ_MODEL` | No | `qwen/qwen3.6-27b` | Fallback model |
| `PORT` | No | `7860` | Local/server HTTP port |
| `MAX_UPLOAD_MB` | No | `25` | Maximum request size |
| `APP_STORAGE_DIR` | No | `.` | Base runtime storage directory |
| `DATA_DIR` | No | `<APP_STORAGE_DIR>/data` | Uploaded PDF directory |
| `DB_DIR` | No | `<APP_STORAGE_DIR>/db` | Generated retrieval index directory |
| `EMBEDDING_MODEL` | No | `BAAI/bge-small-en-v1.5` | FastEmbed embedding model |
| `RERANKER_MODEL` | No | `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RAG_LOW_MEMORY_MODE` | No | `1` | Releases ONNX sessions between stages |
| `FRONTEND_ORIGINS` | No | — | Additional comma-separated CORS origins |

## Deployment

### Backend on Render

The included `render.yaml` and `Dockerfile` define the backend deployment.

1. Create a Render Blueprint from this repository.
2. Keep the service on the free plan.
3. Set the secret `GROQ_API_KEY` in Render.
4. Deploy. Render uses `/api/health` for health checks.

The container preloads compact ONNX retrieval models during its image build and
runs one Gunicorn worker to stay within free-tier memory limits. Runtime PDFs
and indexes are ephemeral; the browser restores its cached PDFs when needed.

### Frontend on Vercel

Build and deploy the static frontend from the repository root:

```bash
npm run build
npx vercel@latest --prod
```

For the existing production project, maintainers can deploy explicitly with:

```bash
npx vercel@latest --prod --project archive-ai-rag
```

`.vercelignore` ensures that the Groq key, local PDFs, generated indexes, and
Python backend are never included in a frontend CLI upload.

## API Summary

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service, index, and Groq status |
| `GET` | `/api/documents` | List indexed PDFs |
| `POST` | `/api/upload` | Upload PDFs and rebuild the index |
| `POST` | `/api/chat` | Answer with optional attachment scope/history |
| `POST` | `/api/reindex` | Rebuild document memory manually |
| `DELETE` | `/api/documents/<filename>` | Delete a PDF and rebuild the index |

## Data and Security Notes

- Never commit `.env` or place `GROQ_API_KEY` in Vercel variables or frontend
  JavaScript.
- Uploaded PDFs are temporary backend files and browser-local recovery blobs.
- Chats are not stored in a shared server database.
- Only PDF filenames are accepted for document deletion; paths are rejected.
- The API limits upload request size and restricts browser CORS origins.

## License

This portfolio project is provided for demonstration and educational use.
