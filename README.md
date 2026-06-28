Live link _ _ _   '''https://adi080122-local-rag-voice-ai-assistant.hf.space/'''
title: RAG AI Assistant
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Personal AI Assistant (Local RAG + Voice)

A fully local **AI-powered personal assistant** built using a production-grade Retrieval-Augmented Generation (RAG) pipeline. The system can understand documents, answer questions contextually, and interact through **voice input and output** — all without relying on paid APIs.

---

## Features

**📄 Document Understanding**
  Extracts and processes PDFs using semantic-aware recursive chunking (paragraph → sentence → word boundaries).

**🔀 Hybrid RAG Pipeline**
  Combines **dense (FAISS)** + **sparse (BM25)** retrieval with **Reciprocal Rank Fusion (RRF)** for significantly better document retrieval than pure semantic search alone.

**⚡ Cross-Encoder Re-Ranking**
  A lightweight cross-encoder model re-ranks the top retrieved results, ensuring the most relevant context reaches the LLM.

**💬 Conversational Memory**
  Maintains chat history for multi-turn interactions.

**📎 Source Tracking**
  Each response includes the document and page number the answer was derived from.

**🎤 Voice Assistant**
  Speech-to-Text using Whisper
  Text-to-Speech using pyttsx3

**🔧 Configurable LLM**
  Choose any Ollama model via the `OLLAMA_MODEL` environment variable (e.g., `llama3`, `gemma2`, `qwen2.5`, `mistral`).

**💯 Fully Local & Free**
  Runs entirely offline using Ollama (no API costs). All models — embeddings, cross-encoder, and LLM — run locally.

**☁️ Cloud-Ready**
  Deploy to Hugging Face Spaces with Groq API — no local hardware needed.

---

## Tech Stack

| Component  | Technology |
|---|---|
| **Language** | Python |
| **LLM (local)** | Mistral / Llama 3 / Gemma 2 / Qwen 2.5 (via Ollama, configurable via `OLLAMA_MODEL`) |
| **LLM (cloud)** | Mixtral 8x7B (via Groq API, free tier available) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Sparse Retrieval** | BM25 (`rank-bm25`) |
| **Dense Retrieval** | FAISS |
| **Re-Ranking** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Fusion Strategy** | Reciprocal Rank Fusion (RRF) |
| **Speech-to-Text** | Whisper |
| **Text-to-Speech** | pyttsx3 |
| **Web Framework** | Flask + Gunicorn |

---

## RAG Pipeline Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  1. Hybrid Retrieval                │
│  ├── Dense: FAISS (semantic) ────┐  │
│  └── Sparse: BM25 (keyword)  ────┤  │
│      └── RRF Fusion ─────────────┘  │
│      → Top 15 results               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  2. Cross-Encoder Re-Ranking        │
│  → Scores each (query, chunk) pair  │
│  → Top 5 most relevant chunks       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  3. Context Assembly                │
│  → Annotates chunks with source     │
│    metadata (file, page)            │
│  → Builds prompt with history       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  4. LLM Generation                  │
│  ├── Ollama (local) — default       │
│  └── Groq API — if key is set       │
└─────────────────────────────────────┘
    │
    ▼
  Answer + Source Citations
```

### Why this design?

- **Hybrid search (FAISS + BM25)** catches both semantic meaning AND exact keyword matches — much better than either alone
- **RRF fusion** combines rankings without needing to normalize scores across different retrieval methods
- **Cross-encoder re-ranking** is more accurate than bi-encoder (sentence-transformer) similarity, fixing the "good enough for search but not for final answer" problem
- **Recursive chunking** respects natural text boundaries (paragraphs → sentences), avoiding broken context
- **Source citations** give transparency about where answers come from

---

## Project Structure

```
rag-assistant/
│
├── app.py            # Flask web server
├── query.py          # RAG pipeline (hybrid search, re-ranking, generation)
├── ingest.py         # PDF processing + recursive chunking + embedding
├── voice.py          # Voice input/output integration
├── Dockerfile        # HF Spaces container definition
├── requirements.txt
├── .gitattributes
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Aditya10507/Local-RAG-based-Voice-AI-Assistant.git
cd rag-assistant
```

### 2. Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ingest your documents

Place PDF files in the `data/` directory, then run:

```bash
python ingest.py
```

This chunks your PDFs recursively at semantic boundaries, generates embeddings, and builds the FAISS + BM25 indexes.

### 5. Start Ollama

```bash
ollama run mistral
```

Or use a different model:

```bash
# On Windows (PowerShell):
$env:OLLAMA_MODEL="llama3"
ollama run llama3

# On macOS/Linux:
export OLLAMA_MODEL=llama3
ollama run llama3
```

### 6. Run the assistant

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

---

## Environment Variables

| Variable          | Required | Default   | Description                                     |
|-------------------|----------|-----------|-------------------------------------------------|
| `GROQ_API_KEY`    | No*      | —         | Groq API key for cloud LLM (get at console.groq.com) |
| `OLLAMA_MODEL`    | No       | `mistral` | Ollama model name (e.g., `llama3`, `gemma2`, `qwen2.5`) |
| `PORT`            | No       | `7860`    | Port to listen on                               |

*\*One of `GROQ_API_KEY` or local Ollama is required.*

---

## Deploy to Hugging Face Spaces

### Via GitHub

1. Push this repo to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
3. Choose **Docker** as the SDK
4. Connect your GitHub repo
5. Add your **Groq API key** as a secret (`GROQ_API_KEY`)
6. Upload a PDF via the Space's persistent storage or build the index after deploying

The Dockerfile handles everything — no manual configuration needed.

### How it works on HF Spaces

- **FAISS index** stays on the Space's persistent storage (survives restarts)
- **sentence-transformers** runs locally inside the container
- **Groq API** handles LLM inference (Mixtral 8x7B — 32K context, fast)
- The web UI is served by Flask + gunicorn

---

## Use Cases

- Personal knowledge assistant
- Resume/document Q&A system
- Offline AI chatbot
- Voice-enabled AI applications

---

## Key Learnings

- Built a production-grade **RAG architecture from scratch**
- Implemented **hybrid search with RRF fusion** for optimal retrieval
- Integrated **cross-encoder re-ranking** for answer quality
- Built **recursive semantic chunking** that respects text boundaries
- Developed **voice-enabled AI interactions**
- Deployed to **Hugging Face Spaces with Docker**
- Managed **clean Git workflows and project structure**

---

## Author

**Aditya Singh**
Aspiring AI/ML Engineer focused on building real-world intelligent systems.

---

## If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
