---
title: RAG AI Assistant
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Personal AI Assistant (Local RAG + Voice)

A fully local **AI-powered personal assistant** built using a Retrieval-Augmented Generation (RAG) pipeline. The system can understand documents, answer questions contextually, and interact through **voice input and output** — all without relying on paid APIs.

---

## Features

**Document Understanding**
  Extracts and processes PDFs using chunking and embeddings.

**RAG Pipeline**
  Combines semantic search (FAISS) with LLM reasoning for accurate answers.

**Conversational Memory**
  Maintains chat history for multi-turn interactions.

**Voice Assistant**
  Speech-to-Text using Whisper
  Text-to-Speech using pyttsx3

**Fully Local Setup**
  Runs entirely offline using Ollama + Mistral (no API cost).

**Cloud-Ready**
  Deploy to Hugging Face Spaces with Groq API — no local hardware needed.

---

## Tech Stack

* **Language:** Python
* **LLM:** Mistral (via Ollama) locally, or Mixtral 8x7B (via Groq API) on HF Spaces
* **Embeddings:** Sentence Transformers
* **Vector DB:** FAISS
* **Speech-to-Text:** Whisper
* **Text-to-Speech:** pyttsx3

---

## Project Structure

```
rag-assistant/
│
├── app.py            # Flask web server
├── query.py          # RAG pipeline + response generation
├── ingest.py         # PDF processing + embedding generation
├── voice.py          # Voice input/output integration
├── Dockerfile        # HF Spaces container definition
├── requirements.txt
├── .gitattributes
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/Aditya10507/Local-RAG-based-Voice-AI-Assistant.git
cd rag-assistant
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Start Ollama (for Mistral)

```
ollama run mistral
```

### 5. Run the assistant

```
python app.py
```

Then open **http://localhost:7860** (or `5000` locally) in your browser.

---

## Deploy to Hugging Face Spaces

### One-click deploy

1. Push this repo to GitHub
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
3. Choose **Docker** as the SDK
4. Connect your GitHub repo
5. Add your **Groq API key** as a secret (`GROQ_API_KEY`)
6. Upload a PDF via the Space's persistent storage or build the index after deploying

The Dockerfile handles everything — no manual configuration needed.

### Environment Variables

| Variable        | Required | Description                                     |
|-----------------|----------|-------------------------------------------------|
| `GROQ_API_KEY`  | Yes      | Groq API key for cloud LLM (get at console.groq.com) |
| `PORT`          | No       | Port to listen on (default: 7860 for HF Spaces) |

### How it works on HF Spaces

- **FAISS index** stays on the Space's persistent storage (survives restarts)
- **sentence-transformers** runs locally inside the container
- **Groq API** handles LLM inference (Mixtral 8x7B — 32K context, fast)
- The web UI is served by Flask + gunicorn

---

## Use Cases

* Personal knowledge assistant
* Resume/document Q&A system
* Offline AI chatbot
* Voice-enabled AI applications

---

## Key Learnings

* Built a complete **RAG architecture from scratch**
* Implemented **semantic search using FAISS**
* Integrated **LLMs locally without APIs**
* Developed **voice-enabled AI interactions**
* Deployed to **Hugging Face Spaces with Docker**
* Managed **clean Git workflows and project structure**

---

## Author

**Aditya Singh**
Aspiring AI/ML Engineer focused on building real-world intelligent systems.

---

## If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
