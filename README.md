# Personal AI Assistant (Local RAG + Voice)

An intelligent voice assistant that uses Retrieval-Augmented Generation (RAG) to answer queries based on your local documents, all running locally for privacy.

## 🚀 Features
- **RAG (Retrieval-Augmented Generation)**: Answers questions based on your local data.
- **FAISS Vector Store**: Fast local similarity search.
- **Mistral (via Ollama)**: High-performance local language model for inference.
- **Whisper**: Reliable local speech-to-text transcription.
- **pyttsx3**: Cross-platform text-to-speech for voice responses.
- **Conversational Memory**: Maintains context for more natural interactions.

## 🛠 Tech Stack
- **Language**: Python
- **AI Frameworks**: LangChain, Ollama
- **Vector DB**: FAISS
- **Audio**: OpenAI Whisper, pyttsx3

## ⚙️ Setup Instructions
1. **Activate the virtual environment**:
   ```bash
   .\venv\Scripts\activate
   ```
2. **Start Ollama**:
   ```bash
   ollama serve
   ```
3. **Run the Assistant**:
   ```bash
   python main.py
   ```