from pathlib import Path
import json
import os
import subprocess

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# In-memory conversation history. Each item is a dict: {"user": str, "assistant": str}
# This is kept in memory only (no persistence) and will be included in prompts
# to provide conversational context to the LLM.
chat_history = []

# Module-level cache for loaded resources to avoid reloading on every call
_GLOBAL = {
    "index": None,
    "chunks": None,
    "st_model": None,
}


def _ensure_loaded(db_dir: str = "db"):
    """Lazy-load FAISS index, chunks, and sentence-transformers model into module cache."""
    db_path = Path(db_dir)
    index_file = db_path / "index.faiss"
    chunks_file = db_path / "chunks.json"
    if _GLOBAL["index"] is None:
        if not index_file.exists() or not chunks_file.exists():
            raise FileNotFoundError(f"FAISS DB not found in {db_dir}. Run `ingest.py` first.")
        _GLOBAL["index"] = faiss.read_index(str(index_file))
        with open(chunks_file, "r", encoding="utf-8") as f:
            _GLOBAL["chunks"] = json.load(f)
    if _GLOBAL["st_model"] is None:
        _GLOBAL["st_model"] = SentenceTransformer("all-MiniLM-L6-v2")


def _build_prompt(user_query: str, context: str) -> str:
    """Build the full prompt with system instruction, history, and context."""
    last_n = 3
    recent = chat_history[-last_n:]
    history_text = ""
    if recent:
        history_lines = []
        for item in recent:
            history_lines.append(f"User: {item['user']}")
            history_lines.append(f"Assistant: {item['assistant']}")
            history_lines.append("")
        history_text = "\n".join(history_lines)

    system_instruction = (
        "You are a helpful personal AI assistant. Answer clearly and based only on the provided context. "
        "If the answer is not contained in the context, say 'I don't know' rather than guessing. "
        "Respond naturally and conversationally."
    )

    prompt = system_instruction + "\n\n"
    if history_text:
        prompt += f"Conversation history:\n{history_text}\n"

    prompt += (
        "Context (use this information as the only source):\n"
        f"{context}\n\n"
        f"User question: {user_query}\n\n"
        "Assistant (answer conversationally and concisely):"
    )
    return prompt


def _generate_with_groq(prompt: str) -> str:
    """Generate a response using the Groq API (cloud LLM)."""
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7,
    )
    return completion.choices[0].message.content


def _generate_with_ollama(prompt: str) -> str:
    """Generate a response using local Ollama (fallback for development)."""
    try:
        proc = subprocess.run(
            ["ollama", "run", "mistral", "--prompt", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        else:
            return f"Ollama Error: {proc.stderr}"
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama not found and no GROQ_API_KEY set. "
            "Either install Ollama locally or set the GROQ_API_KEY environment variable "
            "to use the Groq cloud API."
        )


def get_response(user_query: str, db_dir: str = "db", k: int = 3) -> str:
    """Answer a question by retrieving from FAISS and generating via Groq or Ollama.

    Uses Groq API if GROQ_API_KEY env var is set, otherwise falls back to local Ollama.
    Returns the assistant's answer as a string and also appends the interaction
    to the in-memory `chat_history`.
    """
    _ensure_loaded(db_dir)

    index = _GLOBAL["index"]
    chunks = _GLOBAL["chunks"]
    st_model = _GLOBAL["st_model"]

    # Embed the question
    q_vec = st_model.encode([user_query], convert_to_numpy=True).astype("float32")

    # Search FAISS
    D, I = index.search(q_vec, k)
    top_idxs = I[0].tolist()
    retrieved = [chunks[i] for i in top_idxs]
    context = "\n\n".join(retrieved)

    prompt = _build_prompt(user_query, context)

    # Choose LLM backend: Groq (cloud) if API key is set, else Ollama (local)
    if os.environ.get("GROQ_API_KEY"):
        answer = _generate_with_groq(prompt)
    else:
        answer = _generate_with_ollama(prompt)

    # Ensure answer is a string and update history
    final_answer = str(answer) if answer is not None else "I'm sorry, I couldn't generate a response."
    chat_history.append({"user": user_query, "assistant": final_answer})

    return final_answer


def query_db(db_dir: str = "db") -> None:
	"""Interactively prompt user for a question and print the response."""
	question = input("Enter your question: ")
	if not question.strip():
		print("No question provided. Exiting.")
		return
	answer = get_response(question, db_dir=db_dir)
	print("\n=== Answer ===\n")
	print(answer)


if __name__ == "__main__":
	query_db()
