from pathlib import Path
import json
import subprocess
import shlex

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


def get_response(user_query: str, db_dir: str = "db", k: int = 3) -> str:
	"""Answer a question by retrieving from FAISS and generating via Ollama/CLI.

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

	# Prepare prompt with system instruction and recent chat history
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

	# Generate answer using Ollama (Python client or CLI fallback)
	answer = None
	try:
		from ollama import Ollama

		client = Ollama()
		resp = client.generate(model="mistral", prompt=prompt, options={"num_predict": 128})
		answer = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
	except (ImportError, Exception):
		try:
			proc = subprocess.run(
				["ollama", "run", "mistral", "--prompt", prompt],
				capture_output=True,
				text=True,
				encoding="utf-8",
				errors="replace",
			)
			answer = proc.stdout.strip() if proc.returncode == 0 else f"Ollama Error: {proc.stderr}"
		except FileNotFoundError:
			raise RuntimeError("Ollama not found. Please install the Ollama CLI.")

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
