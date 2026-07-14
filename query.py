from pathlib import Path
import gc
import json
import os
import re

import faiss
import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()
else:
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

# In-memory conversation history. Each item is a dict: {"user": str, "assistant": str}
# This is kept in memory only (no persistence) and will be included in prompts
# to provide conversational context to the LLM.
chat_history = []

# Module-level cache for loaded resources to avoid reloading on every call
_GLOBAL = {
    "index": None,
    "chunks": None,  # list of dicts: {"text": str, "metadata": dict}
    "embedding_model": None,
    "bm25": None,  # BM25Okapi instance
    "cross_encoder": None,
}

# RRF (Reciprocal Rank Fusion) constant
_RRF_K = 60

# Number of results at each pipeline stage
_DENSE_TOP_K = 20
_SPARSE_TOP_K = 20
_RRF_TOP_K = 15
_DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def _clean_model_answer(answer: str) -> str:
    """Remove reasoning tags if a model returns them despite hidden reasoning settings."""
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()


def reset_rag_cache() -> None:
    """Clear loaded index/retriever resources so the next query reloads rebuilt files."""
    _GLOBAL["index"] = None
    _GLOBAL["chunks"] = None
    _GLOBAL["bm25"] = None


def _low_memory_mode() -> bool:
    """Return whether inference models should be loaded one at a time."""
    return os.environ.get("RAG_LOW_MEMORY_MODE", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _release_model(cache_key: str) -> None:
    """Release an ONNX session and promptly return its model memory."""
    model = _GLOBAL.get(cache_key)
    if model is None:
        return
    _GLOBAL[cache_key] = None
    del model
    gc.collect()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def _ensure_loaded(db_dir: str = "db"):
    """Lazy-load the FAISS, BM25, embedding, and reranking resources."""
    db_path = Path(db_dir)
    index_file = db_path / "index.faiss"
    chunks_file = db_path / "chunks.json"
    manifest_file = db_path / "manifest.json"
    embedding_model_name = os.environ.get(
        "EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL
    )

    # --- Load FAISS + chunks ---
    if _GLOBAL["index"] is None:
        if not index_file.exists() or not chunks_file.exists():
            raise FileNotFoundError(
                f"FAISS DB not found in {db_dir}. Run `python ingest.py` first."
            )

        if not manifest_file.exists():
            raise FileNotFoundError(
                "The document index uses the previous embedding format. "
                "Re-upload the PDFs or run `python ingest.py` to rebuild it."
            )
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("embedding_model") != embedding_model_name:
            raise FileNotFoundError(
                "The document index was built with a different embedding model. "
                "Re-upload the PDFs or run `python ingest.py` to rebuild it."
            )

        _GLOBAL["index"] = faiss.read_index(str(index_file))
        with open(chunks_file, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        # Handle backwards compatibility: old format was list[str]
        if raw_chunks and isinstance(raw_chunks[0], str):
            _GLOBAL["chunks"] = [
                {"text": c, "metadata": {"source": "unknown"}}
                for i, c in enumerate(raw_chunks)
            ]
        else:
            _GLOBAL["chunks"] = raw_chunks

    # --- Load quantized ONNX embedding model ---
    if _GLOBAL["embedding_model"] is None:
        _GLOBAL["embedding_model"] = TextEmbedding(
            model_name=embedding_model_name,
            threads=1,
        )

    # --- Build BM25 index ---
    if _GLOBAL["bm25"] is None:
        tokenized_corpus = [_tokenize(c["text"]) for c in _GLOBAL["chunks"]]
        _GLOBAL["bm25"] = BM25Okapi(tokenized_corpus)

def _search_hybrid(query: str, allowed_indices: list[int] | None = None) -> list[int]:
    """Hybrid search combining dense (FAISS) and sparse (BM25) retrieval with RRF.

    When ``allowed_indices`` is provided, retrieval is restricted to those
    chunks so a message attachment is answered from that document only.
    Returns a list of chunk indices sorted by combined relevance.
    """
    allowed = set(allowed_indices) if allowed_indices is not None else None

    # --- Dense search via FAISS ---
    try:
        q_vec = np.asarray(
            list(_GLOBAL["embedding_model"].query_embed(query)),
            dtype="float32",
        )
        faiss.normalize_L2(q_vec)
        dense_k = _GLOBAL["index"].ntotal if allowed is not None else min(
            _DENSE_TOP_K, _GLOBAL["index"].ntotal
        )
        _dense_scores, dense_indices = _GLOBAL["index"].search(q_vec, dense_k)
    finally:
        # Render's free instance has 512 MB RAM. Release the embedding session
        # before loading the reranker to avoid overlapping model memory.
        if _low_memory_mode():
            _release_model("embedding_model")

    # --- Sparse search via BM25 ---
    tokenized_query = _tokenize(query)
    bm25_scores = _GLOBAL["bm25"].get_scores(tokenized_query)
    sparse_ranked = np.argsort(bm25_scores)[::-1].tolist()

    dense_ranked = [
        idx for idx in dense_indices[0].tolist()
        if idx >= 0 and (allowed is None or idx in allowed)
    ][:_DENSE_TOP_K]
    sparse_top_indices = [
        idx for idx in sparse_ranked
        if allowed is None or idx in allowed
    ][:_SPARSE_TOP_K]

    # --- RRF: combine rankings ---
    rrf_scores: dict[int, float] = {}

    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (_RRF_K + rank + 1)

    for rank, idx in enumerate(sparse_top_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (_RRF_K + rank + 1)

    # Sort by combined RRF score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _score in ranked[:_RRF_TOP_K]]


def _rerank(query: str, indices: list[int]) -> list[int]:
    """Re-rank retrieved chunks using a cross-encoder model.

    The cross-encoder scores each (query, chunk) pair more accurately than
    the bi-encoder (sentence-transformer) used for initial retrieval.
    """
    if not indices:
        return []

    chunks = _GLOBAL["chunks"]
    documents = [chunks[i]["text"] for i in indices]

    if _GLOBAL["cross_encoder"] is None:
        reranker_model_name = os.environ.get(
            "RERANKER_MODEL", _DEFAULT_RERANKER_MODEL
        )
        _GLOBAL["cross_encoder"] = TextCrossEncoder(
            model_name=reranker_model_name,
            threads=1,
        )

    try:
        # ONNX cross-encoder returns a relevance score per document.
        scores = list(_GLOBAL["cross_encoder"].rerank(query, documents))
    finally:
        if _low_memory_mode():
            _release_model("cross_encoder")

    scored = list(zip(indices, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _score in scored]


def _format_recent_history(last_n: int = 3) -> str:
    """Format recent chat turns for prompt context."""
    recent = chat_history[-last_n:]
    if not recent:
        return ""

    history_lines = []
    for item in recent:
        history_lines.append(f"User: {item['user']}")
        history_lines.append(f"Assistant: {item['assistant']}")
        history_lines.append("")
    return "\n".join(history_lines)


def _build_prompt(user_query: str, context: str) -> str:
    """Build the RAG prompt with system instruction, history, and context."""
    last_n = 3
    history_text = _format_recent_history(last_n)

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


def _build_general_prompt(user_query: str) -> str:
    """Build a general chat prompt used before any document index exists."""
    history_text = _format_recent_history()
    system_instruction = (
        "You are a helpful personal AI assistant. Answer naturally and clearly. "
        "No searchable PDF index is available for this exchange, so do not claim to have read "
        "uploaded or indexed documents. If the user asks about document contents, explain that "
        "they need to upload PDFs and rebuild the index for document-grounded answers."
    )

    prompt = system_instruction + "\n\n"
    if history_text:
        prompt += f"Conversation history:\n{history_text}\n"

    prompt += (
        f"User question: {user_query}\n\n"
        "Assistant (answer conversationally and concisely):"
    )
    return prompt


def _generate_with_groq(prompt: str) -> str:
    """Generate a response using the Groq API."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file "
            "before starting the assistant."
        )

    model = os.environ.get("GROQ_MODEL", _DEFAULT_GROQ_MODEL).strip() or _DEFAULT_GROQ_MODEL

    from groq import Groq

    client = Groq(api_key=api_key)
    request_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    if model.startswith("qwen/"):
        request_kwargs["reasoning_format"] = "hidden"
        request_kwargs["reasoning_effort"] = "none"

    try:
        completion = client.chat.completions.create(
            **request_kwargs
        )
    except Exception as e:
        error_text = str(e)
        if "blocked at the project level" in error_text or "model_permission_blocked_project" in error_text:
            raise RuntimeError(
                f"Groq model '{model}' is blocked for this project. "
                "Change GROQ_MODEL in .env to an enabled model, or enable the model in Groq project settings."
            ) from e
        raise
    return _clean_model_answer(completion.choices[0].message.content)


def get_response(
    user_query: str,
    db_dir: str = "db",
    k: int = 5,
    source_filenames: list[str] | None = None,
) -> str:
    """Answer a question using the full RAG pipeline: hybrid search → re-rank → generate.

    Pipeline:
    1. Hybrid retrieval (FAISS dense + BM25 sparse) with RRF fusion
    2. Cross-encoder re-ranking of top results
    3. Context assembly with source metadata
    4. LLM generation via the Groq API

    Returns the assistant's answer as a string and appends the interaction
    to the in-memory `chat_history`.
    """
    try:
        _ensure_loaded(db_dir)
    except FileNotFoundError:
        prompt = _build_general_prompt(user_query)
        answer = _generate_with_groq(prompt)
        final_answer = str(answer) if answer is not None else "I'm sorry, I couldn't generate a response."
        chat_history.append({"user": user_query, "assistant": final_answer})
        return final_answer

    chunks = _GLOBAL["chunks"]

    allowed_indices = None
    if source_filenames:
        allowed_sources = {Path(name).name for name in source_filenames if name}
        allowed_indices = [
            index for index, chunk in enumerate(chunks)
            if chunk.get("metadata", {}).get("source") in allowed_sources
        ]
        if not allowed_indices:
            raise RuntimeError(
                "The attached document is not available in the search index. "
                "Please attach it again."
            )

    # Step 1: Hybrid search (FAISS + BM25 with RRF)
    hybrid_indices = _search_hybrid(user_query, allowed_indices=allowed_indices)

    # Step 2: Cross-encoder re-ranking
    reranked_indices = _rerank(user_query, hybrid_indices)

    # Step 3: Take top-k results
    top_indices = reranked_indices[:k]

    # Step 4: Build context with source tracking
    retrieved = [chunks[i] for i in top_indices]
    context_parts = []
    all_sources = []
    for chunk in retrieved:
        meta = chunk.get("metadata", {})
        source_label = ""
        if meta.get("source") and meta.get("source") != "unknown":
            source_label = f"[Source: {meta['source']}"
            if meta.get("page"):
                source_label += f", Page {meta['page']}"
            source_label += "]"
            loc = f" (p. {meta['page']})" if meta.get("page") else ""
            all_sources.append(f"{meta['source']}{loc}")

        if source_label:
            context_parts.append(f"{source_label}\n{chunk['text']}")
        else:
            context_parts.append(chunk["text"])

    context = "\n\n".join(context_parts)

    # Step 5: Build prompt and generate
    prompt = _build_prompt(user_query, context)

    answer = _generate_with_groq(prompt)

    final_answer = str(answer) if answer is not None else "I'm sorry, I couldn't generate a response."

    # Append source references for transparency
    unique_sources = list(dict.fromkeys(all_sources))  # preserve order, deduplicate
    if unique_sources and len(final_answer) > 0:
        final_answer += f"\n\n---\n*Sources: {', '.join(unique_sources)}*"

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
