from collections.abc import Iterator
from pathlib import Path
import gc
import json
import os
import re

import faiss
import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

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
_DEFAULT_FAST_GROQ_MODEL = "openai/gpt-oss-20b"
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

        _GLOBAL["chunks"] = raw_chunks

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

    if _GLOBAL["embedding_model"] is None:
        embedding_model_name = os.environ.get(
            "EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL
        )
        _GLOBAL["embedding_model"] = TextEmbedding(
            model_name=embedding_model_name,
            threads=1,
        )

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


def _format_recent_history(
    last_n: int = 3,
    conversation_history: list[dict] | None = None,
) -> str:
    """Format recent chat turns for prompt context."""
    history = chat_history if conversation_history is None else conversation_history
    recent = history[-last_n:]
    if not recent:
        return ""

    history_lines = []
    for item in recent:
        history_lines.append(f"User: {item['user']}")
        history_lines.append(f"Assistant: {item['assistant']}")
        history_lines.append("")
    return "\n".join(history_lines)


def _build_prompt(
    user_query: str,
    context: str,
    attached_sources: list[str] | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """Build the RAG prompt with system instruction, history, and context."""
    last_n = 3
    history_text = _format_recent_history(last_n, conversation_history)

    if attached_sources:
        source_names = ", ".join(attached_sources)
        system_instruction = (
            "You are a precise document analyst. The context below was extracted directly from "
            f"the user's attached document(s): {source_names}. Treat that context as the attached "
            "document; never claim that no document is attached or ask for a separate document. "
            "Answer the request directly and synthesize useful details instead of merely describing "
            "what information exists. For a summary or briefing, lead with a concise overview and "
            "then organize the most important experience, skills, projects, achievements, dates, or "
            "risks that are actually present. Do not invent missing facts."
        )
    else:
        system_instruction = (
            "You are a precise document analyst. Answer clearly and only from the provided context. "
            "If the answer is not contained in the context, say so briefly rather than guessing. "
            "Synthesize the information into a useful answer instead of talking about the retrieval process."
        )

    prompt = system_instruction + "\n\n"
    if history_text:
        prompt += f"Conversation history:\n{history_text}\n"

    prompt += (
        "Context (use this information as the only source):\n"
        f"{context}\n\n"
        f"User question: {user_query}\n\n"
        "Assistant (give a direct, polished, well-structured answer):"
    )
    return prompt


def _build_general_prompt(
    user_query: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """Build a general chat prompt used before any document index exists."""
    history_text = _format_recent_history(conversation_history=conversation_history)
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


def _groq_client_and_models():
    """Create a Groq client and return models in preferred fallback order."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or .env file "
            "before starting the assistant."
        )

    configured_model = os.environ.get("GROQ_MODEL", _DEFAULT_GROQ_MODEL).strip() or _DEFAULT_GROQ_MODEL
    fast_model = os.environ.get(
        "GROQ_LOW_LATENCY_MODEL", _DEFAULT_FAST_GROQ_MODEL
    ).strip() or _DEFAULT_FAST_GROQ_MODEL
    model_candidates = list(dict.fromkeys([fast_model, configured_model]))

    from groq import Groq

    return Groq(api_key=api_key), model_candidates


def _completion_request(prompt: str, model: str, *, stream: bool = False) -> dict:
    """Build one Groq completion request with model-specific reasoning settings."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 700,
        "temperature": 0.2,
        "stream": stream,
    }
    if model.startswith("qwen/"):
        request["reasoning_format"] = "hidden"
        request["reasoning_effort"] = "none"
    elif model.startswith("openai/gpt-oss"):
        request["reasoning_format"] = "hidden"
        request["reasoning_effort"] = "low"
    return request


def _model_is_unavailable(error: Exception) -> bool:
    """Return whether generation can safely fall back to another configured model."""
    error_text = str(error).lower()
    return any(marker in error_text for marker in (
        "blocked at the project level",
        "model_permission_blocked_project",
        "model_not_found",
        "does not exist",
        "permission",
    ))


def _generate_with_groq(prompt: str) -> str:
    """Generate one complete response using the Groq API."""
    client, model_candidates = _groq_client_and_models()
    last_error = None
    for position, model in enumerate(model_candidates):
        try:
            completion = client.chat.completions.create(
                **_completion_request(prompt, model)
            )
            return _clean_model_answer(completion.choices[0].message.content)
        except Exception as e:
            last_error = e
            model_unavailable = _model_is_unavailable(e)
            if position < len(model_candidates) - 1 and model_unavailable:
                continue
            if model_unavailable:
                raise RuntimeError(
                    f"Groq model '{model}' is unavailable for this project. "
                    "Enable it in Groq model permissions or configure another model."
                ) from e
            raise

    raise RuntimeError(f"Groq generation failed: {last_error}")


def _stream_with_groq(prompt: str) -> Iterator[str]:
    """Yield Groq response deltas as soon as the model produces them."""
    client, model_candidates = _groq_client_and_models()
    last_error = None

    for position, model in enumerate(model_candidates):
        emitted_text = False
        try:
            completion = client.chat.completions.create(
                **_completion_request(prompt, model, stream=True)
            )
            for chunk in completion:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    emitted_text = True
                    yield delta
            if not emitted_text:
                raise RuntimeError("Groq returned an empty response.")
            return
        except Exception as e:
            last_error = e
            model_unavailable = _model_is_unavailable(e)
            if not emitted_text and position < len(model_candidates) - 1 and model_unavailable:
                continue
            if model_unavailable:
                raise RuntimeError(
                    f"Groq model '{model}' is unavailable for this project. "
                    "Enable it in Groq model permissions or configure another model."
                ) from e
            raise

    raise RuntimeError(f"Groq generation failed: {last_error}")


def _prepare_response(
    user_query: str,
    db_dir: str = "db",
    k: int = 5,
    source_filenames: list[str] | None = None,
    conversation_history: list[dict] | None = None,
):
    """Prepare a grounded prompt and its ordered source labels."""
    try:
        _ensure_loaded(db_dir)
    except FileNotFoundError:
        return _build_general_prompt(user_query, conversation_history), []

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

    normalized_query = user_query.lower()
    summary_request = any(term in normalized_query for term in (
        "summar", "brief", "overview", "key points", "tell me about", "describe this"
    ))
    searchable_sources = {
        chunk.get("metadata", {}).get("source")
        for index, chunk in enumerate(chunks)
        if (allowed_indices is None or index in allowed_indices)
        and chunk.get("metadata", {}).get("source")
    }
    single_document_search = len(searchable_sources) == 1

    if single_document_search and summary_request:
        summary_indices = allowed_indices if allowed_indices is not None else list(range(len(chunks)))
        reranked_indices = summary_indices[:12]
    else:
        hybrid_indices = _search_hybrid(user_query, allowed_indices=allowed_indices)
        reranked_indices = (
            hybrid_indices
            if single_document_search
            else _rerank(user_query, hybrid_indices)
        )

    result_limit = 10 if single_document_search and summary_request else k
    retrieved = [chunks[i] for i in reranked_indices[:result_limit]]

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
            location = f" (p. {meta['page']})" if meta.get("page") else ""
            all_sources.append(f"{meta['source']}{location}")

        context_parts.append(
            f"{source_label}\n{chunk['text']}" if source_label else chunk["text"]
        )

    prompt = _build_prompt(
        user_query,
        "\n\n".join(context_parts),
        attached_sources=source_filenames,
        conversation_history=conversation_history,
    )
    return prompt, list(dict.fromkeys(all_sources))


def _source_footer(sources: list[str]) -> str:
    """Format ordered, de-duplicated source labels for the final answer."""
    return f"\n\n---\n*Sources: {', '.join(sources)}*" if sources else ""


def get_response(
    user_query: str,
    db_dir: str = "db",
    k: int = 5,
    source_filenames: list[str] | None = None,
    conversation_history: list[dict] | None = None,
    record_history: bool = True,
) -> str:
    """Answer a question using the full RAG pipeline: hybrid search → re-rank → generate.

    Pipeline:
    1. Hybrid retrieval (FAISS dense + BM25 sparse) with RRF fusion
    2. Cross-encoder re-ranking of top results
    3. Context assembly with source metadata
    4. LLM generation via the Groq API

    Returns the assistant's answer as a string. CLI callers can retain the
    interaction in the in-memory `chat_history`; the web UI stores history in
    browser IndexedDB and disables server-side recording.
    """
    prompt, sources = _prepare_response(
        user_query,
        db_dir=db_dir,
        k=k,
        source_filenames=source_filenames,
        conversation_history=conversation_history,
    )
    answer = _generate_with_groq(prompt)
    final_answer = str(answer) if answer is not None else "I'm sorry, I couldn't generate a response."
    final_answer += _source_footer(sources)

    if record_history:
        chat_history.append({"user": user_query, "assistant": final_answer})
    return final_answer


def stream_response(
    user_query: str,
    db_dir: str = "db",
    k: int = 5,
    source_filenames: list[str] | None = None,
    conversation_history: list[dict] | None = None,
    record_history: bool = True,
) -> Iterator[str]:
    """Yield a grounded answer incrementally, followed by its source footer."""
    prompt, sources = _prepare_response(
        user_query,
        db_dir=db_dir,
        k=k,
        source_filenames=source_filenames,
        conversation_history=conversation_history,
    )

    parts = []
    for delta in _stream_with_groq(prompt):
        parts.append(delta)
        yield delta

    footer = _source_footer(sources)
    if footer:
        parts.append(footer)
        yield footer

    if record_history:
        chat_history.append({"user": user_query, "assistant": "".join(parts)})


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
