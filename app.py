"""Flask web server for the Personal AI Assistant (RAG + Groq API).

Run:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import os
from pathlib import Path
import json

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


def _load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs when python-dotenv is not installed."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

from ingest import ingest_pdf
from query import get_response, chat_history, reset_rag_cache

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024

STORAGE_DIR = Path(os.environ.get("APP_STORAGE_DIR", ".")).expanduser()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(STORAGE_DIR / "data"))).expanduser()
DB_DIR = Path(os.environ.get("DB_DIR", str(STORAGE_DIR / "db"))).expanduser()
CHAT_HISTORY_FILE = Path(
    os.environ.get("CHAT_HISTORY_FILE", str(STORAGE_DIR / "chat_history.json"))
).expanduser()
ALLOWED_EXTENSIONS = {"pdf"}


def _ensure_storage_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_chat_history() -> None:
    """Load saved chat turns from disk for a simple persistent chat experience."""
    if not CHAT_HISTORY_FILE.exists():
        return

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    if isinstance(saved, list):
        chat_history.clear()
        chat_history.extend(
            item for item in saved
            if isinstance(item, dict) and "user" in item and "assistant" in item
        )


def _save_chat_history() -> None:
    _ensure_storage_dirs()
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False)


def _chunk_count() -> int:
    chunks_file = DB_DIR / "chunks.json"
    if not chunks_file.exists():
        return 0

    with open(chunks_file, "r", encoding="utf-8") as f:
        return len(json.load(f))


def _rebuild_document_index() -> dict:
    """Rebuild stored document search data and return index stats."""
    ingest_pdf(data_dir=str(DATA_DIR), db_dir=str(DB_DIR))
    reset_rag_cache()
    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    return {
        "documents": pdf_count,
        "chunks": _chunk_count(),
    }


_ensure_storage_dirs()
_load_chat_history()


def _is_allowed_pdf(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _unique_upload_path(filename: str) -> Path:
    """Return a non-conflicting path inside the upload directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = secure_filename(filename) or "document.pdf"
    candidate = DATA_DIR / safe_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        candidate = DATA_DIR / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "25"))
    return jsonify({"error": f"Upload is too large. Maximum size is {max_upload_mb} MB."}), 413


@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Handle a chat message and return the assistant's response.

    Expects JSON body: {"message": "user question"}
    Returns JSON: {"reply": "assistant answer", "history": [...]}
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_msg = data["message"].strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Attachment metadata is supplied only after /api/upload has safely stored
    # the files. Keep a small, display-only representation with the chat turn
    # so documents remain visually associated with the message that used them.
    raw_attachments = data.get("attachments", [])
    attachments = []
    if isinstance(raw_attachments, list):
        for item in raw_attachments[:10]:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename", "")).strip()
            if not filename:
                continue
            try:
                size = max(0, int(item.get("size", 0)))
            except (TypeError, ValueError):
                size = 0
            attachments.append({"filename": filename, "size": size})

    try:
        reply = get_response(user_msg, db_dir=str(DB_DIR))
        if attachments and chat_history:
            chat_history[-1]["attachments"] = attachments
        _save_chat_history()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

    return jsonify({"reply": reply})


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return the current conversation history."""
    return jsonify({"history": chat_history})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear the conversation history."""
    chat_history.clear()
    _save_chat_history()
    return jsonify({"status": "cleared"})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload one or more PDF documents, store them, and index them immediately."""
    files = request.files.getlist("documents")
    if not files:
        return jsonify({"error": "No PDF files were uploaded."}), 400

    uploaded = []
    rejected = []

    for file in files:
        original_name = file.filename or ""
        if not original_name:
            rejected.append({"filename": original_name, "reason": "Missing filename"})
            continue

        if not _is_allowed_pdf(original_name):
            rejected.append({"filename": original_name, "reason": "Only PDF files are allowed"})
            continue

        destination = _unique_upload_path(original_name)
        file.save(destination)
        uploaded.append({
            "filename": destination.name,
            "size": destination.stat().st_size,
        })

    if not uploaded:
        return jsonify({"error": "No valid PDF files were uploaded.", "rejected": rejected}), 400

    try:
        stats = _rebuild_document_index()
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "uploaded": uploaded, "rejected": rejected}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e), "uploaded": uploaded, "rejected": rejected}), 400
    except Exception as e:
        return jsonify({
            "error": f"Upload saved, but indexing failed: {e}",
            "uploaded": uploaded,
            "rejected": rejected,
        }), 500

    return jsonify({
        "status": "stored_and_indexed",
        "uploaded": uploaded,
        "rejected": rejected,
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "message": f"Stored and indexed {stats['documents']} PDF(s). You can ask about them now.",
    })


@app.route("/api/documents", methods=["GET"])
def api_documents():
    """List uploaded PDF documents."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = [
        {
            "filename": path.name,
            "size": path.stat().st_size,
        }
        for path in sorted(DATA_DIR.glob("*.pdf"))
    ]
    return jsonify({"documents": documents})


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """Refresh searchable document memory from uploaded PDF documents."""
    try:
        stats = _rebuild_document_index()
        return jsonify({
            "status": "reindexed",
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "message": f"Document memory refreshed from {stats['documents']} PDF(s) with {stats['chunks']} chunk(s).",
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected indexing error: {e}"}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    index_exists = (DB_DIR / "index.faiss").exists()
    chunks_exists = (DB_DIR / "chunks.json").exists()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    return jsonify({
        "status": "ok",
        "index_ready": index_exists and chunks_exists,
        "uploaded_documents": pdf_count,
        "stored_messages": len(chat_history),
        "storage_dir": str(STORAGE_DIR),
        "groq_configured": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "groq_model": os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("=" * 60)
    print("  Personal AI Assistant — Web Interface")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=port)
