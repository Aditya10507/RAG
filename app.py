"""Flask web server for the Personal AI Assistant (RAG + Groq API).

Run:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ingest import ingest_pdf
from query import get_response, reset_rag_cache

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024

STORAGE_DIR = Path(os.environ.get("APP_STORAGE_DIR", ".")).expanduser()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(STORAGE_DIR / "data"))).expanduser()
DB_DIR = Path(os.environ.get("DB_DIR", str(STORAGE_DIR / "db"))).expanduser()
ALLOWED_EXTENSIONS = {"pdf"}
EXPLICIT_FRONTEND_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
}


def _is_allowed_frontend_origin(origin: str) -> bool:
    """Allow configured frontends, Vercel previews, and local development."""
    normalized = origin.rstrip("/")
    if normalized in EXPLICIT_FRONTEND_ORIGINS:
        return True

    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and hostname.endswith(".vercel.app"):
        return True
    return parsed.scheme in {"http", "https"} and hostname in {"localhost", "127.0.0.1"}


@app.after_request
def add_frontend_cors_headers(response):
    """Permit the static Vercel UI to call this public API without credentials."""
    origin = request.headers.get("Origin", "")
    if origin and _is_allowed_frontend_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers.add("Vary", "Origin")
    return response


def _ensure_storage_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)


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
    Returns JSON: {"reply": "assistant answer"}
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

    raw_history = data.get("history", [])
    conversation_history = []
    if isinstance(raw_history, list):
        for item in raw_history[-3:]:
            if not isinstance(item, dict):
                continue
            previous_user = str(item.get("user", "")).strip()
            previous_assistant = str(item.get("assistant", "")).strip()
            if previous_user and previous_assistant:
                conversation_history.append({
                    "user": previous_user[:4000],
                    "assistant": previous_assistant[:8000],
                })

    try:
        reply = get_response(
            user_msg,
            db_dir=str(DB_DIR),
            source_filenames=[item["filename"] for item in attachments] or None,
            conversation_history=conversation_history or None,
            record_history=False,
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

    return jsonify({"reply": reply})


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


@app.route("/api/documents/<path:filename>", methods=["DELETE"])
def api_delete_document(filename: str):
    """Delete one stored PDF and rebuild the remaining document index."""
    if Path(filename).name != filename or not _is_allowed_pdf(filename):
        return jsonify({"error": "Invalid document filename."}), 400

    document_path = DATA_DIR / filename
    if not document_path.is_file():
        return jsonify({"error": "Document not found."}), 404

    document_path.unlink()
    remaining_documents = list(DATA_DIR.glob("*.pdf"))

    try:
        if remaining_documents:
            stats = _rebuild_document_index()
        else:
            for index_file in ("index.faiss", "chunks.json", "manifest.json"):
                path = DB_DIR / index_file
                if path.exists():
                    path.unlink()
            reset_rag_cache()
            stats = {"documents": 0, "chunks": 0}
    except Exception as e:
        return jsonify({
            "error": f"Document removed, but reindexing failed: {e}",
            "filename": filename,
        }), 500

    return jsonify({
        "status": "deleted",
        "filename": filename,
        "documents": stats["documents"],
        "chunks": stats["chunks"],
    })


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
        "storage_dir": str(STORAGE_DIR),
        "groq_configured": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "groq_model": os.environ.get("GROQ_LOW_LATENCY_MODEL", "openai/gpt-oss-20b"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("=" * 60)
    print("  Personal AI Assistant — Web Interface")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=port)
