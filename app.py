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

DATA_DIR = Path("data")
ALLOWED_EXTENSIONS = {"pdf"}


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

    try:
        reply = get_response(user_msg)
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
    return jsonify({"status": "cleared"})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload one or more PDF documents into the data directory."""
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

    return jsonify({
        "status": "uploaded",
        "uploaded": uploaded,
        "rejected": rejected,
        "message": "Upload complete. Rebuild the index before asking questions about new files.",
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
    """Rebuild the FAISS index from uploaded PDF documents."""
    try:
        ingest_pdf(data_dir=str(DATA_DIR), db_dir="db")
        reset_rag_cache()

        chunks_file = Path("db") / "chunks.json"
        chunk_count = 0
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunk_count = len(json.load(f))

        pdf_count = len(list(DATA_DIR.glob("*.pdf")))
        return jsonify({
            "status": "reindexed",
            "documents": pdf_count,
            "chunks": chunk_count,
            "message": f"Index rebuilt from {pdf_count} PDF(s) with {chunk_count} chunk(s).",
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
    db_path = Path("db")
    index_exists = (db_path / "index.faiss").exists()
    chunks_exists = (db_path / "chunks.json").exists()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    return jsonify({
        "status": "ok",
        "index_ready": index_exists and chunks_exists,
        "uploaded_documents": pdf_count,
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
