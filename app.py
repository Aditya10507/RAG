"""Flask web server for the Personal AI Assistant (RAG + Voice).

Run:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from query import get_response, chat_history

app = Flask(__name__)


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
    except FileNotFoundError:
        return jsonify({
            "error": "FAISS index not found. Run `python ingest.py` first to build the vector database."
        }), 500
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


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    db_path = Path("db")
    index_exists = (db_path / "index.faiss").exists()
    chunks_exists = (db_path / "chunks.json").exists()
    return jsonify({
        "status": "ok",
        "index_ready": index_exists and chunks_exists,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("=" * 60)
    print("  Personal AI Assistant — Web Interface")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=port)
