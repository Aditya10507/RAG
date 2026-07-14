from pathlib import Path
import os

from pypdf import PdfReader
import json


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Recursively split text at semantic boundaries (paragraphs → sentences → words).

    Preserves sentence and paragraph boundaries where possible for better RAG quality.
    Falls back to character-level splitting when no separator produces valid chunks.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
    return _recursive_split(text, separators, chunk_size, overlap, 0)


def _recursive_split(text: str, separators: list, chunk_size: int, overlap: int, depth: int) -> list[str]:
    """Recursively split text, trying progressively finer separators."""
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # If we've exhausted separators, split by character count
    if depth >= len(separators):
        return _char_split(text, chunk_size, overlap)

    sep = separators[depth]
    parts = text.split(sep)

    # If this separator barely splits anything, try the next one
    if len(parts) <= 1:
        return _recursive_split(text, separators, chunk_size, overlap, depth + 1)

    chunks = []
    current_batch = []
    current_len = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        part_len = len(part)

        # If a single part exceeds chunk_size, recurse into it with finer separators
        if part_len > chunk_size:
            # Flush current batch first
            if current_batch:
                chunks.append(sep.join(current_batch))
                current_batch = []
                current_len = 0

            sub_chunks = _recursive_split(part, separators, chunk_size, overlap, depth + 1)
            chunks.extend(sub_chunks)
            continue

        sep_cost = len(sep) if current_batch else 0
        if current_len + sep_cost + part_len > chunk_size and current_batch:
            chunk_text_str = sep.join(current_batch)
            chunks.append(chunk_text_str)

            # Carry overlap characters from the end
            if overlap > 0 and len(chunk_text_str) > overlap:
                carry = chunk_text_str[-overlap:].lstrip()
                if carry:
                    current_batch = [carry]
                    current_len = len(carry)
                else:
                    current_batch = []
                    current_len = 0
            else:
                current_batch = []
                current_len = 0

        current_batch.append(part)
        current_len += part_len + (len(sep) if len(current_batch) > 1 else 0)

    if current_batch:
        chunks.append(sep.join(current_batch))

    return chunks


def _char_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback: split by exact character count with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def ingest_pdf(data_dir: str = "data", db_dir: str = "db") -> None:
    """Load PDFs from `data_dir`, chunk text semantically, embed locally, and save FAISS DB to `db_dir`.

    This implementation:
    - Uses pypdf to read PDF text page-by-page
    - Recursively chunks preserving paragraph/sentence boundaries
    - Tracks per-chunk metadata (source file, page number)
    - Embeds using sentence-transformers (no API keys needed)
    - Saves both the FAISS index and chunk texts with metadata
    """

    # 1) Find PDF files in the data directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(data_path.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. Upload PDFs from the web app "
            "or place .pdf files in this folder, then run `python ingest.py` again."
        )

    all_chunks = []

    for pdf_path in pdf_paths:
        print(f"Loading PDF: {pdf_path}", flush=True)

        # 2) Read PDF text using pypdf, tracking page numbers
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            # 3) Split page text into semantically-aware chunks
            page_chunks = chunk_text(page_text, chunk_size=500, overlap=100)
            for chunk_text_content in page_chunks:
                all_chunks.append({
                    "text": chunk_text_content,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_num,
                    }
                })

    print(f"Split into {len(all_chunks)} total chunks across {len(pdf_paths)} PDF(s)", flush=True)
    if not all_chunks:
        raise RuntimeError(
            "No readable text was extracted from the uploaded PDFs. Try PDFs with selectable text, "
            "or add OCR support for scanned documents."
        )

    # 4) Generate embeddings locally using sentence-transformers
    print("Loading embedding model and generating vectors...", flush=True)
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_texts = [c["text"] for c in all_chunks]
    vectors = st_model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)

    # 5) Build a FAISS index from the vectors
    vectors = np.array(vectors).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    db_path = Path(db_dir)
    db_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(db_path / "index.faiss"))

    # 6) Save chunks with metadata
    with open(db_path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    print(f"Saved FAISS index and {len(all_chunks)} chunks (with metadata) to {db_path}", flush=True)


if __name__ == "__main__":
    ingest_pdf(
        data_dir=os.environ.get("DATA_DIR", "data"),
        db_dir=os.environ.get("DB_DIR", "db"),
    )
