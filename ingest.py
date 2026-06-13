from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
	"""Split `text` into chunks of size `chunk_size` with `overlap` characters of overlap."""
	if chunk_size <= overlap:
		raise ValueError("chunk_size must be larger than overlap")
	chunks = []
	start = 0
	text_len = len(text)
	while start < text_len:
		end = start + chunk_size
		chunks.append(text[start:end])
		start = end - overlap
	return chunks


def ingest_pdf(data_dir: str = "data", db_dir: str = "db") -> None:
	"""Load the first PDF from `data_dir`, chunk text, embed via OpenAI, and save FAISS DB to `db_dir`.

	This implementation uses `pypdf` to read PDF text and `FAISS.from_texts` to build the index,
	avoiding LangChain document loader compatibility issues.
	"""

	# 1) No API keys required: embeddings are generated locally using
	#    the sentence-transformers model `all-MiniLM-L6-v2`.

	# 2) Find a PDF file in the data directory
	data_path = Path(data_dir)
	pdf_paths = list(data_path.glob("*.pdf"))
	if not pdf_paths:
		raise FileNotFoundError(f"No PDF files found in {data_dir}. Place a .pdf file there and retry.")

	pdf_path = pdf_paths[0]
	print(f"Loading PDF: {pdf_path}")

	# 3) Read PDF text using pypdf
	reader = PdfReader(str(pdf_path))
	full_text = []
	for page in reader.pages:
		page_text = page.extract_text() or ""
		full_text.append(page_text)
	text = "\n\n".join(full_text)

	# 4) Chunk the text into ~500-character pieces with 100-character overlap
	chunks = chunk_text(text, chunk_size=500, overlap=100)
	print(f"Split into {len(chunks)} chunks")

	# 5) Generate embeddings locally using sentence-transformers (no API keys)
	#    This avoids remote API calls and uses a compact, fast model.
	st_model = SentenceTransformer("all-MiniLM-L6-v2")
	# Encode all chunks at once to a numpy array (n_chunks, dim)
	vectors = st_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

	# 6) Build a FAISS index from the vectors and save locally along with chunk texts
	vectors = np.array(vectors).astype("float32")
	dim = vectors.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(vectors)

	db_path = Path(db_dir)
	db_path.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(db_path / "index.faiss"))

	# Save chunks to a JSON file so we can map search results back to their text
	with open(db_path / "chunks.json", "w", encoding="utf-8") as f:
		json.dump(chunks, f, ensure_ascii=False)

	print(f"Saved FAISS index and chunks to {db_path}")


if __name__ == "__main__":
	ingest_pdf()

