# ingest.py
# This file does one job: take a PDF, break it into chunks,
# embed each chunk, and store everything in ChromaDB.
# You run this ONCE per document. Think of it as "indexing".

import pdfplumber
import hashlib
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

# Load the embedding model once at module level.
# all-MiniLM-L6-v2 is 80MB, downloads automatically on first run.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialise ChromaDB with a local folder to persist data.
# This means your index survives between sessions — you don't
# re-ingest the same PDF every time you restart the app.
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def get_file_hash(pdf_path: str) -> str:
    """
    Generate a short unique ID for a PDF file based on its contents.
    We use this as the ChromaDB collection name so the same PDF
    always maps to the same collection — enabling caching.
    """
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.
    Returns a list of dicts: {"page": 1, "text": "..."}
    We keep page numbers because we'll use them later as citations.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
            # fix missing spaces around numbers and capital letters
                text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
                text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
            if text and len(text.strip()) > 50:
                # Skip pages with less than 50 chars — usually cover pages,
                # blank pages, or image-only pages with no extractable text.
                pages.append({
                    "page": i + 1,
                    "text": text.strip()
                })
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split page text into overlapping chunks.
    RecursiveCharacterTextSplitter tries to split at paragraph
    boundaries first, then sentences, then words — so chunks
    are semantically cleaner than simple character splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        # These separators tell it to prefer splitting at paragraph
        # breaks first, then newlines, then sentences, then words.
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"]  # carry forward page number as metadata
            })
    return chunks


def ingest_pdf(pdf_path: str) -> str:
    """
    Main function. Call this with a PDF path.
    Returns the collection name (= file hash) so the retriever
    knows which ChromaDB collection to search.

    Flow: PDF → pages → chunks → embeddings → ChromaDB
    """
    pdf_path = str(Path(pdf_path).resolve())
    file_hash = get_file_hash(pdf_path)

    # CACHING: if this collection already exists in ChromaDB,
    # skip re-ingestion entirely. Makes repeat runs instant.
    existing = [c.name for c in chroma_client.list_collections()]
    if file_hash in existing:
        print(f"Collection {file_hash} already exists. Skipping ingestion.")
        return file_hash

    print("Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)
    print(f"  → {len(pages)} pages extracted")

    print("Chunking text...")
    chunks = chunk_pages(pages)
    print(f"  → {len(chunks)} chunks created")

    print("Embedding chunks (this takes ~30-60s on first run)...")
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    print("Storing in ChromaDB...")
    collection = chroma_client.create_collection(name=file_hash)
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"page": c["page"]} for c in chunks],
        # IDs must be unique strings. We use chunk index as ID.
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"Done. {len(chunks)} chunks stored in collection '{file_hash}'")
    return file_hash


# Quick test — run this file directly to test ingestion:
# python ingest.py
if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    collection_id = ingest_pdf(pdf)
    print(f"\nCollection ID: {collection_id}")
    print("Ingestion complete. Now test retriever.py")