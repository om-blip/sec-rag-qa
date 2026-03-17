# retriever.py
# This file does one job: given a question, find the most
# relevant chunks from ChromaDB and return them.
# This runs on EVERY question the user asks.

import chromadb
from sentence_transformers import SentenceTransformer

# Same model as ingest.py — MUST be identical or similarity
# search breaks (you'd be comparing apples to oranges).
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def retrieve_chunks(query: str, collection_name: str, top_k: int = 5) -> list[dict]:
    """
    Given a question and a collection name (from ingest_pdf),
    returns the top_k most semantically similar chunks.

    Each returned dict has:
      - "text": the chunk content
      - "page": which page it came from (for citations)
      - "score": similarity score (0-1, higher = more relevant)
    """
    collection = chroma_client.get_collection(name=collection_name)

    # Embed the question using the same model we used for chunks.
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            # Distance is 0-2 (lower = more similar). Convert to a
            # 0-1 similarity score so it's more intuitive.
            "score": round(1 - results["distances"][0][i] / 2, 3)
        })

    return chunks


# Quick test — run directly after ingest.py:
# python retriever.py <collection_id> "What are the key risks?"
if __name__ == "__main__":
    import sys
    collection_id = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What are the key risks?"

    chunks = retrieve_chunks(query, collection_id)
    print(f"\nQuery: {query}")
    print(f"Top {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}] Page {chunk['page']} | Score: {chunk['score']}")
        print(f"    {chunk['text'][:200]}...")
        print()