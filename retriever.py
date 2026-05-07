"""
retriever.py - Retrieve relevant chunks from ChromaDB for a query.

Usage:
    python retriever.py "What is the role of TAF6 delta in apoptosis?"
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_assignment"
EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Embed query and retrieve top_k nearest chunks from ChromaDB.

    Returns:
        List of chunk dicts containing all metadata fields plus
        'similarity_score' (cosine similarity, higher = more relevant).
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        chunks.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": doc,
                **meta,
                # ChromaDB cosine space: distance = 1 - similarity
                "similarity_score": round(1.0 - dist, 4),
            }
        )

    return chunks


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is TAF6 delta?"
    print(f"Query: {query}\n")
    hits = retrieve(query, top_k=5)
    for h in hits:
        page = (
            f"p.{h['page_start']}"
            if h["page_start"] == h["page_end"]
            else f"pp.{h['page_start']}-{h['page_end']}"
        )
        print(f"[{h['similarity_score']:.3f}] {h['source_file']} | {h['section']} | {page}")
        print(f"  {h['text'][:250]}\n")
