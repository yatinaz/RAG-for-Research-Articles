"""
embedder.py - Embed chunks into ChromaDB using sentence-transformers.

Runs the chunking pipeline automatically if chunks.jsonl is missing.

Usage:
    python embedder.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

CHUNKS_FILE = Path("chunks.jsonl")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_assignment"
EMBED_MODEL = "all-MiniLM-L6-v2"


def _load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _run_chunking_pipeline() -> None:
    print("chunks.jsonl not found — running chunking pipeline...")
    result = subprocess.run(
        [sys.executable, "run.py", "--input", ".", "--output", str(CHUNKS_FILE)],
    )
    if result.returncode != 0:
        raise RuntimeError("Chunking pipeline failed. Check run.py output above.")


def main() -> None:
    if not CHUNKS_FILE.exists():
        _run_chunking_pipeline()

    chunks = _load_chunks(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    from sentence_transformers import SentenceTransformer
    import chromadb

    t0 = time.time()
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embed_dim = int(embeddings.shape[1])

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size].tolist()
        collection.upsert(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_embeddings,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[
                {
                    "source_file": c["source_file"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "section": c["section"],
                    "chunk_type": c["chunk_type"],
                    "chunk_index": c["chunk_index"],
                    "token_count": c["token_count"],
                }
                for c in batch_chunks
            ],
        )
        print(
            f"  Upserted batch {i // batch_size + 1}"
            f" ({len(batch_chunks)} chunks)"
        )

    elapsed = time.time() - t0
    print(f"\nTotal chunks embedded : {len(chunks)}")
    print(f"Embedding dimension   : {embed_dim}")
    print(f"Time taken            : {elapsed:.1f}s")
    print(f"ChromaDB collection   : {COLLECTION_NAME} @ {CHROMA_DIR}")


if __name__ == "__main__":
    main()
