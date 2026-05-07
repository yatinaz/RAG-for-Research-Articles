"""
generator.py - Generate answers from retrieved chunks using Groq API.

Usage:
    python generator.py "How does TAF6 delta interact with p53?"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Best model available on Groq free tier — large context, fast.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env or export as env var."
            )
        _client = Groq(api_key=api_key)
    return _client


def _format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        page = (
            f"p.{chunk['page_start']}"
            if chunk["page_start"] == chunk["page_end"]
            else f"pp.{chunk['page_start']}-{chunk['page_end']}"
        )
        parts.append(
            f"[Source {i}: {chunk['source_file']} | "
            f"Section: {chunk['section']} | {page}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a grounded answer from retrieved chunks via Groq.

    Args:
        query: User question.
        chunks: Retrieved chunk dicts with text, source_file, section,
                page_start, page_end fields.
        model: Groq model name.

    Returns:
        Generated answer string with inline citations.
    """
    client = _get_client()
    context = _format_context(chunks)

    system_prompt = (
        "You are a precise research assistant. "
        "Answer ONLY using information from the provided context. "
        "For every factual claim, cite the source using "
        "[Source N: filename, Section: ..., page]. "
        "If the context does not contain enough information to answer "
        "the question, state that explicitly. "
        "Do not speculate or add information beyond what the sources provide."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from retriever import retrieve

    query = " ".join(sys.argv[1:]) or "What is the role of TAF6 delta in apoptosis?"
    print(f"Query: {query}\n")
    chunks = retrieve(query, top_k=5)
    print(f"Retrieved {len(chunks)} chunks\n")
    answer = generate(query, chunks)
    print("Answer:\n")
    print(answer)
