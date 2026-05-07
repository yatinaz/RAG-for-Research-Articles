"""
eval.py - Retrieval evaluation: candidate generation and precision measurement.

Two-phase workflow:

Phase 1 — Generate candidates (requires Groq API key in .env):
    python eval.py --generate

    Reads chunks.jsonl, samples representative chunks across all papers,
    calls Groq to produce 25 question-answer pairs, saves to
    eval_candidates.json. Review and prune that file before Phase 2.

Phase 2 — Run evaluation (requires ChromaDB populated via embedder.py):
    python eval.py --run-eval

    Loads approved eval_candidates.json, runs retrieval for each question,
    computes precision@3 and precision@5, prints results, saves to
    eval_results.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

CHUNKS_FILE = Path("chunks.jsonl")
CANDIDATES_FILE = Path("eval_candidates.json")
RESULTS_FILE = Path("eval_results.json")

# Target ~4 questions per paper; 6 papers → ~24-25 total.
QUESTIONS_PER_PAPER = 4

# Cycle through question types so each paper gets mixed coverage.
_QUESTION_TYPES = ("factual", "comparative", "locating")

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "factual": (
        "Generate a specific factual question about the methods, findings, "
        "experimental conditions, or quantitative results described in this text. "
        "The answer must be explicitly stated in the text and must NOT be "
        "inferable from the question alone."
    ),
    "comparative": (
        "Generate a question that compares two things described in the text, "
        "asks how they differ, or asks what distinguishes one approach/finding "
        "from another. The complete answer must be present in the text."
    ),
    "locating": (
        "Generate a question of the form 'Which paper / study / section discusses X?' "
        "where X is a specific topic, method, or finding mentioned in this text. "
        "The answer should identify this paper or section explicitly."
    ),
}


def _load_chunks() -> list[dict]:
    chunks: list[dict] = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _sample_chunks_for_eval(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group text chunks by paper; pick diverse high-quality samples."""
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        # Only use substantive text chunks (skip garbled figure/table noise).
        if c["chunk_type"] == "text" and c["token_count"] >= 80:
            by_paper.setdefault(c["source_file"], []).append(c)

    sampled: dict[str, list[dict]] = {}
    rng = random.Random(42)  # reproducible sampling

    for paper, paper_chunks in by_paper.items():
        by_section: dict[str, list[dict]] = {}
        for c in paper_chunks:
            by_section.setdefault(c["section"], []).append(c)

        # Exclude boilerplate-adjacent sections.
        _skip = {"Introduction", "Abstract"}
        content_sections = [
            s for s in by_section if s not in _skip
        ] or list(by_section.keys())

        rng.shuffle(content_sections)
        selected: list[dict] = []

        for sec in content_sections:
            if len(selected) >= QUESTIONS_PER_PAPER:
                break
            candidates = by_section[sec]
            # Prefer richest chunk (most tokens = most content).
            selected.append(max(candidates, key=lambda c: c["token_count"]))

        # If not enough content sections, fill from abstract/intro.
        if len(selected) < QUESTIONS_PER_PAPER:
            for sec in _skip:
                if len(selected) >= QUESTIONS_PER_PAPER:
                    break
                if sec in by_section:
                    selected.append(
                        max(by_section[sec], key=lambda c: c["token_count"])
                    )

        sampled[paper] = selected[:QUESTIONS_PER_PAPER]

    return sampled


def _generate_qa_pair(
    chunk: dict,
    question_type: str,
    model: str,
) -> dict | None:
    """Call Groq to generate a single QA pair from a chunk."""
    import os
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env")

    client = Groq(api_key=api_key)

    page = (
        f"p.{chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp.{chunk['page_start']}-{chunk['page_end']}"
    )

    prompt = (
        f"You are building a retrieval-augmented generation evaluation dataset.\n\n"
        f"Paper: {chunk['source_file']}\n"
        f"Section: {chunk['section']}\n"
        f"Pages: {page}\n\n"
        f"Text:\n{chunk['text']}\n\n"
        f"Task: {_TYPE_INSTRUCTIONS[question_type]}\n\n"
        f"Requirements:\n"
        f"- The question must require reading this specific passage to answer correctly.\n"
        f"- The answer must be directly supported by the text above.\n"
        f"- Do NOT rephrase the question so that the answer is obvious from it.\n\n"
        f"Respond with ONLY valid JSON in exactly this format (no extra text):\n"
        f'{{\"question\": \"...\", \"expected_answer\": \"...\"}}'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()

        # Extract JSON blob from response (model may add preamble).
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            print(f"    [warn] No JSON found in response for chunk {chunk['chunk_id'][:8]}")
            return None

        parsed = json.loads(content[start:end])
        if not parsed.get("question") or not parsed.get("expected_answer"):
            return None

        return {
            "question": parsed["question"].strip(),
            "expected_answer": parsed["expected_answer"].strip(),
            "relevant_chunk_ids": [chunk["chunk_id"]],
            "source_file": chunk["source_file"],
            "section": chunk["section"],
            "question_type": question_type,
        }

    except Exception as exc:
        print(f"    [warn] QA generation failed for {chunk['chunk_id'][:8]}: {exc}")
        return None


def generate_candidates(model: str) -> None:
    """Generate eval candidates and save to eval_candidates.json."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. Run: python embedder.py"
        )

    chunks = _load_chunks()
    sampled = _sample_chunks_for_eval(chunks)

    print(f"Sampling from {len(sampled)} papers...")
    for paper, paper_chunks in sampled.items():
        print(f"  {paper}: {len(paper_chunks)} chunks selected")

    candidates: list[dict] = []
    qa_id = 1

    for paper, paper_chunks in sampled.items():
        print(f"\nGenerating QA pairs for: {paper}")
        for i, chunk in enumerate(paper_chunks):
            qtype = _QUESTION_TYPES[i % len(_QUESTION_TYPES)]
            print(
                f"  [{qa_id:2d}] {qtype:12} | section={chunk['section'][:45]}"
            )
            qa = _generate_qa_pair(chunk, qtype, model)
            if qa:
                qa["id"] = qa_id
                candidates.append(qa)
                print(f"         Q: {qa['question'][:80]}")
            else:
                print("         [skipped — generation failed]")
            qa_id += 1
            time.sleep(0.3)

    with CANDIDATES_FILE.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(candidates)} candidates to {CANDIDATES_FILE}")
    print(
        "\nNext steps:\n"
        "  1. Review eval_candidates.json — remove bad questions, fix answers.\n"
        "  2. Run: python eval.py --run-eval"
    )


def run_evaluation() -> None:
    """Load approved candidates, run retrieval, compute precision@3 and @5."""
    from retriever import retrieve

    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"{CANDIDATES_FILE} not found. Run: python eval.py --generate first."
        )

    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        candidates: list[dict] = json.load(f)

    if not candidates:
        raise ValueError(f"{CANDIDATES_FILE} is empty.")

    print(f"Running eval on {len(candidates)} questions...\n")

    results: list[dict] = []
    hits_at_3 = 0
    hits_at_5 = 0

    for q in candidates:
        retrieved = retrieve(q["question"], top_k=5)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        relevant = set(q["relevant_chunk_ids"])

        hit3 = any(rid in relevant for rid in retrieved_ids[:3])
        hit5 = any(rid in relevant for rid in retrieved_ids[:5])

        if hit3:
            hits_at_3 += 1
        if hit5:
            hits_at_5 += 1

        status = "HIT@3" if hit3 else ("HIT@5" if hit5 else "MISS ")
        print(
            f"  [{q['id']:2d}] {status} | {q['question_type']:12} | "
            f"{q['source_file'][:38]} | {q['question'][:55]}"
        )

        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "question_type": q["question_type"],
                "source_file": q["source_file"],
                "hit_at_3": hit3,
                "hit_at_5": hit5,
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": list(relevant),
            }
        )

    n = len(candidates)
    p3 = hits_at_3 / n if n else 0.0
    p5 = hits_at_5 / n if n else 0.0

    print(f"\n{'=' * 55}")
    print(f"Precision@3 : {p3:.3f}  ({hits_at_3}/{n})")
    print(f"Precision@5 : {p5:.3f}  ({hits_at_5}/{n})")

    # Breakdown by question type.
    print("\nBy question type:")
    for qtype in _QUESTION_TYPES:
        tr = [r for r in results if r["question_type"] == qtype]
        if not tr:
            continue
        t3 = sum(r["hit_at_3"] for r in tr)
        t5 = sum(r["hit_at_5"] for r in tr)
        nt = len(tr)
        print(f"  {qtype:12}: P@3={t3/nt:.3f} ({t3}/{nt})  P@5={t5/nt:.3f} ({t5}/{nt})")

    output = {
        "total_questions": n,
        "precision_at_3": round(p3, 4),
        "precision_at_5": round(p5, 4),
        "hits_at_3": hits_at_3,
        "hits_at_5": hits_at_5,
        "by_type": {
            qtype: {
                "precision_at_3": round(
                    sum(r["hit_at_3"] for r in results if r["question_type"] == qtype)
                    / max(1, sum(1 for r in results if r["question_type"] == qtype)),
                    4,
                ),
                "precision_at_5": round(
                    sum(r["hit_at_5"] for r in results if r["question_type"] == qtype)
                    / max(1, sum(1 for r in results if r["question_type"] == qtype)),
                    4,
                ),
            }
            for qtype in _QUESTION_TYPES
        },
        "results": results,
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {RESULTS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG retrieval evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate",
        action="store_true",
        help="Generate eval candidate Q&A pairs via Ollama.",
    )
    group.add_argument(
        "--run-eval",
        action="store_true",
        help="Run precision evaluation on approved eval_candidates.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Groq model name (default: llama3-70b-8192).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk sampling (default: 42).",
    )
    args = parser.parse_args()

    if args.generate:
        model = args.model or "llama-3.3-70b-versatile"
        print(f"Using Groq model: {model}\n")
        generate_candidates(model)
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
