"""
audit_eval.py — One-shot audit + cleanup of eval_candidates.json.

Produces eval_candidates_v2.json and prints a pruning log.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CANDIDATES_FILE = Path("eval_candidates.json")
OUT_FILE = Path("eval_candidates_v2.json")
CHUNKS_FILE = Path("chunks.jsonl")
MODEL = "llama-3.3-70b-versatile"

# ── Audit decisions ───────────────────────────────────────────────────────────

REMOVE = {
    3:  "locating — answer is just filename+section, tests no content",
    6:  "duplicate of id14 (same question, same answer; 14 from primary paper kept)",
    7:  "locating — chunk text is about D-luciferin/mice dissociation, not TRIzol; wrong source",
    11: "locating — answer is 'This paper, Discussion section p.16'; self-referential",
    13: "duplicate of id8 (identical question+answer; 8 kept as it appears in Conclusions)",
    15: "locating — chunk is garbled figure caption, not about Seahorse measurements",
    19: "locating — answer is filename+section only, no substantive content",
    22: "locating — answer is filename+page, answer inferable from question",
    23: "metadata — tests publication date, inferable without retrieval",
}

# Questions that stay as-is (keep list, for clarity)
KEEP = {1, 2, 4, 5, 8, 9, 10, 12, 14, 16, 17, 18, 20, 21}

# Papers that end up with < 3 questions after pruning → need replacements
NEED_REPLACEMENT = {
    "40478_2019_Article_712.pdf": 1,          # has 5, 8 → need 1 more
    "inhibition-of-mitochondrial-respiration-prevents-braf-mutant-melanoma-brain-metastasis.pdf": 1,  # has 14, 16 → need 1 more
    "permissive-zones-for-the-centromere-binding-protein-parb-on-the-caulobacter-crescentus-chromosome.pdf": 1,  # has 17, 18 → need 1 more
    "taf6delta-controls-apoptosis-and-gene-expression-in-the-absence-of-p53.pdf": 1,  # has 20, 21 → need 1 more
}


def _load_chunks() -> list[dict]:
    chunks: list[dict] = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _pick_replacement_chunk(
    chunks: list[dict],
    paper: str,
    used_chunk_ids: set[str],
) -> dict | None:
    """Pick richest unused text chunk from paper with token_count >= 100."""
    candidates = [
        c for c in chunks
        if c["source_file"] == paper
        and c["chunk_type"] == "text"
        and c["token_count"] >= 100
        and c["chunk_id"] not in used_chunk_ids
        # Skip sections already heavily sampled
        and c["section"] not in {"Abstract", "abstract"}
    ]
    if not candidates:
        # Relax section filter
        candidates = [
            c for c in chunks
            if c["source_file"] == paper
            and c["chunk_type"] == "text"
            and c["token_count"] >= 100
            and c["chunk_id"] not in used_chunk_ids
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["token_count"])


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
}


def _generate_qa(chunk: dict, qtype: str, client: Groq) -> dict | None:
    page = (
        f"p.{chunk['page_start']}"
        if chunk["page_start"] == chunk["page_end"]
        else f"pp.{chunk['page_start']}-{chunk['page_end']}"
    )
    prompt = (
        f"You are building a retrieval evaluation dataset.\n\n"
        f"Paper: {chunk['source_file']}\n"
        f"Section: {chunk['section']}\n"
        f"Pages: {page}\n\n"
        f"Text:\n{chunk['text']}\n\n"
        f"Task: {_TYPE_INSTRUCTIONS[qtype]}\n\n"
        f"Requirements:\n"
        f"- The question must require reading this passage to answer correctly.\n"
        f"- The answer must be directly supported by the text above.\n"
        f"- Do NOT make the answer obvious from the question alone.\n\n"
        f"Respond with ONLY valid JSON:\n"
        f'{{\"question\": \"...\", \"expected_answer\": \"...\"}}'
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
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
            "question_type": qtype,
        }
    except Exception as exc:
        print(f"  [warn] generation failed: {exc}")
        return None


def main() -> None:
    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        original: list[dict] = json.load(f)

    chunks = _load_chunks()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # ── Print pruning log ─────────────────────────────────────────────────────
    print("=" * 70)
    print("PRUNING LOG")
    print("=" * 70)
    for q in original:
        if q["id"] in REMOVE:
            print(f"  REMOVE id={q['id']:2d} | {q['question'][:60]!r}")
            print(f"         reason: {REMOVE[q['id']]}")
    print()

    # ── Build kept set ────────────────────────────────────────────────────────
    kept: list[dict] = [q for q in original if q["id"] in KEEP]
    used_chunk_ids: set[str] = {
        cid for q in kept for cid in q["relevant_chunk_ids"]
    }

    print(f"Kept {len(kept)} questions after pruning.\n")

    # ── Generate replacements ─────────────────────────────────────────────────
    replacements: list[dict] = []
    next_id = max(q["id"] for q in original) + 1
    qtypes_cycle = ["factual", "comparative"]

    for i, (paper, count) in enumerate(NEED_REPLACEMENT.items()):
        print(f"Generating {count} replacement(s) for: {paper}")
        for j in range(count):
            chunk = _pick_replacement_chunk(chunks, paper, used_chunk_ids)
            if chunk is None:
                print(f"  [warn] no unused chunk found for {paper}")
                continue
            qtype = qtypes_cycle[(i + j) % len(qtypes_cycle)]
            print(f"  chunk={chunk['chunk_id'][:8]} section={chunk['section'][:40]} tokens={chunk['token_count']} type={qtype}")
            qa = _generate_qa(chunk, qtype, client)
            if qa:
                qa["id"] = next_id
                next_id += 1
                replacements.append(qa)
                used_chunk_ids.add(chunk["chunk_id"])
                print(f"  Q: {qa['question'][:80]}")
            else:
                print(f"  [skipped — generation failed]")
            time.sleep(0.4)
        print()

    # ── Assemble final set ────────────────────────────────────────────────────
    final = kept + replacements

    # Re-number sequentially
    for idx, q in enumerate(final, 1):
        q["id"] = idx

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Final question count : {len(final)}")
    print(f"Saved to             : {OUT_FILE}")
    print()

    # ── Summary by paper ──────────────────────────────────────────────────────
    by_paper: dict[str, int] = {}
    for q in final:
        by_paper[q["source_file"]] = by_paper.get(q["source_file"], 0) + 1
    print("By paper:")
    for paper, count in by_paper.items():
        print(f"  {count}  {paper}")


if __name__ == "__main__":
    main()
