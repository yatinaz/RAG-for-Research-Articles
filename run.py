#!/usr/bin/env python
"""
run.py - CLI entry point for the RAG PDF chunking pipeline.

Usage examples:
  # Process all PDFs in a directory, output to chunks.jsonl
  python run.py --input ./pdfs --output chunks.jsonl

  # Custom token targets
  python run.py --input ./pdfs --output chunks.jsonl \\
      --target-tokens 300 --overlap-tokens 80

  # Analyze an existing output file
  python run.py --analyze chunks.jsonl

  # Process and immediately analyze
  python run.py --input ./pdfs --output chunks.jsonl --analyze

  # Print 3 random sample chunks
  python run.py --input ./pdfs --output chunks.jsonl --sample 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import textwrap
from pathlib import Path

import chunker as ck


def _print_analysis(stats: dict) -> None:
    """Pretty-print the analysis stats."""
    print("\n" + "=" * 70)
    print("  RAG CHUNK QUALITY ANALYSIS")
    print("=" * 70)

    print(f"\nTotal chunks : {stats['total_chunks']}")
    print(f"Total files  : {stats['total_files']}")
    print(f"Sections     : {stats['sections_detected']}")

    ts = stats["token_stats"]
    print("\n--- Token Distribution ---")
    print(f"  Mean   : {ts['mean']}")
    print(f"  Median : {ts['median']}")
    print(f"  Stdev  : {ts['stdev']}")
    print(f"  Min    : {ts['min']}")
    print(f"  Max    : {ts['max']}")
    print(f"  P10    : {ts['p10']}")
    print(f"  P25    : {ts['p25']}")
    print(f"  P75    : {ts['p75']}")
    print(f"  P90    : {ts['p90']}")

    print("\n--- Per-File Summary ---")
    for fname, fs in stats["per_file"].items():
        print(f"  {fname[:55]:<55}  chunks={fs['chunk_count']:>4}  "
              f"mean={fs['mean_tokens']:>6}  min={fs['min_tokens']:>4}  max={fs['max_tokens']:>4}")

    print("\n--- Sections Detected (first 30) ---")
    for sec in stats.get("section_names", []):
        print(f"  * {sec}")

    print("\n--- Quality Flags ---")
    for flag in stats.get("quality_flags", []):
        marker = "[OK]" if flag.startswith("OK") else "[!!]"
        print(f"  {marker}  {flag}")

    print("=" * 70 + "\n")


def _print_sample(jsonl_path: Path, n: int) -> None:
    """Print n random chunks from the output file."""
    chunks = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    sample = random.sample(chunks, min(n, len(chunks)))
    print(f"\n--- {len(sample)} Sample Chunks ---")
    for c in sample:
        print(f"\n[chunk_id={c['chunk_id'][:12]}...  file={c['source_file']}  "
              f"pages={c['page_start']}-{c['page_end']}  tokens={c['token_count']}]")
        wrapped = textwrap.fill(c["text"][:400], width=80, initial_indent="  ",
                                subsequent_indent="  ")
        print(wrapped)
        if len(c["text"]) > 400:
            print("  [...]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RAG-ready PDF chunking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Directory containing PDF files, or a single PDF path.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Output JSONL file path (default: chunks.jsonl).",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=380,
        help="Target tokens per chunk (default: 380).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Hard maximum tokens per chunk (default: 512).",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=60,
        help="Overlap tokens between consecutive chunks (default: 60).",
    )
    parser.add_argument(
        "--garble-threshold",
        type=float,
        default=0.02,
        metavar="RATIO",
        help=(
            "Drop figure chunks whose space-char ratio is below this value "
            "(0.0–1.0). Words merged without spaces score near 0. Default: 0.02."
        ),
    )
    parser.add_argument(
        "--analyze",
        nargs="?",
        const=True,
        metavar="JSONL_PATH",
        help=(
            "Analyze chunk quality. Pass a path to analyze an existing file, "
            "or use without argument to analyze the current --output after processing."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Print N random sample chunks after processing.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    # ---- Analyze-only mode ---------------------------------------------------
    if args.analyze and args.analyze is not True:
        analyze_path = Path(args.analyze)
        if not analyze_path.exists():
            print(f"ERROR: file not found: {analyze_path}", file=sys.stderr)
            return 1
        stats = ck.analyze_output(analyze_path)
        _print_analysis(stats)
        if args.sample:
            _print_sample(analyze_path, args.sample)
        return 0

    # ---- Processing mode -----------------------------------------------------
    if args.input is None:
        parser.error("--input is required when not in analyze-only mode.")

    config = ck.ChunkingConfig(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        garble_threshold=args.garble_threshold,
    )

    if args.input.is_dir():
        chunks_iter = ck.process_directory(args.input, config)
    elif args.input.is_file() and args.input.suffix.lower() == ".pdf":
        chunks_iter = ck.process_pdf(args.input, config)
    else:
        print(f"ERROR: --input must be a directory or a .pdf file: {args.input}",
              file=sys.stderr)
        return 1

    total = ck.write_jsonl(chunks_iter, args.output)
    print(f"\nWrote {total} chunks -> {args.output}")

    if args.analyze or args.sample:
        stats = ck.analyze_output(args.output)
        if args.analyze:
            _print_analysis(stats)
        if args.sample:
            _print_sample(args.output, args.sample)

    return 0


if __name__ == "__main__":
    sys.exit(main())
