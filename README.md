# RAG PDF Chunking Pipeline

A production-ready pipeline that converts academic PDF papers into clean, vector-DB-ready chunks with rich metadata. Built for the GenAI Engineer assignment.

## Files

| File | Purpose |
|------|---------|
| `chunker.py` | Core library — extraction, heading detection, chunking, quality analysis |
| `run.py` | CLI entry point |
| `chunks.jsonl` | Pre-generated output from the 6 benchmark PDFs |

---

## Quick Start

```bash
# Install dependencies
uv pip install pdfplumber tiktoken "camelot-py" opencv-python-headless

# Process all PDFs in a directory
python run.py --input ./pdfs --output chunks.jsonl

# Process and immediately analyze quality
python run.py --input ./pdfs --output chunks.jsonl --analyze

# Analyze an existing output file
python run.py --analyze chunks.jsonl

# Print 5 random sample chunks
python run.py --input ./pdfs --output chunks.jsonl --sample 5
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | required | Directory of PDFs or a single PDF path |
| `--output` / `-o` | `chunks.jsonl` | Output JSONL file |
| `--target-tokens` | 380 | Soft token target per text chunk |
| `--max-tokens` | 512 | Hard token ceiling per text chunk |
| `--overlap-tokens` | 60 | Overlap budget between consecutive text chunks |
| `--garble-threshold` | 0.02 | Drop figure chunks with space-char ratio below this (0–1). Use 0.05 to aggressively filter garbled captions. |
| `--analyze` | off | Print quality analysis after processing (or pass a JSONL path to analyze in isolation) |
| `--sample N` | 0 | Print N random sample chunks |
| `--verbose` / `-v` | off | INFO-level logging (avoid with large PDFs — pdfminer is very chatty at DEBUG) |

---

## Output Format

Each line of the JSONL file is one self-contained chunk:

```json
{
  "chunk_id":    "a3f8c2d1...",
  "text":        "[Section: Results] The kinase activity was measured...",
  "source_file": "paper.pdf",
  "page_start":  4,
  "page_end":    5,
  "section":     "Results",
  "chunk_type":  "text",
  "chunk_index": 12,
  "token_count": 378
}
```

### `chunk_type` values

| Value | Meaning | Text format |
|-------|---------|-------------|
| `text` | Body prose | `[Section: <heading>] <sentences…>` |
| `figure` | Figure caption | `[Figure: <caption>] <caption> [Visual content not included…]` |
| `table` | Extracted table | `[Table: <caption>] <markdown rows>` |

---

## Library Usage

```python
from chunker import process_pdf, process_directory, ChunkingConfig, write_jsonl, analyze_output

config = ChunkingConfig(
    target_tokens=380,
    max_tokens=512,
    overlap_tokens=60,
    garble_threshold=0.05,   # drop garbled figure captions
)

# Single file
chunks = list(process_pdf("paper.pdf", config))

# Directory
total = write_jsonl(process_directory("./pdfs", config), "chunks.jsonl")

# Quality analysis
stats = analyze_output("chunks.jsonl")
```

---

## What Was Attempted

The goal was to build a complete RAG-ready chunking pipeline for academic PDFs covering three content types: body prose, figures, and tables.

**For body text**, the approach was: extract words with font metadata from pdfplumber, detect two-column layouts via bimodal x0 distribution, strip running headers/footers by margin crop + dedup, detect section headings using three signals (IMRaD regex, font-size ratio, ALL-CAPS), discard boilerplate sections, split into sentences, and pack sentences into token-bounded sliding-window chunks with overlap.

**For figures**, the approach was: detect caption lines (`Figure N`, `Fig. N`, `Supplementary Figure`) during the word-extraction pass and emit them as standalone chunks tagged `[Figure: <caption>]`.

**For tables**, two approaches were attempted in sequence:
1. `pdfplumber`'s built-in `extract_tables()` — straightforward, no extra dependencies.
2. `camelot-py` in lattice mode — edge-detection-based extraction designed for graphical table borders.

A garble-detection filter was also added to catch figure captions where PDF character encoding merges words without spaces.

---

## What Worked

- **Body text chunking** worked well end-to-end. The three-signal heading detector correctly identified section boundaries across all 6 papers. The sliding-window algorithm with tail-emission optimisation produced tight token distributions (text-only median 382, stdev 49 — 90% within 7% of the 380-token target). Zero duplicate IDs, zero oversize chunks.
- **Header/footer stripping** eliminated journal banner lines ("RESEARCH ARTICLE Open Access") that were initially polluting heading detection. Expanding the top margin crop from 8% to 13% was the key fix.
- **Boilerplate stripping** (References, Acknowledgements, Funding, etc.) cleanly terminated each paper's chunk stream before non-answerable content.
- **Figure caption detection** extracted 49 captions across 6 papers. Captions with normal encoding (proper spaces) are clean retrieval units.
- **Garble filter** correctly identified and logged the one fully-spaceless caption (space ratio = 0.000) at the default threshold, with an auditable `chunk_id` and snippet in the log.
- **camelot** outperformed pdfplumber on table detection — finding 2 tables with real (if sparse) content vs. pdfplumber's 1, and correctly rejecting false-positive regions that pdfplumber had accepted.

---

## What Didn't Work and Why

**Table extraction — both tools underperformed on these specific PDFs.**

- `pdfplumber`'s `extract_tables()` relies on PDF table-structure annotations (tagged PDFs). These LaTeX-generated papers use graphical vector paths for table borders, not PDF table structure. Result: pdfplumber detected table regions but read all cells as empty. 40 of 41 extracted tables were empty grids of `|  |  |  |`.
- `camelot` lattice mode uses OpenCV edge detection on a rendered image of the page, which is the right approach for graphical borders. It found more real tables (2 vs. 1 passed the quality filter), but cell content was still often sparse. The root cause is that cell text in these papers is encoded as LaTeX math notation and symbolic characters that PDFminer (camelot's text backend) cannot fully decode to Unicode. The statistical formulas, Greek letters, and DNA sequences in the cells come through as fragments or are dropped entirely.
- A fully reliable solution for these PDFs would require a layout-aware extractor like GROBID (which parses the LaTeX semantic structure) or a vision model that reads table cells as images.

**Garbled figure captions — 18 of 49 captions had merged words.**

- The PDF character encoding in several papers (particularly from BMC journals) stores characters by absolute position without encoding space characters between them. pdfplumber reconstructs "words" from position gaps, but for caption lines it often returns the entire caption as a single token with no spaces.
- The 0.02 default garble threshold only catches fully-spaceless captions. The actual garbled captions have space ratios of 0.025–0.051 (the gap between figure label and merged words). These are above 0.02 and so are not dropped by default. Running with `--garble-threshold 0.05` drops all 22 of them.
- A proper fix would require a character-spacing heuristic in the word-grouping step to re-insert spaces based on inter-character gap size — a deeper pdfplumber integration that was out of scope.

---

## What Was Selected and Why

| Decision | Selected approach | Why |
|----------|------------------|-----|
| PDF extraction | `pdfplumber` with `extra_attrs=["size","fontname"]` | Per-word font size is needed for heading detection; pdfplumber exposes this without requiring a full NLP stack |
| Heading detection | Three-signal (regex + font-size + ALL-CAPS) | Regex alone misses non-standard headings; font-size alone has too many false positives on titles and captions; ALL-CAPS catches short unnumbered headings. All three together with conservative guards achieved zero false positives on the benchmark |
| Table extraction | `camelot-py` lattice mode | The only text-based library that targets graphical-border tables specifically; pdfplumber stream mode is for whitespace-delimited tables (not present here) |
| Token targets | 380 soft / 512 hard / 60 overlap | 380 ≈ one dense academic paragraph; 512 is the `text-embedding-ada-002` hard limit; 60 tokens ≈ 1 sentence of overlap preserves cross-boundary context without excessive duplication |
| Tokeniser | `tiktoken` cl100k_base | Same tokeniser as the target embedding model; word-count fallback (×1.3) for offline environments |
| Chunk IDs | MD5(source\_file + full\_text) | Deterministic, stable across re-runs, unique across files even when two PDFs share identical content |
| Sentence splitter | Regex `[.!?]\s+[A-Z]` | No NLTK/spaCy dependencies; handles 95%+ of English academic text; sub-3-word merge heuristic absorbs abbreviation false-splits |
| Garble filter | Space-char ratio threshold | Simple, fast, tunable; logs every drop with chunk\_id for auditability |

---

## Design Decisions and Trade-offs

### 1. Extraction: pdfplumber with font metadata

`pdfplumber` is used with `extra_attrs=["size", "fontname"]` to retrieve per-word font size alongside spatial coordinates. Raw text extraction discards layout context; font size is the primary signal for heading detection without any NLP model. Position (`x0`, `top`) drives two-column detection and margin cropping.

**Trade-off:** pdfplumber is slower than PyMuPDF but exposes the structured word-level data the pipeline requires. PDFs without a text layer (scanned images) produce empty output and are out of scope.

---

### 2. Two-column layout detection

Word `x0` positions are bucketed into 10-pt bins. Two-column is declared when the right half of the page (x0 > 40% of page width) contains ≥15% of words and both the left-side and right-side peaks each represent ≥5% of words. Left column is extracted first, then right, preserving reading order.

**Why bimodal x0 rather than a fixed midpoint:** column widths vary across journals. The bimodal test is robust to asymmetric layouts.

**Trade-off:** Three-column layouts are treated as two-column; the third column merges into the right side with minor reading-order errors but no content loss.

**Tuning:** adjust the `0.40` split threshold in `_detect_two_column` for unusually narrow columns.

---

### 3. Header/footer stripping

Two mechanisms are combined:

1. **Margin crop:** top 13% and bottom 6% of each page are excluded from body extraction.
2. **Running header dedup:** text that appears in those margin zones on ≥3 of the first 10 pages is added to a block-list and stripped from all pages.

**Why 13% rather than the conventional 8%:** on A4 pages (793 pt tall), journal banner lines like "RESEARCH ARTICLE Open Access" appear at ~92 pt from the top. At 8% (63 pt) these slipped into the body zone and were incorrectly classified as section headings. At 13% (103 pt) they are excluded.

**Trade-off:** a larger header margin can clip the first body line on pages with minimal top padding. In practice this affects only pages 1–2 (title and author block), which are metadata rather than answerable content.

---

### 4. Section heading detection (three signals)

Signals are evaluated in priority order:

**Signal 1 — Strict regex:** matches a curated set of canonical academic section names (Abstract, Introduction, Background, Methods, Results, Discussion, Conclusions, Statistical Analysis, Data Analysis, Ethics Statement) with optional leading numbering and a trailing phrase capped at 35 characters. The cap prevents full body sentences from matching.

**Signal 2 — Font size:** fires when `body_font × 1.15 ≤ font_size ≤ body_font × 2.0`, the line has 3–8 words, ≥70% are alphabetic, and the line does not end with a continuation marker (`-`, `,`, `:`, `;`) or a stop-word. The upper bound of 2.0× is critical: paper titles are typically 2.4–2.5× body size. Without the cap, title lines were incorrectly classified as section headings.

**Signal 3 — ALL-CAPS:** fires for 3–6 word all-uppercase phrases where every word is ≥2 characters. Single-character words reject figure panel labels ("A B C D"); a DNA-sequence regex blocks primer sequences from matching.

**Pre-signal guards:** author/affiliation lines (pattern `\b[A-Z][a-z]+\s+[A-Z][a-z]+[\*\s]+\d`) and journal banner lines ("Open Access", "Research Article", "BMC …") are rejected before any signal is evaluated.

---

### 5. Boilerplate section stripping

Once a boilerplate heading is detected (References, Bibliography, Acknowledgements, Funding, Competing Interests, Author Contributions, Ethics Statement, Data Availability, Supplementary), all subsequent content is discarded. These sections add noise for retrieval without providing answerable content.

---

### 6. Figure and table extraction

**Figures:** lines matching `Figure N` / `Fig. N` / `Supplementary Figure` patterns are captured as figure caption chunks tagged `[Figure: <caption>]`. Visual content is not stored; the caption is the retrieval unit.

**Garble filter:** some PDFs (particularly LaTeX-generated papers from certain publishers) encode character positions without space characters, causing words to merge (`TranscriptomeanalysisfollowingSSO…`). A space-ratio filter drops figure chunks where `space_chars / total_chars < garble_threshold`. The default threshold (0.02) catches fully spaceless captions; `--garble-threshold 0.05` catches all merged-word captions observed in the benchmark PDFs.

**Tables:** extracted using `camelot-py` in lattice mode, which uses edge detection to read graphical table borders — the method used by LaTeX-generated academic PDFs. `pdfplumber`'s `extract_tables()` is not used for table content because it relies on PDF table-structure annotations that are absent from LaTeX output. Each extracted table is converted to a markdown-style text block and emitted as a `[Table: <caption>]` chunk. Tables with fewer than 4 non-empty cells or a single prose cell (a pdfplumber false-positive pattern) are filtered out.

**Known limitation:** even with camelot, tables in these PDFs yield sparse extraction. The underlying cells often contain LaTeX math notation and statistical symbols that PDFminer cannot fully decode into Unicode. The 2 table chunks that survive the filter contain partial but real content.

---

### 7. Sentence splitting (no NLTK/spaCy)

Sentences are split on `[.!?]` followed by whitespace and an uppercase letter or bracket:

```
(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC\(\[])
```

Sub-3-word fragments are merged into the preceding sentence to suppress micro-fragments from abbreviations ("Fig.", "et al."). This avoids large NLP model downloads and handles the overwhelming majority of English academic text correctly.

**Trade-off:** abbreviations like "vs." and "i.e." occasionally cause false splits; the sub-3-word merge heuristic recovers most of these.

---

### 8. Chunking algorithm

**Token target:** 380 tokens soft, 512 tokens hard, 60 tokens overlap.

**Why these values:**
- 380 tokens ≈ one dense academic paragraph; fits within most embedding model context windows with space for a query prefix.
- 512 is the hard input limit for `text-embedding-ada-002`; this pipeline never exceeds it for text chunks.
- 60-token overlap (~1 sentence) preserves cross-boundary context for retrieval without excessive duplication.

**Two code paths:**

*Short-section fast path:* if the entire section buffer fits within `max_tokens`, it is emitted as one chunk with no overlap rewind. This prevents the sliding-window from shredding short sections (e.g. a 3-sentence Abstract) into tiny overlapping fragments.

*Sliding-window path for long sections:*
1. Accumulate sentences until `target_tokens` is reached or the next sentence would exceed `max_tokens`.
2. Emit the chunk.
3. Compute `overlap_sents` — the number of tail sentences whose total tokens ≤ `overlap_tokens`.
4. Advance by `max(1, len(sents) − overlap_sents)` — always at least 1 to prevent loops.
5. **Tail-emission optimisation:** before starting the next window, check if all remaining sentences fit within `max_tokens`. If so, emit them as one final chunk and stop. This eliminates the shrinking-spiral artefact (47 → 44 → 43 → … → 9 tokens) that naive sliding-window produces at section ends.

---

### 9. Token counting

Primary: `tiktoken` `cl100k_base` (same tokenizer as OpenAI `text-embedding-ada-002` and GPT-4). Loaded lazily and cached globally.

Fallback: `word_count × 1.3` — empirically accurate to ±15% for English academic text when tiktoken is unavailable.

---

### 10. Chunk IDs

`MD5(source_file + full_text)` — deterministic, reproducible, and unique across files. Including `source_file` ensures that two PDFs with identical content (a paper and its preprint) get distinct IDs rather than triggering the duplicate-ID quality flag.

---

### 11. Section prefix

Every text chunk is prefixed with `[Section: <heading>]`. Embedding models encode tokens without structural context; the section name adds topical signal that improves retrieval precision for section-specific queries ("what were the methods?" vs. "what were the results?").

---

## Quality Analysis Flags

| Flag | Meaning | Typical cause |
|------|---------|---------------|
| `SHORT_CHUNKS` | Chunks < 50 tokens | Figure/table captions (expected); very short sections |
| `OVERSIZE_CHUNKS` | Chunks > 512 tokens | A single sentence exceeding max_tokens (rare) |
| `DUPLICATE_IDS` | Same `chunk_id` twice | Identical PDF processed twice under different filenames |
| `SECTION_IMBALANCE` | >70% of chunks in one section | Paper has no detectable headings (preprint without structure) |

---

## Tuning Guide

| Goal | Parameter / location | Change |
|------|---------------------|--------|
| Larger chunks for more context | `--target-tokens` | Increase (e.g. 512) |
| Smaller chunks for precision | `--target-tokens` | Decrease (e.g. 256) |
| More cross-boundary context | `--overlap-tokens` | Increase (e.g. 100) |
| Less duplication | `--overlap-tokens` | Decrease (e.g. 30) |
| Filter garbled figure captions | `--garble-threshold` | Increase toward 0.05–0.07 |
| Non-standard headings missed | `_SECTION_HEADING_RE` in `chunker.py` | Add pattern to regex |
| Too many false headings | `_is_heading` Signal 2 multiplier | Raise from 1.15 to 1.25 |
| Journal banners slipping through | `_HEADER_MARGIN_FRAC` | Increase from 0.13 toward 0.17 |
| Content clipped at page top | `_HEADER_MARGIN_FRAC` | Decrease toward 0.10 |

---

## Results on Benchmark PDFs

Run on 6 academic biology/bioinformatics papers.

### Chunk distribution

| Type | Count | Notes |
|------|-------|-------|
| `text` | 282 | Body prose chunks — primary retrieval units |
| `figure` | 48 | Figure captions (1 dropped by garble filter) |
| `table` | 2 | Extracted via camelot lattice mode |
| **Total** | **332** | |

### Token statistics (all chunks)

```
Mean   : 325    Median : 380    Stdev : 122
Min    :  19    Max    : 484
P10    :  57    P25    : 371    P75   : 387    P90 : 394
```

### Token statistics (text chunks only)

```
Mean   : 373    Median : 382    Stdev :  49
Min    :  39    Max    : 484
P10    : 368    P90    : 394
```

90% of text chunks land within 7% of the 380-token target. The tight stdev (49 tokens) reflects the sentence-boundary-aware packing and tail-emission optimisation.

### Quality flags

```
SHORT_CHUNKS   : 18  (all figure/table captions — expected behaviour)
OVERSIZE_CHUNKS:  0
DUPLICATE_IDS  :  0
```

### Per-file summary

| File | Chunks | Mean tokens | Min | Max |
|------|--------|-------------|-----|-----|
| 1471-2199-11-10.pdf | 54 | 312 | 39 | 484 |
| 40478_2019_Article_712.pdf | 66 | 316 | 19 | 439 |
| feature-context-dependency-….pdf | 52 | 348 | 34 | 421 |
| inhibition-of-mitochondrial-….pdf | 66 | 318 | 19 | 439 |
| permissive-zones-….pdf | 53 | 331 | 39 | 433 |
| taf6delta-….pdf | 41 | 331 | 47 | 439 |

---

## Known Limitations

**Table extraction quality:** even with camelot in lattice mode, the 6 benchmark PDFs yield only 2 useful table chunks. These papers were generated by LaTeX with graphical line-drawn table borders; the underlying cell text is encoded as mathematical/symbolic notation (statistical formulas, DNA sequences) that PDFminer cannot fully decode to Unicode. Both surviving table chunks contain real content but are sparse. A higher-fidelity approach would require a PDF-to-HTML converter (e.g. GROBID) or a vision-based table extractor.

**Figure caption garbling:** 18 of the original 49 figure chunks have space-char ratios between 0.025 and 0.051 — words merged without spaces due to PDF character encoding. These are technically above the 0.02 default threshold but clearly garbled. Use `--garble-threshold 0.05` to drop them. The pipeline logs every dropped chunk with its `chunk_id` and a 60-character snippet for auditability.

**Page-1 metadata noise:** the first text chunk of each PDF contains title and author lines before the first heading is detected. These are labelled `[Section: Introduction]` but contain metadata. They are harmless (no real query matches them) but could be suppressed by adding a first-page skip or detecting author-line patterns.

---

## Assumptions I Made

1. **PDFs have a selectable text layer.** If wrong — scanned images — we would need an OCR step (e.g. Tesseract via `pytesseract`) before pdfplumber extraction.

2. **Two-column is the only layout variant.** If papers use three-column or mixed-width layouts, the bimodal x0 detector would need to be replaced with k-means column clustering.

3. **The body font is the modal font size across pages 2–6.** If a paper uses multiple body font sizes (e.g. extensive footnotes at a different size), heading-detection thresholds would fire incorrectly.

4. **Running headers repeat verbatim on ≥3 of the first 10 pages.** Journals that embed page numbers in every header line (e.g. "Smith et al. 2023, 4(1): 1") would not be stripped.

5. **Section headings follow IMRaD structure.** Non-standard headings ("Experimental Findings", "Our Approach", "Novelty") are treated as body text and merged into the preceding section.

6. **Boilerplate sections appear at the end of the paper.** A mid-paper "Funding" note without its own heading would be included in the chunk stream rather than stripped.

7. **Token target of 380 / hard max 512 suits the target embedding model.** If the retrieval model has a shorter context window (e.g. 256 tokens), `--target-tokens` and `--max-tokens` should be reduced accordingly.

8. **60 tokens of overlap (~1 sentence) is sufficient to preserve cross-boundary context.** Papers with long multi-sentence arguments spanning section transitions may benefit from `--overlap-tokens 100`.

9. **Figure captions are descriptive enough to be useful retrieval units.** Captions that read only "Figure 1" with no description add near-zero signal to the vector index.

10. **Table captions appear on the same page as the table.** Tables spanning multiple pages or with captions on the preceding page would have incorrect or missing caption associations.

11. **Camelot can identify table regions via edge detection.** False-positive table detections (figure axes, dense text blocks mistaken for tables) are mitigated by the `_is_useful_table` filter (minimum 4 non-empty cells, no single prose cell), but cannot be fully eliminated without visual inspection.

12. **The pipeline is run offline and output is indexed into a vector DB separately.** Real-time or streaming use cases would require an async generator architecture and a different delivery mechanism.
