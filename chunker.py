"""
chunker.py - RAG-ready PDF chunking pipeline (importable library).

Design decisions
================
Extraction
----------
- pdfplumber provides per-word font size and x0 position metadata.
- Two-column detection: bucket word x0 positions into 10-pt bins. If a
  secondary peak exists past 40% of page width accounting for >=15% of
  words, treat as two-column and extract left column then right column.
- Running headers/footers: crop the top 8% and bottom 6% of each page,
  collect text that repeats verbatim on >=3 of the first 10 pages, and
  strip those strings from every subsequent page.

Heading detection (three signals, conservative)
-----------------------------------------------
1. Strict regex: only canonical academic section names (Abstract, Methods,
   Results, Discussion, etc.) with an optional short trailing phrase.
   Broad biology keywords ("DNA", "ChIP", "Feature") were removed after
   they matched body-text sentence starts in testing.
2. Font-size: size >= body_font * 1.15, AND >= 3 words, AND <= 8 words,
   AND >= 70% alphabetic words, AND does NOT end with a continuation
   marker (hyphen, comma, colon, semicolon) or a stop-word like "for/of".
3. All-caps: isupper(), >= 3 words, every word >= 2 chars (rejects figure
   panel labels like "A B C"), <= 6 words.

Author/affiliation lines are explicitly rejected via a pattern that matches
"Name Name <digit>" sequences.

Boilerplate stripping
---------------------
References, Acknowledgements, Funding, Competing Interests, Author
Contributions, Ethics, Data Availability, and Supplementary sections are
detected by regex and everything thereafter is discarded.

Chunking
--------
- Sentence splitting: regex split on [.!?] followed by whitespace + capital.
  Sub-3-word fragments are merged into the preceding sentence.
- If the entire section buffer fits within max_tokens (512), it is emitted
  as a single chunk with NO overlap rewind. This prevents the overlap
  mechanism from shredding short sections into tiny overlapping fragments.
- For long sections (buffer > max_tokens), chunks are built to target_tokens
  (380) and the next chunk starts (len(sents) - overlap_sents) positions
  forward, where overlap_sents is the number of tail sentences whose total
  token count <= overlap_tokens (60). Always advance >= 1.
- Every chunk is prefixed with [Section: <heading>].

Identifiers
-----------
- chunk_id: MD5 of (source_file + full_text) to guarantee uniqueness even
  when two PDFs share identical content.

Output
------
JSONL with fields: chunk_id, text, source_file, page_start, page_end,
section, chunk_index, token_count.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import pdfplumber

try:
    import camelot as _camelot
    _CAMELOT_AVAILABLE = True
except ImportError:
    _CAMELOT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
_enc = None


def _get_encoder():
    global _enc
    if _enc is not None:
        return _enc
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
        return _enc
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base, falling back to word*1.3."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    section: str
    chunk_type: str  # "text" | "table" | "figure"
    chunk_index: int
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Conservative: only unambiguous canonical academic section names.
# The optional suffix (?:\s+[A-Za-z][\w ]{0,35})? covers "and Discussion",
# "of Variance", "for Differential Expression", etc. — capped at 35 chars
# so it cannot swallow a full body-text sentence.
_SECTION_HEADING_RE = re.compile(
    r"^\s*"
    r"(?:\d{1,2}\.?\d*\.?\s*)?"
    r"(?:"
    r"Abstract|Introduction|Background|Overview|Summary|"
    r"Methods?|Materials\s+and\s+Methods?|Materials\s+Methods?|"
    r"Experimental\s+(?:Methods?|Procedures?|Section|Design)?|"
    r"Results?|Results?\s+and\s+Discussion|"
    r"Discussion|"
    r"Conclusions?|Concluding\s+Remarks?|"
    r"Statistical\s+(?:Analysis|Methods?)|"
    r"Data\s+(?:Analysis|Availability|Collection)|"
    r"Author\s+(?:Contributions?|Information|Summary)|"
    r"Ethics\s+(?:Statement|Declaration|Approval)"
    r")"
    r"(?:\s+[A-Za-z][\w ]{0,35})?"
    r"\s*$",
    re.IGNORECASE,
)

# Boilerplate: discard from this heading onward.
_BOILERPLATE_SECTIONS = re.compile(
    r"^\s*(?:"
    r"References?|Bibliography|"
    r"Acknowledgements?|Acknowledgments?|"
    r"Funding(?:\s+Information)?|"
    r"Competing\s+Interests?|Conflict(?:s)?\s+of\s+Interests?|"
    r"Author\s+Contributions?|"
    r"Ethics\s+(?:Statement|Approval|Declaration)|"
    r"(?:Data|Code)\s+Availability|"
    r"Supplementary\s+(?:Data|Materials?|Information|Figures?|Tables?)|"
    r"Additional\s+(?:files?|information|data)"
    r")\s*$",
    re.IGNORECASE,
)

# Figure captions — emitted as figure chunks.
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"Fig(?:ure)?\.?\s*\d+|"
    r"Additional\s+(?:file|Figure)\s*\d*|"
    r"Supplementary\s+(?:Figure|File)\s*\d*"
    r")",
    re.IGNORECASE,
)

# Table captions — associated with nearest extracted table.
_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"Table\s*\d+|"
    r"Additional\s+Table\s*\d*|"
    r"Supplementary\s+Table\s*\d*"
    r")",
    re.IGNORECASE,
)

# Combined, for quick rejection in body-text extraction.
_CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"Fig(?:ure)?\.?\s*\d+|"
    r"Table\s*\d+|"
    r"Additional\s+(?:file|Figure|Table)\s*\d*|"
    r"Supplementary\s+(?:Figure|Table|File)\s*\d*"
    r")",
    re.IGNORECASE,
)

# Standalone page numbers.
_PAGE_NUM_RE = re.compile(r"^\s*(?:Page\s+)?\d{1,4}(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)

# Author/affiliation lines: "Firstname Lastname 1" or "Name Name* 1,2,3"
_AUTHOR_LINE_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+[\*\s]+\d")

# Continuation-ending words: heading cannot end with these prepositions/articles
_CONTINUATION_ENDS = re.compile(
    r"\s+(?:for|of|the|and|or|in|with|to|a|an|by|on|at|from|into|that|which|"
    r"were|was|is|are|be|been)\s*$",
    re.IGNORECASE,
)

# Sentence splitter.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC\(\[])")

# DNA/RNA sequence pattern — all-uppercase nucleotide strings (ACGTN + spaces).
# Used to reject primer sequences from the all-caps heading heuristic.
_DNA_SEQ_RE = re.compile(r"^[ACGTNURYKMBDHVSWX\s]+$")

# Journal banner lines that should never be treated as section headings.
_JOURNAL_BANNER_RE = re.compile(
    r"(?:Open\s+Access|Research\s+Article|BMC\s+\w+|"
    r"UC\s+Davis|Previously\s+Published|Copyright\s+Information|"
    r"Journal\s+of\s+\w+)",
    re.IGNORECASE,
)

# Layout constants.
# 0.13 (~103 pt on A4) captures journal banner lines like "RESEARCH ARTICLE Open
# Access" that sit below the traditional 8% crop boundary.
_HEADER_MARGIN_FRAC = 0.13
_FOOTER_MARGIN_FRAC = 0.06


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _detect_two_column(words: list, page_width: float) -> bool:
    """True if word x0 positions show a bimodal distribution (two columns)."""
    if not words:
        return False
    bins: Counter = Counter(round(w["x0"] / 10) * 10 for w in words)
    total = sum(bins.values())
    right_total = sum(cnt for x, cnt in bins.items() if x > page_width * 0.40)
    if right_total < total * 0.15:
        return False
    left_peak = max((cnt for x, cnt in bins.items() if x < page_width * 0.40), default=0)
    right_peak = max((cnt for x, cnt in bins.items() if x > page_width * 0.40), default=0)
    return left_peak >= total * 0.05 and right_peak >= total * 0.05


def _group_words_into_lines(words: list, tol: float = 2.0) -> List[Tuple[str, float]]:
    """Group word dicts into lines by top-coordinate proximity. Returns (text, avg_size)."""
    if not words:
        return []
    lines: list = [[words[0]]]
    for word in words[1:]:
        if abs(word["top"] - lines[-1][0]["top"]) <= tol:
            lines[-1].append(word)
        else:
            lines.append([word])
    result = []
    for lw in lines:
        text = " ".join(w["text"] for w in lw)
        sizes = [w.get("size") or 0 for w in lw]
        result.append((text, sum(sizes) / len(sizes) if sizes else 0))
    return result


def _is_heading(text: str, font_size: float, body_font_size: float) -> bool:
    """
    Return True if the line is a section heading.

    Signal 1 - Regex: matches canonical academic section name.
    Signal 2 - Font size: enlarged AND enough real words AND no continuation markers.
    Signal 3 - ALL-CAPS: every word >= 2 chars, 3-6 words.
    """
    s = text.strip()
    if not s:
        return False

    # Reject author/affiliation lines like "Jane Doe 1,2"
    if _AUTHOR_LINE_RE.search(s):
        return False

    # Reject journal banner lines regardless of font size.
    if _JOURNAL_BANNER_RE.search(s):
        return False

    # Signal 1: known section name
    if _SECTION_HEADING_RE.match(s):
        return True

    # Headings always start with an uppercase letter or a digit.
    if s[0].islower():
        return False

    words = s.split()
    alpha_words = [w for w in words if re.match(r"^[A-Za-z\-\']+$", w)]
    word_count = len(words)

    # Signal 2: font-size enlarged but not title-scale (titles are ~2.5x body).
    # Upper bound of 2.0x excludes 24pt paper titles while keeping 11-14pt
    # section headings typical of academic PDFs.
    if (
        body_font_size > 0
        and body_font_size * 1.15 <= font_size <= body_font_size * 2.0
        and 3 <= word_count <= 8
        and len(alpha_words) >= word_count * 0.70
        and not s.endswith(("-", ",", ":", ";"))
        and not _CONTINUATION_ENDS.search(s)
    ):
        return True

    # Signal 3: all-caps short phrase.
    # Extra guards: no single-char words (eliminates "A B C" figure labels),
    # and not a DNA/RNA primer sequence (only ACGTN nucleotide chars).
    if (
        s.isupper()
        and 3 <= word_count <= 6
        and all(len(w) >= 2 for w in words)
        and len(alpha_words) >= 3
        and not _DNA_SEQ_RE.match(s)
    ):
        return True

    return False


def _estimate_body_font_size(pdf: pdfplumber.PDF, sample_pages: int = 5) -> float:
    """Modal font size over early pages. Falls back to 9.0."""
    counts: Counter = Counter()
    for page in pdf.pages[1: sample_pages + 1]:
        for w in page.extract_words(extra_attrs=["size"]):
            sz = w.get("size")
            if sz and sz > 5:
                counts[round(sz, 1)] += 1
    return counts.most_common(1)[0][0] if counts else 9.0


def _collect_running_headers(pdf: pdfplumber.PDF, sample: int = 10) -> Set[str]:
    """Text in header/footer margins that repeats on >= 3 pages."""
    candidates: Counter = Counter()
    for page in pdf.pages[: min(sample, len(pdf.pages))]:
        h, w = page.height, page.width
        for crop in (
            page.crop((0, 0, w, h * _HEADER_MARGIN_FRAC), strict=False),
            page.crop((0, h * (1 - _FOOTER_MARGIN_FRAC), w, h), strict=False),
        ):
            t = (crop.extract_text() or "").strip()
            if t:
                candidates[t] += 1
    return {t for t, n in candidates.items() if n >= 3}


def _extract_page_text_blocks(
    page: pdfplumber.page.Page,
    body_font_size: float,
    running_headers: Set[str],
) -> Tuple[List[Tuple[str, bool]], List[Tuple[str, str]]]:
    """
    Return (body_lines, captions).

    body_lines: list of (line_text, is_heading) for prose content.
    captions:   list of (caption_text, caption_type) where caption_type
                is "figure" or "table".
    """
    h, w = page.height, page.width
    all_words = page.extract_words(extra_attrs=["size", "fontname"], use_text_flow=True)
    if not all_words:
        return [], []

    two_col = _detect_two_column(all_words, w)
    regions = [(0, w / 2), (w / 2, w)] if two_col else [(0, w)]

    body_top = h * _HEADER_MARGIN_FRAC
    body_bot = h * (1 - _FOOTER_MARGIN_FRAC)
    body_lines: List[Tuple[str, bool]] = []
    captions: List[Tuple[str, str]] = []

    for x0, x1 in regions:
        crop = page.crop((x0, body_top, x1, body_bot), strict=False)
        col_words = crop.extract_words(extra_attrs=["size", "fontname"], use_text_flow=True)
        if not col_words:
            continue
        for line_text, line_size in _group_words_into_lines(col_words):
            s = line_text.strip()
            if not s:
                continue
            if _PAGE_NUM_RE.match(s):
                continue
            if s in running_headers:
                continue
            if _FIGURE_CAPTION_RE.match(s):
                captions.append((s, "figure"))
                continue
            if _TABLE_CAPTION_RE.match(s):
                captions.append((s, "table"))
                continue
            body_lines.append((s, _is_heading(s, line_size, body_font_size)))

    return body_lines, captions


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------
def _format_table(rows: list) -> str:
    """Convert pdfplumber table rows (list-of-lists) to a markdown-style text block."""
    if not rows:
        return ""
    cleaned = []
    for row in rows:
        cells = [str(c or "").replace("\n", " ").strip() for c in row]
        cleaned.append(cells)
    if not cleaned:
        return ""
    col_count = max(len(r) for r in cleaned)
    lines: List[str] = []
    for i, row in enumerate(cleaned):
        padded = row + [""] * (col_count - len(row))
        lines.append(" | ".join(padded))
        if i == 0:
            lines.append("-" * min(len(lines[0]), 80))
    return "\n".join(lines)


def _space_ratio(text: str) -> float:
    """Fraction of space characters in text. Near-zero means words were merged (garbled)."""
    if not text:
        return 0.0
    return text.count(" ") / len(text)


def _extract_tables_camelot(pdf_path: "Path | str") -> "dict[int, list]":
    """
    Extract all tables from a PDF using camelot lattice mode (one pass, all pages).
    Returns {page_number: [rows, ...]} where each rows is a list-of-lists of strings.
    Falls back to empty dict if camelot is unavailable or fails.
    """
    if not _CAMELOT_AVAILABLE:
        logger.warning("camelot-py not installed; table extraction disabled.")
        return {}
    try:
        tables = _camelot.read_pdf(str(pdf_path), flavor="lattice", pages="all")
    except Exception as exc:
        logger.warning("Camelot failed on %s: %s", Path(pdf_path).name, exc)
        return {}
    result: dict = {}
    for t in tables:
        rows = [list(row) for row in t.df.values.tolist()]
        result.setdefault(t.page, []).append(rows)
    return result


def _is_useful_table(rows: list) -> bool:
    """
    Return False for tables that pdfplumber extracted but are not genuinely useful:

    1. Fewer than 4 non-empty cells across the whole table — likely a false-positive
       region (e.g. pdfplumber treating the abstract block or a figure as a table).
    2. A single cell that contains more than 200 characters — characteristic of a
       mis-detected paragraph (the entire abstract crammed into one cell).
    3. Fewer than 2 rows or fewer than 2 columns — not really a table.
    """
    if not rows or len(rows) < 2:
        return False
    col_count = max(len(r) for r in rows)
    if col_count < 2:
        return False
    non_empty = [
        str(c or "").strip()
        for row in rows
        for c in row
        if str(c or "").strip()
    ]
    if len(non_empty) < 4:
        return False
    # Paragraph mis-detection: one huge cell with prose text
    if any(len(cell) > 200 for cell in non_empty):
        return False
    return True


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> List[str]:
    """Lightweight regex sentence splitter. Merges sub-3-word fragments."""
    parts = _SENT_SPLIT_RE.split(text)
    sentences: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences and len(part.split()) < 3:
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def _make_chunks(
    sentences: List[str],
    page_nums: List[int],
    section: str,
    source_file: str,
    chunk_index_start: int,
    target_tokens: int = 380,
    max_tokens: int = 512,
    overlap_tokens: int = 60,
) -> List[Chunk]:
    """
    Pack sentences into token-bounded chunks.

    Short-section optimisation: if the entire buffer fits in one chunk
    (total <= max_tokens with prefix), emit it as-is without applying the
    overlap rewind. This prevents tiny trailing overlap chunks that would
    otherwise degrade short sections into noise.

    For longer sections the standard sliding-window algorithm applies:
    advance by (N - overlap_sents) after each chunk, always >= 1.
    """
    assert len(sentences) == len(page_nums)

    prefix = f"[Section: {section}] "
    prefix_toks = count_tokens(prefix)

    def _make_one(sents: List[str], pages: List[int], idx: int) -> Chunk:
        body = " ".join(sents)
        full = prefix + body
        return Chunk(
            chunk_id=hashlib.md5((source_file + full).encode("utf-8")).hexdigest(),
            text=full,
            source_file=source_file,
            page_start=min(pages),
            page_end=max(pages),
            section=section,
            chunk_type="text",
            chunk_index=idx,
            token_count=count_tokens(full),
        )

    # Fast path: entire buffer fits in one chunk — emit without overlap.
    full_body = " ".join(sentences)
    if prefix_toks + count_tokens(full_body) <= max_tokens:
        return [_make_one(sentences, page_nums, chunk_index_start)]

    # Sliding-window path for long sections.
    chunks: List[Chunk] = []
    idx = chunk_index_start
    i = 0

    while i < len(sentences):
        sents: List[str] = []
        pages: List[int] = []
        tok = 0
        j = i

        while j < len(sentences):
            st = count_tokens(sentences[j])
            if sents and prefix_toks + tok + st > max_tokens:
                break
            sents.append(sentences[j])
            pages.append(page_nums[j])
            tok += st
            j += 1
            if prefix_toks + tok >= target_tokens:
                break

        if not sents:
            sents = [sentences[i]]
            pages = [page_nums[i]]
            j = i + 1

        chunks.append(_make_one(sents, pages, idx))
        idx += 1

        # Compute overlap tail.
        overlap_sents = 0
        overlap_tok = 0
        for sent in reversed(sents):
            t = count_tokens(sent)
            if overlap_tok + t > overlap_tokens:
                break
            overlap_tok += t
            overlap_sents += 1

        advance = max(1, len(sents) - overlap_sents)
        next_i = i + advance

        # Tail-emission optimisation: if all remaining sentences after next_i
        # fit within max_tokens, emit them as one final chunk and stop.
        # This prevents the shrinking-spiral of tiny overlapping tail chunks
        # that the sliding window produces at the end of a long section
        # (e.g. 47 → 44 → 43 → 34 → 17 → 10 → 9 tokens).
        if next_i < len(sentences):
            tail_sents = sentences[next_i:]
            tail_body = " ".join(tail_sents)
            if prefix_toks + count_tokens(tail_body) <= max_tokens:
                chunks.append(_make_one(tail_sents, page_nums[next_i:], idx))
                break

        i = next_i

    return chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class ChunkingConfig:
    target_tokens: int = 380
    max_tokens: int = 512
    overlap_tokens: int = 60
    garble_threshold: float = 0.02  # drop figure chunks with space_ratio below this


def process_pdf(
    pdf_path: "Path | str",
    config: "ChunkingConfig | None" = None,
) -> Iterator[Chunk]:
    """Process a single PDF and yield Chunk objects in reading order."""
    if config is None:
        config = ChunkingConfig()

    pdf_path = Path(pdf_path)
    source_file = pdf_path.name

    current_section = "Introduction"
    in_boilerplate = False
    sentence_buf: List[str] = []
    page_buf: List[int] = []
    global_idx = 0

    def flush() -> Iterator[Chunk]:
        nonlocal global_idx
        if not sentence_buf:
            return
        new_chunks = _make_chunks(
            sentences=list(sentence_buf),
            page_nums=list(page_buf),
            section=current_section,
            source_file=source_file,
            chunk_index_start=global_idx,
            target_tokens=config.target_tokens,
            max_tokens=config.max_tokens,
            overlap_tokens=config.overlap_tokens,
        )
        global_idx += len(new_chunks)
        yield from new_chunks

    def _make_caption_chunk(text: str, ctype: str, page_num: int) -> Chunk:
        nonlocal global_idx
        tag = "Figure" if ctype == "figure" else "Table"
        note = " [Visual content not included; caption describes figure context.]" if ctype == "figure" else ""
        full = f"[{tag}: {text}] {text}{note}"
        c = Chunk(
            chunk_id=hashlib.md5((source_file + full).encode("utf-8")).hexdigest(),
            text=full,
            source_file=source_file,
            page_start=page_num,
            page_end=page_num,
            section=current_section,
            chunk_type=ctype,
            chunk_index=global_idx,
            token_count=count_tokens(full),
        )
        global_idx += 1
        return c

    # Extract tables once per PDF via camelot (lattice mode for graphical borders).
    # Result: {page_number -> [rows, ...]}  where rows is list-of-lists of strings.
    camelot_tables = _extract_tables_camelot(pdf_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pdfplumber.open(str(pdf_path)) as pdf:
            body_font = _estimate_body_font_size(pdf)
            run_hdrs = _collect_running_headers(pdf)
            logger.debug("%s: body_font=%.1f  running_headers=%d",
                         source_file, body_font, len(run_hdrs))

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    body_lines, captions = _extract_page_text_blocks(
                        page, body_font, run_hdrs
                    )
                except Exception as exc:
                    logger.warning("Page %d error in %s: %s", page_num, source_file, exc)
                    continue

                if not in_boilerplate:
                    # Emit figure caption chunks, applying garble filter.
                    for cap_text, cap_type in captions:
                        if cap_type == "figure":
                            ratio = _space_ratio(cap_text)
                            if ratio < config.garble_threshold:
                                # Build a temporary chunk just to get its ID for the log.
                                _preview = f"[Figure: {cap_text}]"
                                _cid = hashlib.md5(
                                    (source_file + _preview).encode("utf-8")
                                ).hexdigest()
                                logger.warning(
                                    "Dropped garbled figure chunk %s "
                                    "(space_ratio=%.3f): %.60s",
                                    _cid, ratio, cap_text,
                                )
                                continue
                            yield _make_caption_chunk(cap_text, "figure", page_num)

                    # Emit table chunks from camelot results for this page.
                    table_caps = [ct for ct, ctype in captions if ctype == "table"]
                    for t_idx, rows in enumerate(camelot_tables.get(page_num, [])):
                        if not _is_useful_table(rows):
                            logger.debug(
                                "Skipping low-quality camelot table on page %d of %s",
                                page_num, source_file,
                            )
                            continue
                        table_text = _format_table(rows)
                        if not table_text.strip():
                            continue
                        cap = table_caps[t_idx] if t_idx < len(table_caps) else f"Table {t_idx + 1}"
                        full = f"[Table: {cap}] {table_text}"
                        yield Chunk(
                            chunk_id=hashlib.md5((source_file + full).encode("utf-8")).hexdigest(),
                            text=full,
                            source_file=source_file,
                            page_start=page_num,
                            page_end=page_num,
                            section=current_section,
                            chunk_type="table",
                            chunk_index=global_idx,
                            token_count=count_tokens(full),
                        )
                        global_idx += 1

                # Process body lines.
                for line_text, is_heading in body_lines:
                    if in_boilerplate:
                        continue

                    if _BOILERPLATE_SECTIONS.match(line_text):
                        in_boilerplate = True
                        yield from flush()
                        sentence_buf.clear()
                        page_buf.clear()
                        continue

                    if is_heading:
                        yield from flush()
                        sentence_buf.clear()
                        page_buf.clear()
                        current_section = line_text.strip()
                        continue

                    for sent in _split_sentences(line_text):
                        if sent:
                            sentence_buf.append(sent)
                            page_buf.append(page_num)

    yield from flush()


def process_directory(
    input_dir: "Path | str",
    config: "ChunkingConfig | None" = None,
    glob_pattern: str = "*.pdf",
) -> Iterator[Chunk]:
    """Process all PDFs in a directory and yield Chunk objects."""
    input_dir = Path(input_dir)
    pdf_files = sorted(input_dir.glob(glob_pattern))
    pdf_files = [p for p in pdf_files if "assignment" not in p.name.lower()]
    logger.info("Found %d PDF files in %s", len(pdf_files), input_dir)
    for pdf_path in pdf_files:
        logger.info("Processing: %s", pdf_path.name)
        yield from process_pdf(pdf_path, config)


def write_jsonl(chunks: Iterator[Chunk], output_path: "Path | str") -> int:
    """Write chunks to JSONL. Returns count of chunks written."""
    output_path = Path(output_path)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Quality analysis
# ---------------------------------------------------------------------------
def analyze_output(jsonl_path: "Path | str") -> dict:
    """
    Compute token distribution stats and quality flags for a JSONL output.

    Flags:
      SHORT_CHUNKS    -- < 50 tokens (parsing noise or very short sections)
      OVERSIZE_CHUNKS -- > 512 tokens (hard-max violation)
      DUPLICATE_IDS   -- same chunk_id emitted more than once
      SECTION_IMBALANCE -- > 70% of chunks in a single section
    """
    import statistics

    jsonl_path = Path(jsonl_path)
    chunks: List[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        return {"error": "No chunks found"}

    tc = [c["token_count"] for c in chunks]
    files = sorted({c["source_file"] for c in chunks})
    sections = sorted({c["section"] for c in chunks})

    per_file: dict = {}
    for fname in files:
        fc = [c["token_count"] for c in chunks if c["source_file"] == fname]
        per_file[fname] = {
            "chunk_count": len(fc),
            "mean_tokens": round(statistics.mean(fc), 1) if fc else 0,
            "median_tokens": round(statistics.median(fc), 1) if fc else 0,
            "min_tokens": min(fc) if fc else 0,
            "max_tokens": max(fc) if fc else 0,
        }

    flags = []
    short = [c for c in chunks if c["token_count"] < 50]
    if short:
        flags.append(f"SHORT_CHUNKS: {len(short)} chunks < 50 tokens")
    oversize = [c for c in chunks if c["token_count"] > 512]
    if oversize:
        flags.append(f"OVERSIZE_CHUNKS: {len(oversize)} chunks > 512 tokens")
    ids = [c["chunk_id"] for c in chunks]
    dup_count = len(ids) - len(set(ids))
    if dup_count:
        flags.append(f"DUPLICATE_IDS: {dup_count} duplicate chunk IDs")
    sec_counts: Counter = Counter(c["section"] for c in chunks)
    dom_sec, dom_cnt = sec_counts.most_common(1)[0]
    if dom_cnt / len(chunks) > 0.70:
        flags.append(
            f"SECTION_IMBALANCE: '{dom_sec}' holds {dom_cnt}/{len(chunks)} chunks"
        )

    stc = sorted(tc)
    n = len(stc)
    return {
        "total_chunks": len(chunks),
        "total_files": len(files),
        "files": files,
        "token_stats": {
            "mean": round(statistics.mean(tc), 1),
            "median": round(statistics.median(tc), 1),
            "stdev": round(statistics.stdev(tc), 1) if n > 1 else 0,
            "min": stc[0],
            "max": stc[-1],
            "p10": stc[max(0, int(n * 0.10))],
            "p25": stc[max(0, int(n * 0.25))],
            "p75": stc[min(n - 1, int(n * 0.75))],
            "p90": stc[min(n - 1, int(n * 0.90))],
        },
        "sections_detected": len(sections),
        "section_names": sections[:30],
        "per_file": per_file,
        "quality_flags": flags if flags else ["OK - no quality issues detected"],
    }
