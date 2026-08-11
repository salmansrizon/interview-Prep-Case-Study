"""
Recursive/paragraph-aware chunking with overlap, preserving per-chunk
page-number metadata (Class 1 Lecture, section 3.4).

HIGHLIGHTS: এখানে NAIVE FIXED-SIZE chunking (প্রতি N ক্যারেক্টারে কাটা,
বাক্য/প্যারাগ্রাফ কোথায় শেষ হচ্ছে সেটা না দেখেই) ইচ্ছাকৃতভাবে এড়ানো হয়েছে।
এর বদলে এই module প্রথমে প্যারাগ্রাফ ব্রেক (blank line) দিয়ে ভাগ করার চেষ্টা
করে, তারপর প্রয়োজনে বাক্য দিয়ে, তারপর প্রয়োজনে raw ক্যারেক্টার দিয়ে — যতক্ষণ
না প্রতিটা টুকরা target chunk_size-এর নিচে আসে। এভাবে মাঝ-বাক্যে কাটা এড়ানো
যায়, যা লেকচারের section 3.4-এর মূল পয়েন্ট: মাঝ-বাক্যে কাটলে অর্থ ভেঙে
যায় আর embedding সেই ভাঙা অর্থ ধরতে পারে না।
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import get_config
from src.utils.logger import get_logger

logger = get_logger("chunking")

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # includes Bengali দাঁড়ি (।) as a sentence end


@dataclass(frozen=True)
class Chunk:
    """One indexable unit of text, with the metadata (source, page) that
    powers BOTH citations and the document-picker filter (README, config.py).

    HIGHLIGHTS: ``text``, ``source``, ``page`` — exactly the three fields
    src/vectorstore/store.py needs: ``text`` becomes the ChromaDB
    "document", and ``{"source": source, "page": page}`` becomes its
    metadata dict. Keeping this dataclass in the chunking module (rather
    than a bare dict) gives every downstream consumer (vectorstore,
    pipeline, tests) a typed, self-documenting shape instead of guessing
    dict keys.
    """

    text: str
    source: str
    page: int


def _split_large_unit(text: str, max_size: int) -> list[str]:
    """Split a single paragraph (or sentence) that's still too big.

    Tries sentence boundaries first; if even a single sentence exceeds
    ``max_size``, falls back to a hard character split as a last resort.

    HIGHLIGHTS: এটা "recursive" নামের মূল কারণ — একটা বড় প্যারাগ্রাফের ওপর
    সরাসরি hard-cut না করে, আগে সূক্ষ্মতর বিভাজক (বাক্য) দিয়ে আবার ভাগ করার
    চেষ্টা করা হয়, hard character split শুধু সত্যিই কোনো উপায় না থাকলে
    (একটা একক বাক্যই chunk_size-এর চেয়ে বড়) ব্যবহৃত হয়।
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        # No sentence boundary found (or already one sentence) — last
        # resort: hard character split, never silently drop text.
        return [text[i : i + max_size] for i in range(0, len(text), max_size)]

    units: list[str] = []
    for sentence in sentences:
        if len(sentence) > max_size:
            units.extend(
                [sentence[i : i + max_size] for i in range(0, len(sentence), max_size)]
            )
        else:
            units.append(sentence)
    return units


def _merge_units_with_overlap(
    atoms: list[tuple[str, int]],
    chunk_size: int,
    overlap_chars: int,
    min_chunk_size: int,
) -> list[tuple[str, int]]:
    """Greedily merge (text, page) atoms into (chunk_text, page) pairs,
    carrying a trailing-character overlap forward between consecutive chunks.

    HIGHLIGHTS: প্রতিটা chunk-এর page attribution হয় তার প্রথম "নতুন"
    (non-overlap) atom-এর page দিয়ে — যেমন lecture-প্রম্প্ট চায়: "একটা chunk
    পেজ বর্ডারের কাছে হলে সেন্সিবলি একটা পেজে attribute হওয়া উচিত"। Overlap
    text (আগের chunk-এর শেষ অংশ, যেটা কনটেক্সট বাঁচানোর জন্য পুনরায় জোড়া
    হয়েছে) কখনো নতুন chunk-এর page ঠিক করে না — কারণ সেটা যুক্তিগতভাবে এখনও
    আগের chunk-এরই কনটেন্ট, শুধু continuity-র জন্য duplicate করা হয়েছে।
    """
    chunks: list[tuple[str, int]] = []
    current_texts: list[str] = []
    current_len = 0
    current_primary_page: int | None = None
    overlap_prefix = ""

    def flush() -> None:
        nonlocal current_texts, current_len, current_primary_page, overlap_prefix
        if not current_texts:
            return
        body = "\n\n".join(current_texts).strip()
        if body and len(body) >= min_chunk_size:
            page = current_primary_page if current_primary_page is not None else 1
            chunks.append((body, page))
            # Seed the next chunk's overlap with the tail of this chunk.
            overlap_prefix = body[-overlap_chars:] if overlap_chars > 0 else ""
        else:
            overlap_prefix = ""
        current_texts = []
        current_len = 0
        current_primary_page = None

    for text, page in atoms:
        if overlap_prefix and not current_texts:
            # Start the new chunk with the carried-over overlap text (does
            # NOT set current_primary_page — see HIGHLIGHTS above).
            current_texts.append(overlap_prefix)
            current_len += len(overlap_prefix)
            overlap_prefix = ""

        addition_len = len(text) + (2 if current_texts else 0)  # +2 for the "\n\n" join
        if current_texts and current_len + addition_len > chunk_size:
            flush()
            if overlap_prefix:
                current_texts.append(overlap_prefix)
                current_len += len(overlap_prefix)
                overlap_prefix = ""

        current_texts.append(text)
        current_len += len(text) + (2 if len(current_texts) > 1 else 0)
        if current_primary_page is None:
            current_primary_page = page

    flush()
    return chunks


def chunk_page_texts(
    pages: list[tuple[int, str]],
    source: str,
) -> list[Chunk]:
    """Chunk a document's page-labeled text into overlap-preserving,
    page-tagged ``Chunk`` objects.

    Args:
        pages: ``(page_number, page_text)`` pairs, e.g. as returned by
            ``src.extraction.pdf_extractor.extract_pages``. 1-indexed.
        source: The document's filename — stored verbatim on every chunk's
            metadata (used for citations AND the document-picker filter).

    Returns:
        A list of ``Chunk`` objects in document order.

    HIGHLIGHTS: প্রতিটা page আলাদাভাবে প্যারাগ্রাফে ভাগ করে একটা ফ্ল্যাট
    ``(paragraph_text, page)`` atom list বানানো হয় — এভাবে page boundary
    তথ্যটা কখনো হারায় না, এমনকি একটা chunk একাধিক atom (এবং তাই সম্ভবত
    একাধিক page) থেকে টেক্সট মার্জ করলেও। fragile substring-search দিয়ে
    "এই chunk কোন offset-এ শুরু হয়েছে" খুঁজে বের করার বদলে, প্রতিটা atom তার
    উৎস page সরাসরি বহন করে — যা অনেক বেশি নির্ভরযোগ্য।
    """
    cfg = get_config().chunking

    atoms: list[tuple[str, int]] = []
    for page_number, page_text in pages:
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(page_text) if p.strip()]
        if not paragraphs:
            continue
        for para in paragraphs:
            if len(para) > cfg.chunk_size_chars:
                for unit in _split_large_unit(para, cfg.chunk_size_chars):
                    atoms.append((unit, page_number))
            else:
                atoms.append((para, page_number))

    if not atoms:
        logger.warning("No paragraphs found to chunk for source=%r", source)
        return []

    merged = _merge_units_with_overlap(
        atoms,
        chunk_size=cfg.chunk_size_chars,
        overlap_chars=cfg.chunk_overlap_chars,
        min_chunk_size=cfg.min_chunk_size_chars,
    )

    chunks = [Chunk(text=text, source=source, page=page) for text, page in merged]
    logger.info("Chunked %r into %d chunk(s)", source, len(chunks))
    return chunks
