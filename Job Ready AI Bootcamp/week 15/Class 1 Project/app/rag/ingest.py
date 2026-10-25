"""PDF -> page texts -> overlapping chunks (self-contained copy of the Week 12 logic).

HIGHLIGHTS: ADR 0002 — এই project অন্য week-এর folder import করে না। Week 12-এর
extraction + recursive chunking এখানে ছোট করে আবার লেখা। Chunk id = text-এর
SHA-256 — একই PDF দুবার upload করলে একই id, তাই vector store-এ duplicate হয় না
(idempotent ingestion)।
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)   # broken uploads are reported as BadDocument, not log noise

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?।])\s+")


class BadDocument(ValueError):
    """The upload is not a readable, text-based PDF within the limits."""


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int


def extract_pages(data: bytes, filename: str, max_pages: int = 300) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that have a text layer."""
    if not data.startswith(b"%PDF-"):
        raise BadDocument(f"{filename!r} is not a PDF (missing %PDF- header)")
    try:
        reader = PdfReader(io.BytesIO(data))
        n = len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types
        raise BadDocument(f"Could not open {filename!r} as a PDF: {exc}") from exc
    if n > max_pages:
        raise BadDocument(f"{filename!r} has {n} pages; the limit is {max_pages}")
    pages = [(i, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages, start=1)]
    pages = [(i, t) for i, t in pages if t]
    if not pages:
        raise BadDocument(f"No extractable text in {filename!r} (scanned image? OCR is out of scope)")
    return pages


def _units(text: str, size: int) -> list[str]:
    """Paragraphs first; split long ones by sentence; hard-cut only as a last resort."""
    out = []
    for para in (p.strip() for p in _PARA.split(text) if p.strip()):
        if len(para) <= size:
            out.append(para)
            continue
        for sent in (s.strip() for s in _SENT.split(para) if s.strip()):
            out.extend([sent[i:i + size] for i in range(0, len(sent), size)] if len(sent) > size else [sent])
    return out


def chunk_pages(pages: list[tuple[int, str]], source: str, size: int = 700, overlap: int = 100,
                min_size: int = 40) -> list[Chunk]:
    """Greedy merge of units into ~size-char chunks, carrying `overlap` chars forward within a page.

    A chunk never crosses a page boundary, so its `page` citation is exact (the overlap restarts per page)."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    page: int | None = None
    carry = ""

    def flush():
        nonlocal buf, page, carry
        body = "\n\n".join(([carry] if carry else []) + buf).strip()
        if len(body) >= min_size:
            cid = hashlib.sha256(f"{source}\x00{body}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, body, source, page or 1))
            carry = body[-overlap:] if overlap else ""
        buf, page = [], None

    for pno, text in pages:
        if buf:
            flush()
        carry = ""
        for unit in _units(text, size):
            if buf and len(carry) + sum(map(len, buf)) + len(unit) > size:
                flush()
            buf.append(unit)
            page = page or pno
    if buf:
        flush()
    return chunks
