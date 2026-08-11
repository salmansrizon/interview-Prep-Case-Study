"""
Text extraction: turn a plain-text paste, a PDF's bytes, or a DOCX's bytes
into a single plain-text string that the embedding service can consume.

HIGHLIGHTS: this module has ZERO dependency on sentence-transformers, numpy,
or streamlit — it only knows how to go "raw bytes/string in, plain text
out". That narrow responsibility (a "deep module" with a small interface:
just `extract_text`) is what lets it be unit-tested in isolation (see
tests/test_pipeline.py) without ever touching the network or the embedding
model, and it's why app.py can call this file directly instead of
duplicating PDF/DOCX parsing logic inline in the Streamlit UI code.
"""

from __future__ import annotations

import io

from pypdf import PdfReader
from docx import Document

from src.utils.logger import get_logger

logger = get_logger("extraction")


class ExtractionError(Exception):
    """Raised when text cannot be extracted from a given source.

    HIGHLIGHTS: a dedicated exception type (rather than letting raw
    pypdf/python-docx exceptions bubble up, or silently returning "") lets
    app.py distinguish "this file genuinely has no extractable text" (a
    scanned/image-only PDF, an empty upload) from "this call is generally
    broken", and show the user a clear, specific warning instead of a
    crash or a silent wrong answer.
    """


def extract_text_from_plain(text: str) -> str:
    """Normalize a pasted plain-text string.

    HIGHLIGHTS: this looks trivial (just `.strip()`), but keeping it as a
    real function — rather than inlining `.strip()` at every call site —
    means all three input paths (plain/PDF/DOCX) go through the SAME
    "clean up whitespace" step before reaching the embedding model, and
    the dispatcher below (`extract_text`) can treat all three uniformly.
    """
    return text.strip()


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF's raw bytes using pypdf's text layer.

    HIGHLIGHTS: this deliberately does NOT fall back to OCR (e.g.
    pytesseract) when a PDF has no extractable text layer (a scanned
    image saved as PDF). Two reasons, both worth stating explicitly for a
    student reading this:
      1. Scope discipline — OCR is a materially different, much heavier
         dependency (a Tesseract binary, image preprocessing, usually much
         slower and noisier output). Bundling it "just in case" would
         betray the project's "lightweight, CPU-fast, fully local" design
         goal from the lecture's section 4.2.
      2. Honesty over guessing — silently OCR'ing a low-quality scan can
         produce garbled text that then gets embedded and silently
         corrupts the similarity ranking with no visible sign anything
         went wrong. It's better to raise ExtractionError and let the UI
         tell the user plainly "no text found in this PDF" than to feed
         the embedding model garbage it can't distinguish from real text.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf can raise several different error types
        raise ExtractionError(f"Could not open PDF: {exc}") from exc

    pages_text = []
    for page in reader.pages:
        # extract_text() returns "" (not None) when a page has no text
        # layer at all — e.g. a scanned image page. We collect whatever
        # text layers exist and only raise if NOTHING was found across
        # the whole document (a partially-scanned PDF still contributes
        # its readable pages).
        page_text = page.extract_text() or ""
        if page_text:
            pages_text.append(page_text)

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        # No OCR fallback (see docstring) — we fail loudly and specifically
        # so the caller (app.py) can show "no text layer found, this PDF
        # may be a scanned image" instead of silently ranking an empty CV.
        raise ExtractionError(
            "No extractable text found in this PDF. It may be a scanned "
            "image with no text layer — OCR is intentionally out of scope "
            "for this project (text-layer extraction only)."
        )

    return full_text


def extract_text_from_docx(data: bytes) -> str:
    """Extract text from a DOCX file's raw bytes using python-docx.

    HIGHLIGHTS: DOCX is a zipped XML format, so unlike a PDF there's no
    "text layer vs. scanned image" ambiguity — if a .docx file was created
    normally (not e.g. a scanned image pasted into a Word doc as a
    picture), its text is always programmatically readable. We still
    raise ExtractionError on an empty result for consistency with the PDF
    path: the dispatcher and the UI shouldn't need to know which format
    produced an empty CV, only that one did.
    """
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # python-docx raises various errors for bad files
        raise ExtractionError(f"Could not open DOCX: {exc}") from exc

    # Paragraph text covers the common case. Tables are common in CVs too
    # (e.g. a skills grid), so we also walk table cells — otherwise a CV
    # laid out as a table would silently lose most of its content.
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)

    full_text = "\n".join(parts).strip()

    if not full_text:
        raise ExtractionError("No extractable text found in this DOCX file.")

    return full_text


def extract_text(source: str | bytes, filetype: str) -> str:
    """Unified dispatcher: extract plain text from any supported source.

    Args:
        source: Either a plain-text string (filetype="txt"/"plain") or raw
            file bytes (filetype="pdf"/"docx").
        filetype: One of "plain", "txt", "pdf", "docx" (case-insensitive).

    Returns:
        Extracted plain text, stripped of leading/trailing whitespace.

    Raises:
        ExtractionError: if the filetype is unsupported or no text could
            be extracted.

    HIGHLIGHTS: this single entry point is the ONLY thing app.py calls —
    it never calls PdfReader/Document directly. That indirection means if
    we ever add a new format (e.g. .rtf), app.py doesn't change at all;
    only this dispatcher and one new `extract_text_from_*` function do.
    It also means the "how do I know which extractor to call" branching
    logic lives in exactly one place instead of being duplicated wherever
    a file gets uploaded (job description AND multiple CVs, in this app).
    """
    ft = filetype.lower().lstrip(".")

    if ft in ("plain", "txt"):
        if not isinstance(source, str):
            raise ExtractionError("Plain-text source must be a str.")
        text = extract_text_from_plain(source)
    elif ft == "pdf":
        if not isinstance(source, (bytes, bytearray)):
            raise ExtractionError("PDF source must be raw bytes.")
        text = extract_text_from_pdf(bytes(source))
    elif ft == "docx":
        if not isinstance(source, (bytes, bytearray)):
            raise ExtractionError("DOCX source must be raw bytes.")
        text = extract_text_from_docx(bytes(source))
    else:
        raise ExtractionError(f"Unsupported filetype: {filetype!r}")

    if not text:
        raise ExtractionError("Extracted text is empty after cleanup.")

    logger.info("Extracted %d chars from filetype=%s", len(text), ft)
    return text
