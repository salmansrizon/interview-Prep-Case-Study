"""
Page-aware PDF text extraction: turn a PDF's raw bytes into a list of
``(page_number, text)`` pairs, using pypdf's text-layer extraction only.

HIGHLIGHTS: এই module ZERO dependency রাখে chromadb/sentence-transformers/
ollama-এর ওপর — শুধু "raw PDF bytes in, page-labeled text out"। এই সংকীর্ণ
দায়িত্বই এটাকে (tests/test_pipeline.py-তে) বিচ্ছিন্নভাবে ইউনিট-টেস্ট করা
সম্ভব করে, network বা model touch না করেই। Week 10-এর
src/extraction/text_extractor.py-এর "single dispatcher, no OCR fallback"
স্টাইলের রেফারেন্স নিয়ে লেখা হয়েছে, কিন্তু এখানে fresh, self-contained
কোড — week 10-এর ফোল্ডার থেকে import করা হয়নি (self-contained-artifacts
নিয়ম)।

কেন PAGE-AWARE (শুধু ``extract_text()`` না)? RAG-এর metadata contract
(README, config.py) অনুযায়ী প্রতিটা indexed chunk-এর সাথে ``{source, page}``
সেভ করতে হবে — সেটা citation আর document-filter দুটোরই ভিত্তি। তাই এই module
পুরো ডকুমেন্টের টেক্সট একটা bare string হিসেবে না দিয়ে PER-PAGE টুকরায় ভাগ
করে ফেরত দেয়, যাতে পরের ধাপ (chunker) জানে কোন chunk কোন পাতা থেকে এসেছে।
"""

from __future__ import annotations

import io

from pypdf import PdfReader

from src.utils.logger import get_logger

logger = get_logger("extraction")


class ExtractionError(Exception):
    """Raised when no extractable text can be found in a PDF.

    HIGHLIGHTS: week 10-এর ExtractionError-এর মতোই একই যুক্তি — raw pypdf
    exception bubble up করতে দেওয়া বা silently "" ফেরত দেওয়ার বদলে একটা
    dedicated exception type থাকলে app.py "এই PDF-এ সত্যিই কোনো text layer
    নেই (স্ক্যান করা ইমেজ)" বনাম "কিছু generically ভেঙে গেছে" — এই দুটো আলাদা
    করতে পারে, আর ইউজারকে একটা স্পষ্ট warning দেখাতে পারে, crash বা silent
    ভুল উত্তরের বদলে।
    """


def extract_pages(data: bytes, filename: str = "document.pdf") -> list[tuple[int, str]]:
    """Extract text from a PDF's raw bytes, one entry per page.

    Args:
        data: Raw PDF file bytes.
        filename: Original filename, used only for log messages / error text.

    Returns:
        A list of ``(page_number, text)`` tuples, 1-indexed, for every page
        that has a non-empty text layer. Pages with no text layer (e.g. a
        scanned image page) are simply skipped — see the "no OCR fallback"
        HIGHLIGHTS below.

    Raises:
        ExtractionError: if the PDF can't be opened, or NO page in the
            whole document has any extractable text.

    HIGHLIGHTS: কেন NO OCR FALLBACK? (week 10-এর extract_text_from_pdf-এর
    একই দুটো কারণের প্রতিফলন, এই project-এর জন্য পুনরায় লেখা)
      ১. Scope discipline — OCR একটা ভিন্ন, অনেক ভারী dependency (Tesseract
         binary, image preprocessing) — এই প্রজেক্টের "lightweight, fully
         local" নকশার লক্ষ্যকে ব্যাহত করবে।
      ২. Honesty over guessing — একটা low-quality scan silently OCR করে
         garbled text ইনডেক্স করা মানে ভুল/অর্থহীন চাঙ্ক সিমেন্টিক সার্চে
         মিশে যাওয়া, কোনো দৃশ্যমান সতর্কতা ছাড়াই। ExtractionError raise করে
         UI-কে স্পষ্টভাবে "এই PDF-এ কোনো text layer পাওয়া যায়নি" বলতে দেওয়া
         ভালো, model-কে না-বোঝা garbage খাওয়ানোর চেয়ে।
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf can raise several different error types
        raise ExtractionError(f"Could not open {filename!r} as a PDF: {exc}") from exc

    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        # extract_text() returns "" (not None) when a page has no text
        # layer at all. We keep whatever pages DO have text and only raise
        # if the ENTIRE document came back empty — a partially-scanned PDF
        # still contributes its readable pages.
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append((index, page_text))

    if not pages:
        raise ExtractionError(
            f"No extractable text found in {filename!r}. It may be a "
            f"scanned image with no text layer — OCR is intentionally out "
            f"of scope for this project (text-layer extraction only)."
        )

    logger.info(
        "Extracted text from %d/%d page(s) of %r",
        len(pages),
        len(reader.pages),
        filename,
    )
    return pages
