"""Text extraction package: pull raw text out of pasted strings, PDFs, and DOCX files."""

from src.extraction.text_extractor import (
    extract_text,
    extract_text_from_docx,
    extract_text_from_pdf,
    extract_text_from_plain,
)

__all__ = [
    "extract_text",
    "extract_text_from_plain",
    "extract_text_from_pdf",
    "extract_text_from_docx",
]
