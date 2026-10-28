"""Build samples/clinic_handbook.pdf from data/kb/*.md — the printable version of the assistant's knowledge base.

Run:  python samples/make_handbook.py

The Markdown pages are the source of truth (they are what gets indexed); this PDF is for people. Every
section heading carries its citation id, so a reader can look up "[kb:fever:4]" from an answer.
"""
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.rag.kb import DISCLAIMER, load_kb  # noqa: E402


def build(path: Path, kb_dir: Path = ROOT / "data" / "kb") -> int:
    """Write the PDF and return its page count."""
    sections = load_kb(kb_dir)
    pdf = FPDF()

    def line(h, text):          # multi_cell, then back to the left margin on the next line
        pdf.multi_cell(0, h, text, new_x="LMARGIN", new_y="NEXT")

    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    line(10, "Greenfield Community Clinic\nPatient Handbook")
    pdf.set_font("Helvetica", "I", 11)
    line(6, f"\n{DISCLAIMER}\nThe clinic, its staff and its policies are fictional. "
            "In an emergency in Bangladesh, call 999.\n")
    pdf.set_font("Helvetica", size=11)
    pages = list(dict.fromkeys(s.page for s in sections))
    line(6, "Contents:\n" + "\n".join(f"  {i}. {p}" for i, p in enumerate(pages, 1)))
    current = None
    for s in sections:
        if s.page != current:
            current = s.page
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            line(9, s.page)
            pdf.set_font("Helvetica", "I", 9)
            line(5, DISCLAIMER)
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        line(7, f"{s.title}  [{s.id}]")
        pdf.set_font("Helvetica", size=11)
        line(6, s.text)
    pdf.output(str(path))
    return pdf.pages_count


if __name__ == "__main__":
    out = Path(__file__).with_name("clinic_handbook.pdf")
    print(out, build(out), "pages")
