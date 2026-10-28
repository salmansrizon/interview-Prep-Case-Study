"""
PDF investment memo with fpdf2 — numbers from code, prose from the agents.

HIGHLIGHTS:
* Metrics table আর Risk Score সরাসরি ``RiskResult`` dict থেকে বসে, LLM-এর লেখা থেকে না। তাই PDF-এর
  প্রতিটা সংখ্যা code-এর সংখ্যা; agent-এর লেখা শুধু ব্যাখ্যা, আর সেটাও publish-এর আগে verify হয়েছে।
* Price chart = fpdf2-র ``polyline`` দিয়ে আঁকা — শুধু chart-এর জন্য matplotlib লাগে না।
* Core font (Helvetica) শুধু latin-1 বোঝে; LLM-এর "—", "’", "≥" ইত্যাদি ``_latin1()`` বদলে দেয়,
  নাহলে fpdf2 exception ছোড়ে।
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fpdf import FPDF

SYNTHETIC_DISCLAIMER = "Educational project, synthetic data, not investment advice."
_SUBS = str.maketrans({"—": "-", "–": "-", "−": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "•": "-",
                       "…": "...", "≥": ">=", "≤": "<=", "→": "->"})
LABELS = {"volatility": "Annualized volatility", "max_drawdown": "Maximum drawdown",
          "var_95": "Historical VaR (95%, 1-day)", "sentiment": "Net news sentiment (-1..+1)"}


def _latin1(text: str) -> str:
    return text.translate(_SUBS).encode("latin-1", "replace").decode("latin-1")


def write_memo_pdf(path: Path, *, company: str, source: str, risk: dict, brief: str, memo: str,
                   prices: list[tuple[str, float]], mode: str, model: str = "llama3.1:8b") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, _latin1(f"Investment Risk Memo: {risk['ticker']} ({company})"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.cell(0, 5, _latin1(f"Data as of {risk['as_of']} | generated {datetime.now():%Y-%m-%d %H:%M} | agents: "
                           f"{mode} mode, {model} (local) | data source: {source}"), new_x="LMARGIN", new_y="NEXT")
    disclaimer = (SYNTHETIC_DISCLAIMER if source == "synthetic"
                  else f"Educational project, data from {source} (unverified), not investment advice.")
    pdf.ln(2)
    pdf.set_fill_color(253, 236, 234)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 7, _latin1(disclaimer), border=1, fill=True, align="C", new_x="LMARGIN")
    pdf.set_fill_color(255, 255, 255)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"Risk Score: {risk['score']:.1f} / 100 ({risk['band'].upper()})", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    with pdf.table(col_widths=(70, 35, 45, 30), text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT")) as table:
        table.row(["Metric", "Value", "Risk points (0-100)", "Weight"])
        for k, label in LABELS.items():
            v = risk["metrics"][k]
            table.row([label, f"{v:+.2f}" if k == "sentiment" else f"{v * 100:.1f}%",
                       f"{risk['components'][k]:.1f}", f"{risk['weights'][k]:.2f}"])

    closes = [c for _, c in prices]
    if len(closes) > 1:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, f"Daily close, {prices[0][0]} to {prices[-1][0]} (low {min(closes):.2f}, "
                       f"high {max(closes):.2f})", new_x="LMARGIN", new_y="NEXT")
        _chart(pdf, closes, x=pdf.l_margin, y=pdf.get_y() + 1, w=pdf.epw, h=45)
        pdf.set_y(pdf.get_y() + 50)

    _section(pdf, "News brief (News Scout Agent)", brief)
    _section(pdf, "Risk explanation (Risk Analyst Agent)", memo)
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "The Risk Score and every number in the table are computed by code (src/risk/metrics.py). "
                         "The text is written by local agents and checked by code before publishing: cited news ids "
                         "exist and quoted numbers match the computed values.", new_x="LMARGIN")
    pdf.output(str(path))
    return path


def _chart(pdf: FPDF, closes: list[float], x: float, y: float, w: float, h: float) -> None:
    lo, hi = min(closes), max(closes)
    span = (hi - lo) or 1.0
    pts = [(x + w * i / (len(closes) - 1), y + h - h * (c - lo) / span) for i, c in enumerate(closes)]
    pdf.set_draw_color(190, 190, 190)
    pdf.set_line_width(0.2)
    pdf.rect(x, y, w, h)
    pdf.set_draw_color(30, 90, 160)
    pdf.set_line_width(0.4)
    pdf.polyline(pts)
    pdf.set_draw_color(0, 0, 0)


def _section(pdf: FPDF, title: str, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for line in (text or "(empty)").splitlines():
        s = line.strip()
        if not s:
            pdf.ln(2)
        elif s.startswith("#"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, _latin1(s.lstrip("# ")), new_x="LMARGIN")
            pdf.set_font("Helvetica", size=10)
        else:
            pdf.multi_cell(0, 5, _latin1(s.replace("**", "")), new_x="LMARGIN")
