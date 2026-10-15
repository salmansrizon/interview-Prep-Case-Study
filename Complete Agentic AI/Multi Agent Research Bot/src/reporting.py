import re
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def extract_sources(*texts: str) -> list[str]:
    sources = []
    for text in texts:
        for match in re.findall(r"https?://[^\s<>\])]+", text):
            url = match.rstrip(".,;:")
            if url not in sources:
                sources.append(url)
    return sources


def extract_overview(report: str, word_limit: int = 90) -> tuple[str, list[tuple[str, str]]]:
    intro_lines = []
    fallback_lines = []
    finding_headings = []
    findings_level = None
    current_finding = None
    in_intro = False

    def clean_markdown(text: str) -> str:
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1", text)
        text = re.sub(r"[*_`]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    for raw_line in report.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = clean_markdown(heading.group(2))
            current_finding = None
            if level <= 2:
                in_intro = bool(re.search(r"\b(?:introduction|executive summary)\b", title, re.I))
                if re.search(r"\bkey findings?\b", title, re.I):
                    findings_level = level
                elif findings_level is not None and level <= findings_level:
                    findings_level = None
            elif findings_level is not None and level == findings_level + 1:
                current_finding = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title)
                finding_headings.append([current_finding, []])
            continue

        if (
            not line
            or line.startswith("|")
            or re.fullmatch(r"[-*_]{3,}", line)
            or re.match(r"^\|?\s*:?-{3,}", line)
            or re.match(r"^\*{1,2}Why this is an advance\*{1,2}$", line, re.I)
        ):
            continue
        if current_finding is not None:
            finding_headings[-1][1].append(line)
        elif in_intro:
            intro_lines.append(line)
        elif not any(line.startswith(heading) for heading in ("#",)):
            fallback_lines.append(line)

    summary_text = clean_markdown(" ".join(intro_lines or fallback_lines))
    summary_words = summary_text.split()
    summary = " ".join(summary_words[:word_limit])
    if len(summary_words) > word_limit:
        summary += "…"

    takeaways = []
    for heading, body_lines in finding_headings:
        if re.search(r"\b(?:introduction|conclusion|sources|references?)\b", heading, re.I):
            continue
        body = clean_markdown(" ".join(body_lines))
        body_words = body.split()
        detail = " ".join(body_words[:32])
        if len(body_words) > 32:
            detail += "…"
        takeaways.append((heading, detail or "See the full report for supporting detail."))
        if len(takeaways) == 4:
            break

    return summary, takeaways


def _inline_markup(text: str) -> str:
    value = escape(text, quote=False)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<link href="\2" color="#214e9b">\1</link>',
        value,
    )
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    return value


def create_report_pdf(report: str, title: str) -> bytes:
    output = BytesIO()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111111"),
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="ReportHeading",
        parent=styles["Heading1"],
        textColor=colors.HexColor("#111111"),
        spaceBefore=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubheading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#214e9b"),
        spaceBefore=9,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportBullet",
        parent=styles["BodyText"],
        leftIndent=12,
        firstLineIndent=-8,
    ))
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 8
    page_width = letter[0] - 1.3 * inch
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.75 * inch,
        title=title,
    )
    story: list[Flowable] = [Paragraph(escape(title), styles["ReportTitle"])]
    paragraph_lines = []

    def flush_paragraph():
        if paragraph_lines:
            story.append(Paragraph(_inline_markup(" ".join(paragraph_lines)), styles["BodyText"]))
            paragraph_lines.clear()

    lines = report.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or re.match(r"^[-*_]{3,}$", line):
            flush_paragraph()
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush_paragraph()
            style = styles["ReportHeading"] if len(heading.group(1)) <= 2 else styles["ReportSubheading"]
            story.append(Paragraph(_inline_markup(heading.group(2)), style))
            index += 1
            continue

        if line.startswith("|"):
            flush_paragraph()
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append([
                        Paragraph(_inline_markup(cell), styles["BodyText"])
                        for cell in cells
                    ])
                index += 1
            if rows:
                table = Table(rows, colWidths=[page_width / len(rows[0])] * len(rows[0]), repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7dce2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.extend((table, Spacer(1, 8))) # type: ignore
            continue

        bullet = re.match(r"^(?:[-*]\s+|\d+[.)]\s+)(.+)$", line)
        if bullet:
            flush_paragraph()
            story.append(Paragraph(
                f'<font color="#214e9b">&#8226;</font> {_inline_markup(bullet.group(1))}',
                styles["ReportBullet"],
            ))
            index += 1
            continue

        paragraph_lines.append(line)
        index += 1

    flush_paragraph()

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#687386"))
        canvas.drawString(0.65 * inch, 0.42 * inch, "Research Studio")
        canvas.drawRightString(letter[0] - 0.65 * inch, 0.42 * inch, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return output.getvalue()
