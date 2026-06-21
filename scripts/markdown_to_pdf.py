from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]


def styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    base = "STSong-Light"
    mono = "Courier"
    return {
        "title": ParagraphStyle("TitleCN", fontName=base, fontSize=16, leading=20, spaceAfter=6),
        "h1": ParagraphStyle("H1CN", fontName=base, fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=3),
        "h2": ParagraphStyle("H2CN", fontName=base, fontSize=9, leading=11, spaceBefore=4, spaceAfter=2),
        "body": ParagraphStyle("BodyCN", fontName=base, fontSize=7.0, leading=8.8, spaceAfter=2),
        "small": ParagraphStyle("SmallCN", fontName=base, fontSize=6.2, leading=7.8, spaceAfter=2),
        "code": ParagraphStyle("Code", fontName=mono, fontSize=5.8, leading=7, leftIndent=6, rightIndent=6, spaceAfter=3),
    }


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def image_flowable(path: Path) -> Image:
    max_w = 6.8 * inch
    max_h = 2.7 * inch
    from PIL import Image as PILImage

    with PILImage.open(path) as img:
        w, h = img.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def parse_table(lines: list[str], index: int) -> tuple[Table, int]:
    table_lines = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        table_lines.append(lines[index].strip())
        index += 1
    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table, index


def markdown_to_story(markdown: str) -> list:
    st = styles()
    story = []
    lines = markdown.splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []
    bullet_lines: list[str] = []

    def flush_bullets() -> None:
        nonlocal bullet_lines
        if bullet_lines:
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(item), st["body"])) for item in bullet_lines],
                    bulletType="bullet",
                    leftIndent=14,
                )
            )
            bullet_lines = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), st["code"]))
                code_lines = []
                in_code = False
            else:
                flush_bullets()
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if not stripped:
            flush_bullets()
            story.append(Spacer(1, 0.018 * inch))
            index += 1
            continue

        if stripped.startswith("|"):
            flush_bullets()
            table, index = parse_table(lines, index)
            story.append(table)
            story.append(Spacer(1, 0.035 * inch))
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            flush_bullets()
            image_path = ROOT / image_match.group(2)
            story.append(image_flowable(image_path))
            story.append(Spacer(1, 0.04 * inch))
            index += 1
            continue

        if stripped.startswith("- "):
            bullet_lines.append(stripped[2:])
            index += 1
            continue

        if re.match(r"\d+\. ", stripped):
            bullet_lines.append(re.sub(r"^\d+\. ", "", stripped))
            index += 1
            continue

        flush_bullets()
        if stripped.startswith("# "):
            story.append(Paragraph(inline(stripped[2:]), st["title"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(inline(stripped[3:]), st["h1"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(inline(stripped[4:]), st["h2"]))
        else:
            story.append(Paragraph(inline(stripped), st["body"]))
        index += 1

    flush_bullets()
    return story


def main() -> None:
    markdown = (ROOT / "report.md").read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(ROOT / "report.pdf"),
        pagesize=A4,
        rightMargin=0.48 * inch,
        leftMargin=0.48 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
    )
    doc.build(markdown_to_story(markdown))


if __name__ == "__main__":
    main()
