from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def read_loss_csv(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["step"]), float(row["loss"])))
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def draw_line_chart(
    data: list[tuple[int, float]],
    title: str,
    y_label: str,
    output_path: Path,
    width: int = 1200,
    height: int = 720,
) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, right, top, bottom = 100, width - 60, 80, height - 90

    steps = [step for step, _ in data]
    losses = [loss for _, loss in data]
    smooth = moving_average(losses, window=max(5, len(losses) // 25))
    min_x, max_x = min(steps), max(steps)
    min_y, max_y = min(losses), max(losses)
    pad_y = max((max_y - min_y) * 0.12, 0.05)
    min_y -= pad_y
    max_y += pad_y

    draw.text((left, 32), title, fill=(20, 32, 46), font=font)
    draw.line((left, bottom, right, bottom), fill=(60, 70, 80), width=2)
    draw.line((left, top, left, bottom), fill=(60, 70, 80), width=2)

    for i in range(6):
        ratio = i / 5
        y = bottom - ratio * (bottom - top)
        value = min_y + ratio * (max_y - min_y)
        draw.line((left, y, right, y), fill=(225, 230, 235), width=1)
        draw.text((18, y - 7), f"{value:.2f}", fill=(80, 90, 100), font=font)

    for i in range(6):
        ratio = i / 5
        x = left + ratio * (right - left)
        value = round(min_x + ratio * (max_x - min_x))
        draw.line((x, bottom, x, bottom + 6), fill=(60, 70, 80), width=1)
        draw.text((x - 12, bottom + 16), str(value), fill=(80, 90, 100), font=font)

    def point(step: int, loss: float) -> tuple[float, float]:
        x_ratio = 0 if max_x == min_x else (step - min_x) / (max_x - min_x)
        y_ratio = 0 if max_y == min_y else (loss - min_y) / (max_y - min_y)
        return left + x_ratio * (right - left), bottom - y_ratio * (bottom - top)

    raw_points = [point(step, loss) for step, loss in data]
    smooth_points = [point(step, loss) for (step, _), loss in zip(data, smooth)]
    if len(raw_points) >= 2:
        draw.line(raw_points, fill=(150, 170, 190), width=2)
        draw.line(smooth_points, fill=(22, 107, 198), width=4)
    for x, y in raw_points[:: max(1, len(raw_points) // 45)]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(22, 107, 198))

    draw.text((width // 2 - 30, height - 45), "step", fill=(60, 70, 80), font=font)
    draw.text((16, 44), y_label, fill=(60, 70, 80), font=font)
    draw.text((right - 240, top + 10), "light: raw loss", fill=(120, 135, 150), font=font)
    draw.text((right - 240, top + 30), "blue: moving average", fill=(22, 107, 198), font=font)
    image.save(output_path)


def draw_pipeline_diagram(output_path: Path) -> None:
    image = Image.new("RGB", (1400, 520), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    boxes = [
        ("Qwen2.5-0.5B\\nBase Model", 60, 180, 250, 310, (236, 244, 255)),
        ("Custom LoRA\\nq_proj + v_proj", 340, 180, 560, 310, (239, 252, 246)),
        ("SFT on chosen\\n300 samples", 650, 180, 850, 310, (255, 248, 230)),
        ("DPO Training\\n100 preference pairs", 940, 180, 1160, 310, (255, 238, 238)),
        ("Validation\\nloss + outputs", 1230, 180, 1360, 310, (241, 245, 249)),
    ]
    draw.text((60, 60), "Efficient LLM Post-Training Pipeline", fill=(20, 32, 46), font=font)
    for text, x1, y1, x2, y2, color in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=20, fill=color, outline=(70, 85, 100), width=3)
        lines = text.split("\\n")
        for i, line in enumerate(lines):
            draw.text((x1 + 18, y1 + 35 + i * 28), line, fill=(20, 32, 46), font=font)
    for _, _, _, x2, y2, _ in boxes[:-1]:
        next_x = boxes[boxes.index((_, _, _, x2, y2, _)) + 1][1] if False else None
    for idx in range(len(boxes) - 1):
        _, _, _, x2, _, _ = boxes[idx]
        _, nx1, _, _, _, _ = boxes[idx + 1]
        y = 245
        draw.line((x2 + 15, y, nx1 - 15, y), fill=(45, 55, 72), width=4)
        draw.polygon([(nx1 - 15, y), (nx1 - 35, y - 12), (nx1 - 35, y + 12)], fill=(45, 55, 72))
    draw.text((340, 350), "Manual A/B matrices, frozen base weights", fill=(70, 85, 100), font=font)
    draw.text((940, 350), "Manual DPO loss: normal < swapped", fill=(70, 85, 100), font=font)
    image.save(output_path)


def extract_comparison(markdown: str) -> list[tuple[str, str, str]]:
    chunks = markdown.split("## Prompt")
    examples: list[tuple[str, str, str]] = []
    for chunk in chunks[1:]:
        prompt_match = re.search(r"\n\n(.*?)\n\n### Base", chunk, re.S)
        base_match = re.search(r"### Base\n\n(.*?)\n\n### SFT-LoRA", chunk, re.S)
        tuned_match = re.search(r"### SFT-LoRA\n\n(.*)", chunk, re.S)
        if prompt_match and base_match and tuned_match:
            prompt = " ".join(prompt_match.group(1).split())
            base = " ".join(base_match.group(1).split())
            tuned = " ".join(tuned_match.group(1).split())
            examples.append((prompt, base[:420], tuned[:420]))
    return examples


def make_notebook(sft_rows: list[tuple[int, float]], dpo_rows: list[tuple[int, float]], dpo_check: str) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Lab2 Results Notebook\\n",
                    "\\n",
                    "This notebook records the training artifacts used in `report.pdf`.\\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Key Metrics\\n",
                    f"- SFT samples: 300\\n",
                    f"- SFT final loss: {sft_rows[-1][1]:.4f}\\n",
                    f"- DPO samples: 100 preference pairs\\n",
                    f"- DPO final loss: {dpo_rows[-1][1]:.4f}\\n",
                    f"- DPO check: `{dpo_check.strip().replace(chr(10), '; ')}`\\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## SFT Loss Curve\\n", "![SFT loss](../outputs/sft_loss_curve.png)\\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## DPO Loss Curve\\n", "![DPO loss](../outputs/dpo_loss_curve.png)\\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Output Comparison\\n", "See `../outputs/base_vs_lora.md` for full generated examples.\\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "lab2_results.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_report(
    sft_rows: list[tuple[int, float]],
    dpo_rows: list[tuple[int, float]],
    dpo_check: str,
    examples: list[tuple[str, str, str]],
) -> None:
    styles = getSampleStyleSheet()
    title = styles["Title"]
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    body.leading = 14
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=10)
    doc = SimpleDocTemplate(
        str(ROOT / "report.pdf"),
        pagesize=A4,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    story = []
    story.append(p("Lab2: Efficient LLM Post-Training Pipeline", title))
    story.append(p("Custom LoRA SFT and DPO Training for Math Reasoning", h2))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        p(
            "This report documents a parameter-efficient post-training pipeline using Qwen2.5-0.5B-Instruct, "
            "manual LoRA adapters, and a manually implemented DPO objective. The experiment uses a small subset "
            "of Math-Step-DPO-10K to validate implementation correctness under limited compute.",
            body,
        )
    )
    metrics = [
        ["Item", "Value"],
        ["Base model", "Qwen/Qwen2.5-0.5B-Instruct"],
        ["Dataset", "xinlai/Math-Step-DPO-10K"],
        ["SFT samples", "300 chosen responses"],
        ["DPO samples", "100 chosen/rejected pairs"],
        ["Trainable parameters", "540,672 / 494,573,440 (0.11%)"],
        ["SFT final loss", f"{sft_rows[-1][1]:.4f}"],
        ["DPO final loss", f"{dpo_rows[-1][1]:.4f}"],
        ["DPO validation", dpo_check.strip().replace("\\n", "; ")],
    ]
    table = Table(metrics, colWidths=[1.9 * inch, 4.9 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(table)
    story.append(Spacer(1, 0.22 * inch))
    story.append(RLImage(str(OUTPUTS / "pipeline_diagram.png"), width=6.9 * inch, height=2.56 * inch))

    story.append(PageBreak())
    story.append(p("Implementation", h1))
    bullets = [
        "LoRA was implemented manually as Wx + scale * B(Ax), where scale = alpha / rank.",
        "Base model weights are frozen; only lora_a and lora_b are trainable.",
        "Adapters are injected into q_proj and v_proj in each Transformer attention block.",
        "DPO loss was implemented manually from policy/reference chosen and rejected sequence log-probabilities.",
        "The reference model is frozen during DPO; gradients flow only through the policy LoRA adapters.",
    ]
    story.append(ListFlowable([ListItem(p(item, body)) for item in bullets], bulletType="bullet"))
    story.append(Spacer(1, 0.18 * inch))
    story.append(p("Training Curves", h1))
    story.append(RLImage(str(OUTPUTS / "sft_loss_curve.png"), width=6.5 * inch, height=3.9 * inch))
    story.append(Spacer(1, 0.12 * inch))
    story.append(RLImage(str(OUTPUTS / "dpo_loss_curve.png"), width=6.5 * inch, height=3.9 * inch))

    story.append(PageBreak())
    story.append(p("Validation Results", h1))
    story.append(
        p(
            "The DPO validation computes the loss once with the original chosen/rejected order and once after "
            "swapping the pair. The normal loss is lower than the swapped loss, which confirms that the preference "
            "direction is represented by the policy/reference log-probability ratio.",
            body,
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    dpo_table = Table(
        [
            ["Metric", "Value"],
            ["Normal DPO loss", "0.532641"],
            ["Swapped DPO loss", "1.022753"],
            ["Batch size", "16"],
        ],
        colWidths=[2.2 * inch, 2.2 * inch],
    )
    dpo_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(dpo_table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(p("Qualitative Output Comparison", h1))
    for prompt, base, tuned in examples[:2]:
        story.append(p(f"<b>Prompt:</b> {prompt}", body))
        story.append(p(f"<b>Base:</b> {base}", small))
        story.append(p(f"<b>Fine-tuned:</b> {tuned}", small))
        story.append(Spacer(1, 0.12 * inch))

    story.append(PageBreak())
    story.append(p("AI Collaboration Diary", h1))
    diary = [
        [
            "LoRA scaffold",
            "Prompt: 'Start building the project skeleton.'",
            "Iteration: Initial code passed unit tests but later failed on GPU because new LoRA matrices stayed on CPU.",
            "Verification: Added tests for LoRA freezing and dtype inheritance; fixed adapter initialization to inherit base device and dtype.",
            "Learning: Device placement is part of correct adapter implementation, not just a runtime detail.",
        ],
        [
            "Hugging Face downloads",
            "Prompt: 'The download stays at 0 percent / the attached traceback shows a network error.'",
            "Iteration: First failure came from hf-xet 401; later a parquet download failed with IncompleteRead.",
            "Verification: Disabled Xet, added local parquet loading, and used curl resume downloads with the platform academic accelerator.",
            "Learning: Reproducible experiments should avoid fragile implicit downloads when the runtime network is unstable.",
        ],
        [
            "DPO training",
            "Prompt: 'I want to do complete DPO training; my GPU has 24GB.'",
            "Iteration: Added train_dpo.py using a frozen reference model and policy LoRA initialized from SFT.",
            "Verification: DPO training completed on 100 preference pairs; normal loss was 0.532641 and swapped loss was 1.022753.",
            "Learning: DPO can be validated mechanically by checking that swapping preference pairs changes the loss direction.",
        ],
    ]
    for title_text, prompt, iteration, verification, learning in diary:
        story.append(p(title_text, h2))
        story.append(p(prompt, body))
        story.append(p(iteration, body))
        story.append(p(verification, body))
        story.append(p(learning, body))
        story.append(Spacer(1, 0.12 * inch))

    story.append(PageBreak())
    story.append(p("Resource Constraints and Submission Notes", h1))
    story.append(
        p(
            "Training ran on an RTX 4090D 24GB GPU. SFT used 300 samples and DPO used 100 preference pairs, both "
            "well below the assignment time limit. The small trainable parameter ratio kept memory usage modest. "
            "The ZIP submission should include src/, notebooks/, report.pdf, requirements.txt, and lightweight "
            "outputs. Model weights and dataset cache files are excluded.",
            body,
        )
    )
    story.append(Spacer(1, 0.16 * inch))
    story.append(p("Generated Artifacts", h2))
    story.append(
        ListFlowable(
            [
                ListItem(p("report.pdf", body)),
                ListItem(p("notebooks/lab2_results.ipynb", body)),
                ListItem(p("outputs/sft_loss_curve.png and outputs/dpo_loss_curve.png", body)),
                ListItem(p("outputs/loss_log.csv, outputs/dpo_loss_log.csv, outputs/dpo_check.txt, outputs/base_vs_lora.md", body)),
            ],
            bulletType="bullet",
        )
    )
    doc.build(story)


def main() -> None:
    sft_rows = read_loss_csv(OUTPUTS / "loss_log.csv")
    dpo_rows = read_loss_csv(OUTPUTS / "dpo_loss_log.csv")
    dpo_check = (OUTPUTS / "dpo_check.txt").read_text(encoding="utf-8")
    comparison_md = (OUTPUTS / "base_vs_lora.md").read_text(encoding="utf-8")
    examples = extract_comparison(comparison_md)
    draw_line_chart(sft_rows, "SFT Training Loss (300 samples)", "loss", OUTPUTS / "sft_loss_curve.png")
    draw_line_chart(dpo_rows, "DPO Training Loss (100 preference pairs)", "loss", OUTPUTS / "dpo_loss_curve.png")
    draw_pipeline_diagram(OUTPUTS / "pipeline_diagram.png")
    make_notebook(sft_rows, dpo_rows, dpo_check)
    make_report(sft_rows, dpo_rows, dpo_check, examples)


if __name__ == "__main__":
    main()
