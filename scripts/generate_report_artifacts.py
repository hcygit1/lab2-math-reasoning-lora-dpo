from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
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

CHART_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def chart_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in CHART_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_loss_csv(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["step"]), float(row["loss"])))
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def draw_line_chart(
    data: list[tuple[int, float]],
    title: str,
    output_path: Path,
    width: int = 1200,
    height: int = 720,
) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = chart_font(30)
    label_font = chart_font(20)
    small_font = chart_font(17)
    left, right, top, bottom = 105, width - 60, 86, height - 90

    steps = [step for step, _ in data]
    losses = [loss for _, loss in data]
    smooth = moving_average(losses, window=max(5, len(losses) // 25))
    min_x, max_x = min(steps), max(steps)
    min_y, max_y = min(losses), max(losses)
    pad_y = max((max_y - min_y) * 0.12, 0.05)
    min_y -= pad_y
    max_y += pad_y

    draw.text((left, 28), title, fill=(20, 32, 46), font=title_font)
    draw.line((left, bottom, right, bottom), fill=(60, 70, 80), width=2)
    draw.line((left, top, left, bottom), fill=(60, 70, 80), width=2)

    for i in range(6):
        ratio = i / 5
        y = bottom - ratio * (bottom - top)
        value = min_y + ratio * (max_y - min_y)
        draw.line((left, y, right, y), fill=(225, 230, 235), width=1)
        draw.text((22, y - 10), f"{value:.2f}", fill=(80, 90, 100), font=small_font)

    for i in range(6):
        ratio = i / 5
        x = left + ratio * (right - left)
        value = round(min_x + ratio * (max_x - min_x))
        draw.line((x, bottom, x, bottom + 6), fill=(60, 70, 80), width=1)
        draw.text((x - 14, bottom + 18), str(value), fill=(80, 90, 100), font=small_font)

    def point(step: int, loss: float) -> tuple[float, float]:
        x_ratio = 0 if max_x == min_x else (step - min_x) / (max_x - min_x)
        y_ratio = 0 if max_y == min_y else (loss - min_y) / (max_y - min_y)
        return left + x_ratio * (right - left), bottom - y_ratio * (bottom - top)

    raw_points = [point(step, loss) for step, loss in data]
    smooth_points = [point(step, loss) for (step, _), loss in zip(data, smooth)]
    if len(raw_points) >= 2:
        draw.line(raw_points, fill=(155, 174, 193), width=2)
        draw.line(smooth_points, fill=(22, 107, 198), width=4)
    for x, y in raw_points[:: max(1, len(raw_points) // 45)]:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(22, 107, 198))

    draw.text((width // 2 - 55, height - 45), "Training Step", fill=(60, 70, 80), font=label_font)
    draw.text((18, 52), "Loss", fill=(60, 70, 80), font=label_font)
    draw.text((right - 305, top + 10), "Light line: raw step loss", fill=(120, 135, 150), font=small_font)
    draw.text((right - 305, top + 36), "Blue line: moving average", fill=(22, 107, 198), font=small_font)
    image.save(output_path)


def draw_pipeline_diagram(output_path: Path) -> None:
    image = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(image)
    title_font = chart_font(38)
    stage_font = chart_font(25)
    body_font = chart_font(22)
    note_font = chart_font(19)

    draw.text((70, 48), "Efficient LLM Post-Training Pipeline", fill=(20, 32, 46), font=title_font)
    draw.text(
        (70, 96),
        "Base model -> custom LoRA SFT -> full DPO training -> mechanism verification.",
        fill=(70, 85, 100),
        font=note_font,
    )

    stages = [
        (
            "1. Base Model",
            "Qwen2.5-0.5B-Instruct",
            "Load tokenizer and causal LM; used as SFT init and DPO reference source.",
            (236, 244, 255),
        ),
        (
            "2. Custom LoRA",
            "Freeze base; train low-rank A/B matrices",
            "Inject LoRA into q_proj and v_proj: Wx + scale * B(Ax).",
            (239, 252, 246),
        ),
        (
            "3. SFT Stage",
            "Math-Step-DPO-10K: prompt -> chosen",
            "Train the LoRA adapter on chosen answers and save outputs/sft_lora.",
            (255, 248, 230),
        ),
        (
            "4. DPO Stage",
            "policy = SFT LoRA; reference = frozen base",
            "Use chosen/rejected preference pairs and custom DPO loss to update policy LoRA.",
            (255, 238, 238),
        ),
        (
            "5. Validation",
            "Loss curves, swapped DPO check, output comparison",
            "Check normal loss < swapped loss; generate curves and base_vs_lora.",
            (241, 245, 249),
        ),
    ]

    x1, x2 = 90, 1310
    y = 150
    box_h = 118
    gap = 26
    for index, (stage, headline, detail, color) in enumerate(stages):
        y1 = y + index * (box_h + gap)
        y2 = y1 + box_h
        draw.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=color, outline=(71, 85, 105), width=3)
        draw.rounded_rectangle((x1, y1, x1 + 245, y2), radius=22, fill=(30, 41, 59), outline=(30, 41, 59), width=0)
        draw.rectangle((x1 + 220, y1, x1 + 245, y2), fill=(30, 41, 59))
        draw.text((x1 + 28, y1 + 40), stage, fill="white", font=stage_font)
        draw.text((x1 + 285, y1 + 25), headline, fill=(20, 32, 46), font=stage_font)
        draw.text((x1 + 285, y1 + 68), detail, fill=(55, 65, 81), font=body_font)

        if index < len(stages) - 1:
            cx = (x1 + x2) // 2
            arrow_top = y2 + 6
            arrow_bottom = y2 + gap - 6
            draw.line((cx, arrow_top, cx, arrow_bottom), fill=(45, 55, 72), width=4)
            draw.polygon(
                [(cx, arrow_bottom + 8), (cx - 12, arrow_bottom - 8), (cx + 12, arrow_bottom - 8)],
                fill=(45, 55, 72),
            )

    draw.text(
        (90, 850),
        "Boundary: LoRA adapter and DPO loss are custom core logic; transformers handles model loading and forward passes.",
        fill=(70, 85, 100),
        font=note_font,
    )
    image.save(output_path)


def extract_comparison(markdown: str) -> list[tuple[str, str, str]]:
    chunks = markdown.split("## Prompt")
    examples: list[tuple[str, str, str]] = []
    for chunk in chunks[1:]:
        prompt_match = re.search(r"\n\n(.*?)\n\n### Base", chunk, re.S)
        base_match = re.search(r"### Base\n\n(.*?)\n\n### (?:SFT|DPO)-LoRA", chunk, re.S)
        tuned_match = re.search(r"### (?:SFT|DPO)-LoRA\n\n(.*)", chunk, re.S)
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
                    "# Lab2 实验结果记录\n",
                    "\n",
                    "本 notebook 汇总 `report.pdf` 中使用的训练日志、曲线和验证结果。\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 关键指标\n",
                    "- SFT 样本数：300\n",
                    f"- SFT 最终 loss：{sft_rows[-1][1]:.4f}\n",
                    "- DPO 样本数：100 对偏好样本\n",
                    f"- DPO 最终 loss：{dpo_rows[-1][1]:.4f}\n",
                    f"- DPO 验证：`{dpo_check.strip().replace(chr(10), '; ')}`\n",
                    "\n",
                    "loss 曲线波动较大是正常现象：本实验使用 batch size=1，单步 loss 受样本难度、回答长度和偏好对差异影响很大，因此报告中同时展示逐步 loss 和滑动平均趋势。\n",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## SFT 训练曲线\n", "![SFT loss](../outputs/sft_loss_curve.png)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## DPO 训练曲线\n", "![DPO loss](../outputs/dpo_loss_curve.png)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 输出对比\n", "完整文本见 `../outputs/base_vs_lora.md`。\n"]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / "lab2_results.ipynb").write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


def make_styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    base = "STSong-Light"
    return {
        "title": ParagraphStyle("ChineseTitle", parent=styles["Title"], fontName=base, fontSize=20, leading=26),
        "h1": ParagraphStyle("ChineseH1", parent=styles["Heading1"], fontName=base, fontSize=15, leading=20, spaceAfter=8),
        "h2": ParagraphStyle("ChineseH2", parent=styles["Heading2"], fontName=base, fontSize=12, leading=16, spaceAfter=5),
        "body": ParagraphStyle("ChineseBody", parent=styles["BodyText"], fontName=base, fontSize=9.5, leading=13),
        "small": ParagraphStyle("ChineseSmall", parent=styles["BodyText"], fontName=base, fontSize=7.5, leading=10),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_report(
    sft_rows: list[tuple[int, float]],
    dpo_rows: list[tuple[int, float]],
    dpo_check: str,
    examples: list[tuple[str, str, str]],
) -> None:
    st = make_styles()
    doc = SimpleDocTemplate(
        str(ROOT / "report.pdf"),
        pagesize=A4,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    story = []
    story.append(p("Lab2：高效大模型后训练流水线", st["title"]))
    story.append(p("基于手写 LoRA 与 DPO 的数学推理后训练实验", st["h2"]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(
        p(
            "本实验使用 Qwen2.5-0.5B-Instruct 作为基础模型，在 Math-Step-DPO-10K 小规模子集上完成参数高效后训练。"
            "核心实现包括手写 LoRA 适配器、SFT 训练、手写 DPO loss、完整 DPO 小规模训练，以及 Base 与后训练模型的输出对比。",
            st["body"],
        )
    )
    metrics = [
        ["项目", "结果"],
        ["基础模型", "Qwen/Qwen2.5-0.5B-Instruct"],
        ["数据集", "xinlai/Math-Step-DPO-10K"],
        ["SFT 样本数", "300 条 chosen 回答"],
        ["DPO 样本数", "100 对 chosen/rejected 偏好样本"],
        ["可训练参数", "540,672 / 494,573,440 (0.11%)"],
        ["SFT 最终 loss", f"{sft_rows[-1][1]:.4f}"],
        ["DPO 最终 loss", f"{dpo_rows[-1][1]:.4f}"],
        ["DPO 验证", dpo_check.strip().replace("\n", "；")],
    ]
    table = Table(metrics, colWidths=[1.7 * inch, 5.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(Spacer(1, 0.14 * inch))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(RLImage(str(OUTPUTS / "pipeline_diagram.png"), width=6.9 * inch, height=2.56 * inch))

    story.append(PageBreak())
    story.append(p("一、实现说明", st["h1"]))
    bullets = [
        "LoRA 前向传播手写为 Wx + scale * B(Ax)，其中 scale = alpha / rank。",
        "基础模型参数全部冻结，只训练 lora_a 与 lora_b 两个低秩矩阵。",
        "LoRA 注入 Transformer 注意力模块中的 q_proj 与 v_proj。",
        "DPO loss 使用 policy/reference 在 chosen 与 rejected 序列上的 log probability 手写计算。",
        "DPO 训练时 reference model 冻结，梯度只回传到 policy model 的 LoRA 参数。",
    ]
    story.append(ListFlowable([ListItem(p(item, st["body"])) for item in bullets], bulletType="bullet"))
    story.append(Spacer(1, 0.12 * inch))
    story.append(p("二、训练曲线与波动解释", st["h1"]))
    story.append(
        p(
            "曲线波动较大是本实验设置下的正常现象。训练时 batch size=1，每一步只对应一条样本或一对偏好样本；"
            "数学题长度、推理难度、chosen/rejected 差距都不均匀，所以逐步 loss 会出现明显噪声。"
            "此外本实验只训练 300 条 SFT 样本和 100 对 DPO 样本，目标是验证流程正确性而不是追求平滑收敛。"
            "因此图中同时展示浅色的逐步原始 loss 和蓝色的滑动平均趋势，报告分析应以趋势和验证指标为主。",
            st["body"],
        )
    )
    story.append(RLImage(str(OUTPUTS / "sft_loss_curve.png"), width=6.5 * inch, height=3.9 * inch))
    story.append(Spacer(1, 0.08 * inch))
    story.append(RLImage(str(OUTPUTS / "dpo_loss_curve.png"), width=6.5 * inch, height=3.9 * inch))

    story.append(PageBreak())
    story.append(p("三、DPO 验证与输出对比", st["h1"]))
    story.append(
        p(
            "DPO 验证分别计算正常 chosen/rejected 顺序和交换顺序后的 loss。实验结果显示 normal loss 小于 swapped loss，"
            "说明偏好方向能够被 policy/reference log-probability ratio 区分出来。",
            st["body"],
        )
    )
    dpo_table = Table(
        [["指标", "数值"], ["Normal DPO loss", "0.532641"], ["Swapped DPO loss", "1.022753"], ["Batch size", "16"]],
        colWidths=[2.2 * inch, 2.2 * inch],
    )
    dpo_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(dpo_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(p("定性输出对比", st["h1"]))
    for prompt, base, tuned in examples[:2]:
        story.append(p(f"<b>题目：</b>{prompt}", st["body"]))
        story.append(p(f"<b>Base：</b>{base}", st["small"]))
        story.append(p(f"<b>Fine-tuned：</b>{tuned}", st["small"]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(PageBreak())
    story.append(p("四、AI Collaboration Diary", st["h1"]))
    diary = [
        [
            "LoRA 代码搭建",
            "Prompt：根据实验二目标搭建数学推理 LoRA + DPO 项目骨架。",
            "Iteration：初始代码通过了本地单测，但在云端训练时出现 cuda:0 与 cpu 设备不一致错误。",
            "Verification：检查 LoRA 参数是否冻结、adapter 是否继承 base layer 的 device/dtype，并补充单元测试。",
            "Learning：LoRA 的正确性不只包括公式，还包括参数冻结、梯度范围和设备放置。",
        ],
        [
            "Hugging Face 下载问题",
            "Prompt：数据集下载卡在 0%，并且报 IncompleteRead 或 Xet 401。",
            "Iteration：先禁用 HF Xet 下载路径，再支持本地 parquet 文件读取，最后使用 curl 断点续传。",
            "Verification：确认 parquet 文件大小约 12MB，训练脚本能通过 --data_file 读取本地数据。",
            "Learning：在不稳定网络环境中，实验应避免依赖隐式在线下载。",
        ],
        [
            "完整 DPO 训练",
            "Prompt：24GB 显存足够，希望从单 batch 验证扩展到完整 DPO 训练。",
            "Iteration：新增 train_dpo.py，用 SFT 后 LoRA 初始化 policy，用冻结 base model 作为 reference。",
            "Verification：DPO 训练完成 100 对偏好样本，normal loss=0.532641，swapped loss=1.022753。",
            "Learning：DPO 的机械验证可以通过交换 chosen/rejected 后 loss 变化来完成。",
        ],
    ]
    for title, prompt, iteration, verification, learning in diary:
        story.append(p(title, st["h2"]))
        story.append(p(prompt, st["body"]))
        story.append(p(iteration, st["body"]))
        story.append(p(verification, st["body"]))
        story.append(p(learning, st["body"]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(PageBreak())
    story.append(p("五、资源约束与提交说明", st["h1"]))
    story.append(
        p(
            "实验运行在 RTX 4090D 24GB GPU 上。SFT 使用 300 条样本，DPO 使用 100 对偏好样本，均远低于 3 小时限制。"
            "由于 LoRA 可训练参数仅占 0.11%，显存压力较低。最终提交压缩包包含 src/、notebooks/、report.pdf、requirements.txt "
            "和轻量 outputs；不提交模型权重、adapter_model.pt、数据集 parquet 或缓存目录。",
            st["body"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(p("生成的提交材料", st["h2"]))
    story.append(
        ListFlowable(
            [
                ListItem(p("report.pdf", st["body"])),
                ListItem(p("notebooks/lab2_results.ipynb", st["body"])),
                ListItem(p("outputs/sft_loss_curve.png 与 outputs/dpo_loss_curve.png", st["body"])),
                ListItem(p("outputs/loss_log.csv、outputs/dpo_loss_log.csv、outputs/dpo_check.txt、outputs/base_vs_lora.md", st["body"])),
            ],
            bulletType="bullet",
        )
    )
    doc.build(story)


def main() -> None:
    sft_rows = read_loss_csv(OUTPUTS / "loss_log.csv")
    dpo_rows = read_loss_csv(OUTPUTS / "dpo_loss_log.csv")
    dpo_check = (OUTPUTS / "dpo_check.txt").read_text(encoding="utf-8")
    draw_line_chart(sft_rows, "SFT Loss Curve", OUTPUTS / "sft_loss_curve.png")
    draw_line_chart(dpo_rows, "DPO Loss Curve", OUTPUTS / "dpo_loss_curve.png")
    draw_pipeline_diagram(OUTPUTS / "pipeline_diagram.png")
    make_notebook(sft_rows, dpo_rows, dpo_check)


if __name__ == "__main__":
    main()
