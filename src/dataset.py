from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str


def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or str(item)
                parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def normalize_preference_row(row: dict[str, Any]) -> PreferenceExample:
    prompt = row.get("prompt") or row.get("question") or row.get("instruction") or row.get("input")
    chosen = row.get("chosen") or row.get("response_chosen") or row.get("winner")
    rejected = row.get("rejected") or row.get("response_rejected") or row.get("loser")

    if prompt is None or chosen is None or rejected is None:
        keys = ", ".join(sorted(row.keys()))
        raise KeyError(f"Could not find prompt/chosen/rejected fields in row keys: {keys}")

    return PreferenceExample(
        prompt=_text_from_value(prompt),
        chosen=_text_from_value(chosen),
        rejected=_text_from_value(rejected),
    )


def load_math_step_dpo(
    sample_count: int | None = None,
    split: str = "train",
    data_file: str | None = None,
) -> list[PreferenceExample]:
    split_expr = split if not sample_count or sample_count <= 0 else f"{split}[:{sample_count}]"
    local_data_file = data_file or os.environ.get("MATH_STEP_DPO_PARQUET")
    if local_data_file:
        dataset = load_dataset("parquet", data_files={split: local_data_file}, split=split_expr)
    else:
        dataset = load_dataset("xinlai/Math-Step-DPO-10K", split=split_expr)
    return [normalize_preference_row(row) for row in dataset]
