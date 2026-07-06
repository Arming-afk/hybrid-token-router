"""Assign ALLOWED_MODELS (unknown until launch day) to SMALL/MEDIUM/LARGE tiers by parsed size."""
import os
import re


def _size_b(model_id: str) -> float:
    # "llama-v3p1-8b-instruct" -> 8.0, "qwen3-235b-a22b" -> 235.0
    # No parseable size -> treat as large so we never accidentally rank a big model as SMALL.
    sizes = [
        float(m.group(1).replace("p", "."))
        for m in re.finditer(r"(\d+(?:p\d+)?)b\b", model_id.lower())
    ]
    return max(sizes) if sizes else 999.0


def build_tiers() -> dict[str, str]:
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    models.sort(key=_size_b)
    return {
        "SMALL": models[0],
        "MEDIUM": models[len(models) // 2],
        "LARGE": models[-1],
    }
