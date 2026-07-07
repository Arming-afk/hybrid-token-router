"""Assign ALLOWED_MODELS (unknown until launch day) to SMALL/MEDIUM/LARGE tiers.

Two robustness concerns drive this module:
- Reasoning models (ids containing r1/qwq/"thinking") emit large hidden reasoning
  traces that are billed in full. They must never occupy SMALL or MEDIUM, which the
  common per-category path uses; they are only allowed in LARGE, the rare retry tier.
- Model ids encode size inconsistently. MoE ids like "mixtral-8x7b" are read as total
  params (8*7=56b) rather than the 7b a naive parse would give, so a mid-size MoE is
  not mistaken for the smallest model.
"""
import os
import re

# Ordering: check MoE "NxMb" first so "8x7b" isn't read as the trailing "7b".
_MOE = re.compile(r"(\d+)\s*x\s*(\d+)b\b")
_SIZE = re.compile(r"(\d+(?:p\d+)?)b\b")
_REASONING = re.compile(r"\br1\b|\bo1\b|\bqwq\b|reasoning|thinking|deepthink|-think\b", re.I)

_UNKNOWN_SIZE = 999.0  # unparseable id -> treat as large, never as the SMALL default


def _size_b(model_id: str) -> float:
    text = model_id.lower()
    moe = _MOE.findall(text)
    if moe:
        return float(max(int(n) * int(m) for n, m in moe))
    sizes = [float(m.group(1).replace("p", ".")) for m in _SIZE.finditer(text)]
    return max(sizes) if sizes else _UNKNOWN_SIZE


def _is_reasoning(model_id: str) -> bool:
    return bool(_REASONING.search(model_id))


def build_tiers() -> dict[str, str]:
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if not models:
        raise ValueError("ALLOWED_MODELS is empty")
    # SMALL/MEDIUM are the frequently-used tiers -> draw them from non-reasoning models
    # whenever any exist; fall back to the full set only if every model reasons.
    pool = sorted([m for m in models if not _is_reasoning(m)] or models, key=_size_b)
    return {
        "SMALL": pool[0],
        "MEDIUM": pool[len(pool) // 2],
        "LARGE": max(models, key=_size_b),  # hardest/retry tier: most capable overall
    }
