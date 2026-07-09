"""Assign ALLOWED_MODELS (unknown until launch day) to SMALL/MEDIUM/LARGE/CODE tiers.

Robustness concerns driving this module:
- Reasoning models (ids containing r1/qwq/"thinking") emit large hidden reasoning
  traces that are billed in full. They must never occupy SMALL or MEDIUM, which the
  common per-category path uses; they are only allowed in LARGE (client.py further
  suppresses their thinking with reasoning_effort="none").
- Model ids encode size inconsistently. MoE ids like "mixtral-8x7b" are read as total
  params (8*7=56b) rather than the 7b a naive parse would give, so a mid-size MoE is
  not mistaken for the smallest model.
- Code-specialized models (ids containing "code"/"coder") serve debug/codegen via the
  CODE tier and are excluded from the general tiers — an unsized code MoE must not
  shadow the real LARGE pick. With no code model, CODE falls back to MEDIUM.
"""
import os
import re

# Ordering: check MoE "NxMb" first so "8x7b" isn't read as the trailing "7b".
_MOE = re.compile(r"(\d+)\s*x\s*(\d+)b\b")
_SIZE = re.compile(r"(\d+(?:p\d+)?)b\b")
_REASONING = re.compile(r"\br1\b|\bo1\b|\bqwq\b|reasoning|thinking|deepthink|-think\b", re.I)
_CODE = re.compile(r"\bcode|coder", re.I)
# Low-precision builds (e.g. gemma-...-nvfp4). Same nominal size, lower quality.
_QUANT = re.compile(r"nvfp4|fp4|fp8|int4|int8|awq|gptq|gguf|-q[48]\b|[48]bit|bnb", re.I)

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


def _is_code(model_id: str) -> bool:
    return bool(_CODE.search(model_id))


def _tier_rank(model_id: str) -> float:
    # Ranking key for the SMALL/MEDIUM pool. A quantized build is de-ranked a hair so
    # that, at equal nominal size, the full-precision build wins the accuracy-sensitive
    # MEDIUM slot (a default Person 2 can override once eval measures the quantized one).
    return _size_b(model_id) - (0.5 if _QUANT.search(model_id) else 0.0)


def build_tiers() -> dict[str, str]:
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if not models:
        raise ValueError("ALLOWED_MODELS is empty")
    code_models = [m for m in models if _is_code(m)]
    general = [m for m in models if not _is_code(m)] or models
    # SMALL/MEDIUM are the frequently-used tiers -> draw them from non-reasoning models
    # whenever any exist; fall back to the full set only if every model reasons.
    pool = sorted([m for m in general if not _is_reasoning(m)] or general, key=_tier_rank)
    tiers = {
        "SMALL": pool[0],
        "MEDIUM": pool[len(pool) // 2],
        "LARGE": max(general, key=_size_b),  # hardest tier: most capable general model
    }
    tiers["CODE"] = max(code_models, key=_size_b) if code_models else tiers["MEDIUM"]
    return tiers
