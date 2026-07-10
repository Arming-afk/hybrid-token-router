"""Thin async client. ALL calls must go through FIREWORKS_BASE_URL (the judging proxy)."""
import asyncio
import os
import random

from openai import APIStatusError, AsyncOpenAI

RETRIES = 2
RATE_LIMIT_RETRIES = 4  # 429s get their own, more patient budget
CALL_TIMEOUT_SECONDS = 25  # harness hard rule: response time per request must stay under 30s

# Reasoning models (e.g. minimax-m3) bill hidden reasoning tokens and can burn the
# whole max_tokens budget thinking, returning blank content on hard prompts. "none"
# suppresses that. Sent to every model by default; one rejects it as an unknown
# param, _NO_EFFORT_PARAM remembers that model and stops sending it.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")
_NO_EFFORT_PARAM: set[str] = set()

_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"],
            max_retries=0,  # retry policy lives in complete(); SDK default would multiply it
        )
    return _client


async def aclose() -> None:
    """Close the client inside the running loop; letting GC do it after the loop
    is gone crashes the interpreter at shutdown (exit 139) on Windows."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def _is_permanent(error: Exception) -> bool:
    # 4xx (except 429) fails identically on retry; don't burn deadline time on it.
    return (isinstance(error, APIStatusError)
            and error.status_code < 500 and error.status_code != 429)


def _is_rate_limit(error: Exception) -> bool:
    return isinstance(error, APIStatusError) and error.status_code == 429


async def complete(model: str, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    """Return (text, usage). Retries transient failures; raises after the last attempt.

    429s get a separate, more patient budget: an empty answer is a guaranteed judge
    fail, so waiting out a rate-limit window (with jitter, so concurrent tasks don't
    re-stampede the proxy) is always worth the time while the global deadline allows.
    """
    last_error = None
    failures = 0
    rate_hits = 0
    while True:
        sent_effort = bool(REASONING_EFFORT) and model not in _NO_EFFORT_PARAM
        kwargs = {"reasoning_effort": REASONING_EFFORT} if sent_effort else {}
        try:
            response = await asyncio.wait_for(
                get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                    **kwargs,
                ),
                timeout=CALL_TIMEOUT_SECONDS,
            )
            usage = response.usage
            choice = response.choices[0]
            return choice.message.content or "", {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                # A cap-truncated answer (code cut mid-function, missing Answer
                # line) is likely wrong; callers escalate it like a blank instead
                # of submitting it.
                "truncated": getattr(choice, "finish_reason", None) == "length",
            }
        except Exception as error:
            last_error = error
            if sent_effort and _is_permanent(error):
                # Model rejected reasoning_effort as an unknown param; drop it for
                # this model and retry immediately, free, before the failure budget.
                _NO_EFFORT_PARAM.add(model)
                continue
            if _is_permanent(error):
                break
            if _is_rate_limit(error):
                rate_hits += 1
                if rate_hits > RATE_LIMIT_RETRIES:
                    break
                await asyncio.sleep(4 * rate_hits + random.uniform(0, 3))
                continue
            failures += 1
            if failures > RETRIES:
                break
            await asyncio.sleep(2 * failures - 1)
    raise last_error
