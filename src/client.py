"""Thin async client. ALL calls must go through FIREWORKS_BASE_URL (the judging proxy)."""
import asyncio
import os

from openai import AsyncOpenAI

RETRIES = 2
CALL_TIMEOUT_SECONDS = 60

_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"],
        )
    return _client


async def complete(model: str, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    """Return (text, usage). Retries transient failures; raises after the last attempt."""
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                ),
                timeout=CALL_TIMEOUT_SECONDS,
            )
            usage = response.usage
            return response.choices[0].message.content or "", {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
        except Exception as error:
            last_error = error
            if attempt < RETRIES:
                await asyncio.sleep(1 + 2 * attempt)
    raise last_error
