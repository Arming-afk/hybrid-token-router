"""Thin async client. ALL calls must go through FIREWORKS_BASE_URL (the judging proxy)."""
import asyncio
import os

from openai import APIStatusError, AsyncOpenAI

RETRIES = 2
CALL_TIMEOUT_SECONDS = 60

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
            if _is_permanent(error):
                break
            if attempt < RETRIES:
                await asyncio.sleep(1 + 2 * attempt)
    raise last_error
