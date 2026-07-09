"""client.py regression gate, no network needed.

Covers the reasoning_effort fallback added after the 68.4% -> 57.9% scored-run
regression: minimax-m3 (LARGE) burns max_tokens on hidden reasoning and returns
blank content unless reasoning_effort="none" is sent; some models reject the
param outright, so client.py must drop it per-model and retry for free.
"""
import asyncio
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FIREWORKS_API_KEY", "test-key")
os.environ.setdefault("FIREWORKS_BASE_URL", "http://localhost/v1")

_fake_openai = types.ModuleType("openai")


class FakeAPIStatusError(Exception):
    def __init__(self, status_code, message="error"):
        super().__init__(message)
        self.status_code = status_code


class FakeAsyncOpenAI:
    def __init__(self, **kwargs):
        pass


_fake_openai.APIStatusError = FakeAPIStatusError
_fake_openai.AsyncOpenAI = FakeAsyncOpenAI
sys.modules["openai"] = _fake_openai

from src import client  # noqa: E402


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt_tokens=1, completion_tokens=1):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class FakeCompletions:
    def __init__(self, script):
        """script: list of either an Exception to raise or a content string to return."""
        self.script = list(script)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return _Response(step)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, script):
        self.completions = FakeCompletions(script)
        self.chat = FakeChat(self.completions)

    async def close(self):
        pass


def _install(script):
    fake = FakeClient(script)
    client._client = fake
    client._NO_EFFORT_PARAM.clear()
    return fake


def test_reasoning_effort_sent_by_default():
    fake = _install(["answer"])
    text, _ = asyncio.run(client.complete("acc/big-70b", [{"role": "user", "content": "x"}], 10))
    assert text == "answer"
    assert fake.completions.calls[0]["reasoning_effort"] == "none"


def test_model_rejecting_param_retries_without_it_for_free():
    fake = _install([FakeAPIStatusError(400, "invalid_request_error: unknown param"), "answer"])
    text, _ = asyncio.run(client.complete("acc/big-70b", [{"role": "user", "content": "x"}], 10))
    assert text == "answer"
    assert "reasoning_effort" in fake.completions.calls[0]
    assert "reasoning_effort" not in fake.completions.calls[1]
    assert "acc/big-70b" in client._NO_EFFORT_PARAM


def test_rejected_model_skips_param_on_next_call():
    _install(["first"])
    asyncio.run(client.complete("acc/rejects", [{"role": "user", "content": "x"}], 10))
    # Simulate the model already having been marked in a prior call this run.
    client._NO_EFFORT_PARAM.add("acc/rejects")
    fake = FakeClient(["second"])
    client._client = fake
    text, _ = asyncio.run(client.complete("acc/rejects", [{"role": "user", "content": "x"}], 10))
    assert text == "second"
    assert "reasoning_effort" not in fake.completions.calls[0]


def test_unrelated_400_still_fails_after_free_retry():
    # A genuinely bad request fails again once reasoning_effort is dropped, and
    # must not loop forever or get treated as free a second time.
    _install([
        FakeAPIStatusError(400, "bad request"),
        FakeAPIStatusError(400, "still bad"),
    ])
    try:
        asyncio.run(client.complete("acc/broken", [{"role": "user", "content": "x"}], 10))
        assert False, "expected an exception"
    except FakeAPIStatusError:
        pass


def test_call_timeout_under_hard_rule():
    assert client.CALL_TIMEOUT_SECONDS < 30


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\ntest_client: {len(fns)} passed")
