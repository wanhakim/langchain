"""Integration tests for VLLM."""

import os

import pytest

from langchain_vllm.llms import VLLM

MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
BASE_URL = os.environ.get("VLLM_TEST_BASE_URL") or ""

# Skip the entire module when no chat server URL is configured.
pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason="VLLM_TEST_BASE_URL not set — start a chat server first",
)


def _llm(max_tokens: int = 16) -> VLLM:
    return VLLM(model=MODEL, base_url=BASE_URL, temperature=0, max_tokens=max_tokens)


def test_invoke() -> None:
    """Sync invoke returns a non-empty string."""
    result = _llm().invoke("The capital of France is")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_ainvoke() -> None:
    """Async invoke returns a non-empty string."""
    result = await _llm().ainvoke("The capital of France is")
    assert isinstance(result, str)
    assert len(result) > 0


def test_stream() -> None:
    """Streaming yields string tokens."""
    tokens = list(_llm().stream("Count to three:"))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


async def test_astream() -> None:
    """Async streaming yields string tokens."""
    tokens = [t async for t in _llm().astream("Count to three:")]
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_batch() -> None:
    """Batch sends all prompts in one call and returns one result per prompt."""
    results = _llm().batch(["1 + 1 =", "2 + 2 ="])
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


async def test_abatch() -> None:
    """Async batch returns one result per prompt."""
    results = await _llm().abatch(["Hello", "Goodbye"])
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
