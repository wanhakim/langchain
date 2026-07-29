"""Integration tests for ChatVLLM."""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_tests.integration_tests import ChatModelIntegrationTests
from pydantic import BaseModel, Field

from langchain_vllm.chat_models import ChatVLLM

MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BASE_URL = os.environ.get("VLLM_TEST_BASE_URL") or ""

# Skip the entire module when no chat server URL is configured.
pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason="VLLM_TEST_BASE_URL not set — start a chat server first",
)


class TestChatVLLMStandard(ChatModelIntegrationTests):
    """Standard LangChain integration test suite.

    Properties that are False cause the corresponding standard tests to be
    skipped automatically — this is expected and correct behaviour.
    """

    @property
    def chat_model_class(self) -> type[ChatVLLM]:
        return ChatVLLM

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL, "base_url": BASE_URL}

    # --- capabilities this server supports ---
    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return True

    # --- capabilities not supported; tests for these will be skipped ---
    @property
    def supports_image_inputs(self) -> bool:
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        return False

    @property
    def supports_pdf_tool_message(self) -> bool:
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def enable_vcr_tests(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Custom integration tests
# ---------------------------------------------------------------------------


def _chat(max_tokens: int = 64) -> ChatVLLM:
    return ChatVLLM(
        model=MODEL, base_url=BASE_URL, temperature=0, max_tokens=max_tokens
    )


def test_invoke() -> None:
    """Basic sync invoke returns non-empty content."""
    response = _chat().invoke("Reply with exactly: pong")
    assert isinstance(response, AIMessage)
    assert response.content


def test_invoke_usage_metadata() -> None:
    """Response carries token usage metadata."""
    response = _chat().invoke("Reply with exactly: pong")
    assert response.usage_metadata is not None
    assert response.usage_metadata["input_tokens"] > 0
    assert response.usage_metadata["output_tokens"] > 0


async def test_ainvoke() -> None:
    """Async invoke returns non-empty content."""
    response = await _chat().ainvoke("Reply with exactly: pong")
    assert isinstance(response, AIMessage)
    assert response.content


def test_stream() -> None:
    """Streaming yields multiple chunks that together form non-empty content."""
    chunks = list(_chat().stream("Count: one two three"))
    assert len(chunks) > 0
    assert any(c.content for c in chunks)


async def test_astream() -> None:
    """Async streaming yields multiple chunks."""
    chunks = [c async for c in _chat().astream("Count: one two three")]
    assert len(chunks) > 0
    assert any(c.content for c in chunks)


def test_batch() -> None:
    """Batch returns one response per prompt."""
    responses = _chat().batch(["Say yes.", "Say no."])
    assert len(responses) == 2
    assert all(isinstance(r, AIMessage) for r in responses)
    assert all(r.content for r in responses)


def test_multi_turn() -> None:
    """Multi-turn conversation preserves context."""
    messages = [
        HumanMessage("My name is Alice."),
        AIMessage("Hello Alice!"),
        HumanMessage("What is my name?"),
    ]
    response = _chat().invoke(messages)
    assert isinstance(response, AIMessage)
    assert response.content


def test_tool_calling() -> None:
    """bind_tools returns tool calls when the server supports them.

    Requires the server to be launched with --enable-auto-tool-choice
    and --tool-call-parser hermes.
    """

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny in {city}"

    response = (
        _chat(max_tokens=128)
        .bind_tools([get_weather])
        .invoke("What is the weather in Paris? Use the tool.")
    )
    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["name"] == "get_weather"


def test_structured_output_json_schema() -> None:
    """with_structured_output returns a validated Pydantic instance."""

    class City(BaseModel):
        """A city and its country."""

        name: str = Field(description="City name")
        country: str = Field(description="Country name")

    result = (
        _chat(max_tokens=128)
        .with_structured_output(City, method="json_schema")
        .invoke("Tell me about Paris.")
    )
    assert isinstance(result, City)
    assert result.name
    assert result.country


def test_structured_output_function_calling() -> None:
    """with_structured_output via function_calling returns a Pydantic instance."""

    class Sentiment(BaseModel):
        """Sentiment of a text."""

        label: str = Field(description="positive, negative, or neutral")
        score: float = Field(description="Confidence between 0 and 1")

    result = (
        _chat(max_tokens=128)
        .with_structured_output(Sentiment, method="function_calling")
        .invoke("I love this!")
    )
    assert isinstance(result, Sentiment)
    assert result.label in {"positive", "negative", "neutral"}


@pytest.mark.xfail(reason="json_mode requires explicit schema instructions in prompt")
def test_structured_output_json_mode() -> None:
    """with_structured_output via json_mode returns a dict."""
    result = (
        _chat(max_tokens=128)
        .with_structured_output({"type": "object"}, method="json_mode")
        .invoke('Return JSON: {"city": "Paris"}')
    )
    assert isinstance(result, dict)
