"""Unit tests for ChatVLLM."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_vllm._compat import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _create_usage_metadata,
)
from langchain_vllm.chat_models import ChatVLLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class TestChatVLLM(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatVLLM]:
        return ChatVLLM

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}


def test_initialization_defaults() -> None:
    """Default `base_url` and placeholder `api_key` should be set without input."""
    llm = ChatVLLM(model=MODEL_NAME)
    assert llm.base_url == "http://localhost:8000/v1"
    assert isinstance(llm.api_key, SecretStr)
    assert llm.api_key.get_secret_value() == "EMPTY"
    assert llm._llm_type == "chat-vllm"
    assert llm._client is not None
    assert llm._async_client is not None


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`VLLM_API_KEY` should populate `api_key` when set."""
    monkeypatch.setenv("VLLM_API_KEY", "secret-token")
    llm = ChatVLLM(model=MODEL_NAME)
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "secret-token"


def test_model_name_alias() -> None:
    """`model_name` should be accepted as an alias for `model`."""
    llm = ChatVLLM(model_name="aliased")  # type: ignore[call-arg]
    assert llm.model == "aliased"


def test_unknown_kwargs_routed_to_model_kwargs() -> None:
    """vLLM-specific kwargs not declared as fields flow into `model_kwargs`."""
    llm = ChatVLLM(model=MODEL_NAME, top_k=5, repetition_penalty=1.1)  # type: ignore[call-arg]
    assert llm.model_kwargs == {"top_k": 5, "repetition_penalty": 1.1}


def test_default_params() -> None:
    """Only set params are included in the payload defaults."""
    llm = ChatVLLM(model=MODEL_NAME, temperature=0.5, max_tokens=100)
    params = llm._default_params
    assert params["model"] == MODEL_NAME
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 100
    assert "top_p" not in params


def test_ls_params() -> None:
    """LangSmith params should carry the vllm provider and model info."""
    llm = ChatVLLM(model=MODEL_NAME, temperature=0.3, max_tokens=50)
    ls_params = llm._get_ls_params()
    assert ls_params["ls_provider"] == "vllm"
    assert ls_params["ls_model_name"] == MODEL_NAME
    assert ls_params["ls_model_type"] == "chat"
    assert ls_params["ls_temperature"] == 0.3
    assert ls_params["ls_max_tokens"] == 50


def test_metadata_has_version() -> None:
    """Package version should be recorded in metadata."""
    llm = ChatVLLM(model=MODEL_NAME)
    versions = llm.metadata.get("lc_versions") if llm.metadata else None
    assert versions is not None
    assert "langchain-vllm" in versions


def test_bind_tools_normalizes_tool_choice() -> None:
    """`tool_choice='any'` should be normalized to `'required'`."""

    def my_tool(x: int) -> int:
        """Double x."""
        return x * 2

    llm = ChatVLLM(model=MODEL_NAME)
    bound = llm.bind_tools([my_tool], tool_choice="any")
    assert bound.kwargs["tool_choice"] == "required"  # type: ignore[attr-defined]


def test_bind_tools_named_tool_choice() -> None:
    """A tool name passed as `tool_choice` becomes an OpenAI function dict."""

    def my_tool(x: int) -> int:
        """Double x."""
        return x * 2

    llm = ChatVLLM(model=MODEL_NAME)
    bound = llm.bind_tools([my_tool], tool_choice="my_tool")
    assert bound.kwargs["tool_choice"] == {  # type: ignore[attr-defined]
        "type": "function",
        "function": {"name": "my_tool"},
    }


def test_convert_message_to_dict_roles() -> None:
    """Messages convert to their OpenAI role dicts."""
    assert _convert_message_to_dict(HumanMessage("hi"))["role"] == "user"
    assert _convert_message_to_dict(SystemMessage("sys"))["role"] == "system"
    assert _convert_message_to_dict(AIMessage("out"))["role"] == "assistant"
    tool_dict = _convert_message_to_dict(
        ToolMessage(content="result", tool_call_id="call_1")
    )
    assert tool_dict["role"] == "tool"
    assert tool_dict["tool_call_id"] == "call_1"


def test_convert_dict_to_message_with_tool_calls() -> None:
    """Assistant dicts with tool calls parse into an AIMessage with tool_calls."""
    message = _convert_dict_to_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                }
            ],
        }
    )
    assert isinstance(message, AIMessage)
    assert message.tool_calls[0]["name"] == "get_weather"
    assert message.tool_calls[0]["args"] == {"city": "SF"}


def test_convert_dict_to_message_reasoning_content() -> None:
    """Older vLLM builds emit `reasoning_content`; it lands in additional_kwargs."""
    message = _convert_dict_to_message(
        {"role": "assistant", "content": "4", "reasoning_content": "2+2 is 4"}
    )
    assert isinstance(message, AIMessage)
    assert message.additional_kwargs["reasoning_content"] == "2+2 is 4"


def test_convert_dict_to_message_reasoning_field() -> None:
    """Newer vLLM builds emit `reasoning`; it normalizes to `reasoning_content`.

    Regression test for the reasoning-token loss reported in
    `langchain-ai/langchain#36809`, where vLLM's migration from
    `reasoning_content` to `reasoning` caused thinking tokens to be dropped.
    """
    message = _convert_dict_to_message(
        {"role": "assistant", "content": "4", "reasoning": "2+2 is 4"}
    )
    assert isinstance(message, AIMessage)
    assert message.additional_kwargs["reasoning_content"] == "2+2 is 4"


def test_convert_delta_to_message_chunk_reasoning_field() -> None:
    """Streaming deltas carrying `reasoning` are preserved (the #36809 path)."""
    chunk = _convert_delta_to_message_chunk(
        {"role": "assistant", "content": "", "reasoning": "thinking..."},
        AIMessageChunk,
    )
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["reasoning_content"] == "thinking..."


def test_convert_delta_to_message_chunk_reasoning_content() -> None:
    """Streaming deltas carrying `reasoning_content` are preserved."""
    chunk = _convert_delta_to_message_chunk(
        {"role": "assistant", "content": "", "reasoning_content": "thinking..."},
        AIMessageChunk,
    )
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["reasoning_content"] == "thinking..."


def test_create_usage_metadata() -> None:
    """Token usage dicts convert to UsageMetadata."""
    usage = _create_usage_metadata(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["total_tokens"] == 15


@patch("langchain_vllm._utils.openai.OpenAI")
def test_generate_calls_create(mock_openai: Any) -> None:
    """`_generate` should call the client's `create` and build a ChatResult."""
    mock_client = MagicMock()
    mock_openai.return_value.chat.completions = mock_client
    mock_client.create.return_value = {
        "id": "cmpl-1",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    llm = ChatVLLM(model=MODEL_NAME)
    result = llm.invoke("Hello")
    assert result.content == "Hello there"
    mock_client.create.assert_called_once()


@patch("langchain_vllm._utils.openai.OpenAI")
def test_create_chat_result_null_choices_raises(mock_openai: Any) -> None:
    """A null `choices` value should raise an informative TypeError."""
    mock_openai.return_value.chat.completions = MagicMock()
    llm = ChatVLLM(model=MODEL_NAME)
    with pytest.raises(TypeError, match="null value for 'choices'"):
        llm._create_chat_result({"choices": None})
