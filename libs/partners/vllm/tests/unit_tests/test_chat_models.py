"""Unit tests for ChatVLLM."""

import enum
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
from langchain_core.output_parsers import StrOutputParser
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel, SecretStr

from langchain_vllm._compat import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _create_usage_metadata,
)
from langchain_vllm.chat_models import (
    ChatVLLM,
    _deep_merge_extra_body,
    _extract_guided_choices,
)

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


# ---------------------------------------------------------------------------
# _extract_guided_choices helpers
# ---------------------------------------------------------------------------


class _Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


def test_extract_guided_choices_list() -> None:
    """A plain list of strings is returned verbatim with no enum map."""
    wire, enum_map = _extract_guided_choices(["a", "b", "c"])
    assert wire == ["a", "b", "c"]
    assert enum_map is None


def test_extract_guided_choices_enum() -> None:
    """An Enum subclass produces wire values from .value and a reverse map."""
    wire, enum_map = _extract_guided_choices(_Color)
    assert wire == ["red", "green", "blue"]
    assert enum_map is not None
    assert enum_map["red"] is _Color.RED
    assert enum_map["green"] is _Color.GREEN


def test_extract_guided_choices_rejects_bare_string() -> None:
    """A bare string should raise TypeError with a helpful message."""
    with pytest.raises(TypeError, match="list of strings"):
        _extract_guided_choices("yes")  # type: ignore[arg-type]


def test_extract_guided_choices_rejects_empty_list() -> None:
    """An empty list should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        _extract_guided_choices([])


def test_extract_guided_choices_rejects_empty_enum() -> None:
    """An Enum with no members should raise ValueError."""

    class Empty(enum.Enum):
        pass

    with pytest.raises(ValueError, match="at least one member"):
        _extract_guided_choices(Empty)


# ---------------------------------------------------------------------------
# _deep_merge_extra_body
# ---------------------------------------------------------------------------


def test_deep_merge_extra_body_no_overlap() -> None:
    """Keys that don't overlap are merged without loss."""
    result = _deep_merge_extra_body(
        {"top_k": 5}, {"structured_outputs": {"choice": ["a"]}}
    )
    assert result == {"top_k": 5, "structured_outputs": {"choice": ["a"]}}


def test_deep_merge_extra_body_structured_outputs_union() -> None:
    """Nested structured_outputs sub-keys are unioned."""
    base = {"structured_outputs": {"foo": 1}}
    overlay = {"structured_outputs": {"choice": ["a", "b"]}}
    result = _deep_merge_extra_body(base, overlay)
    assert result["structured_outputs"] == {"foo": 1, "choice": ["a", "b"]}


def test_deep_merge_extra_body_overlay_wins_scalar() -> None:
    """For scalar keys outside structured_outputs, overlay wins."""
    result = _deep_merge_extra_body({"x": 1}, {"x": 2})
    assert result["x"] == 2


# ---------------------------------------------------------------------------
# with_guided_choice — payload shape
# ---------------------------------------------------------------------------


def test_guided_choice_list_extra_body() -> None:
    """with_guided_choice with a list produces the correct structured_outputs fragment.
    """
    llm = ChatVLLM(model=MODEL_NAME)
    bound = llm.with_guided_choice(["positive", "negative", "neutral"])
    # The bound runnable's LLM step carries the extra_body kwarg
    llm_step = bound.first  # type: ignore[attr-defined]
    assert llm_step.kwargs["extra_body"] == {
        "structured_outputs": {"choice": ["positive", "negative", "neutral"]}
    }


def test_guided_choice_enum_extra_body() -> None:
    """with_guided_choice with an Enum uses .value strings in order."""
    llm = ChatVLLM(model=MODEL_NAME)
    bound = llm.with_guided_choice(_Color)
    llm_step = bound.first  # type: ignore[attr-defined]
    assert llm_step.kwargs["extra_body"] == {
        "structured_outputs": {"choice": ["red", "green", "blue"]}
    }


def test_guided_regex_extra_body() -> None:
    """with_guided_regex lands the pattern under structured_outputs.regex."""
    llm = ChatVLLM(model=MODEL_NAME)
    bound = llm.with_guided_regex(r"\d{3}-\d{4}")
    llm_step = bound.first  # type: ignore[attr-defined]
    assert llm_step.kwargs["extra_body"] == {
        "structured_outputs": {"regex": r"\d{3}-\d{4}"}
    }


def test_guided_grammar_extra_body() -> None:
    """with_guided_grammar lands the grammar under structured_outputs.grammar."""
    llm = ChatVLLM(model=MODEL_NAME)
    grammar = 'root ::= "yes" | "no"'
    bound = llm.with_guided_grammar(grammar)
    llm_step = bound.first  # type: ignore[attr-defined]
    assert llm_step.kwargs["extra_body"] == {"structured_outputs": {"grammar": grammar}}


def test_guided_json_extra_body() -> None:
    """with_structured_output(method='guided_json') uses structured_outputs.json."""

    class City(BaseModel):
        name: str
        country: str

    llm = ChatVLLM(model=MODEL_NAME)
    bound = llm.with_structured_output(City, method="guided_json")
    llm_step = bound.first  # type: ignore[attr-defined]
    extra_body = llm_step.kwargs["extra_body"]
    assert "structured_outputs" in extra_body
    assert "json" in extra_body["structured_outputs"]
    assert extra_body["structured_outputs"]["json"]["title"] == "City"


# ---------------------------------------------------------------------------
# Legacy wire format
# ---------------------------------------------------------------------------


def test_guided_choice_legacy_format() -> None:
    """structured_outputs_format='legacy' emits top-level guided_choice."""
    llm = ChatVLLM(model=MODEL_NAME, structured_outputs_format="legacy")
    bound = llm.with_guided_choice(["a", "b"])
    llm_step = bound.first  # type: ignore[attr-defined]
    assert llm_step.kwargs["extra_body"] == {"guided_choice": ["a", "b"]}


def test_guided_regex_legacy_format() -> None:
    """structured_outputs_format='legacy' emits top-level guided_regex."""
    llm = ChatVLLM(model=MODEL_NAME, structured_outputs_format="legacy")
    bound = llm.with_guided_regex(r"\d+")
    llm_step = bound.first  # type: ignore[attr-defined]
    assert llm_step.kwargs["extra_body"] == {"guided_regex": r"\d+"}


def test_guided_grammar_legacy_format() -> None:
    """structured_outputs_format='legacy' emits top-level guided_grammar."""
    llm = ChatVLLM(model=MODEL_NAME, structured_outputs_format="legacy")
    bound = llm.with_guided_grammar('root ::= "x"')
    llm_step = bound.first  # type: ignore[attr-defined]
    assert llm_step.kwargs["extra_body"] == {"guided_grammar": 'root ::= "x"'}


def test_guided_json_legacy_format() -> None:
    """with_structured_output(method='guided_json', legacy) emits guided_json."""

    class Tag(BaseModel):
        label: str

    llm = ChatVLLM(model=MODEL_NAME, structured_outputs_format="legacy")
    bound = llm.with_structured_output(Tag, method="guided_json")
    llm_step = bound.first  # type: ignore[attr-defined]
    extra_body = llm_step.kwargs["extra_body"]
    assert "guided_json" in extra_body
    assert extra_body["guided_json"]["title"] == "Tag"


# ---------------------------------------------------------------------------
# guided_decoding_backend propagation
# ---------------------------------------------------------------------------


def test_guided_decoding_backend_added() -> None:
    """guided_decoding_backend is a top-level extra_body key."""
    llm = ChatVLLM(model=MODEL_NAME)
    bound = llm.with_guided_choice(["yes", "no"], guided_decoding_backend="xgrammar")
    llm_step = bound.first  # type: ignore[attr-defined]
    eb = llm_step.kwargs["extra_body"]
    assert eb["guided_decoding_backend"] == "xgrammar"
    assert "structured_outputs" in eb


def test_guided_decoding_backend_legacy() -> None:
    """guided_decoding_backend is top-level even in legacy format."""
    llm = ChatVLLM(model=MODEL_NAME, structured_outputs_format="legacy")
    bound = llm.with_guided_regex(r"\d+", guided_decoding_backend="outlines")
    llm_step = bound.first  # type: ignore[attr-defined]
    eb = llm_step.kwargs["extra_body"]
    assert eb["guided_decoding_backend"] == "outlines"
    assert "guided_regex" in eb


# ---------------------------------------------------------------------------
# Merge: instance extra_body + guided fragment
# ---------------------------------------------------------------------------


@patch("langchain_vllm._utils.openai.OpenAI")
def test_guided_choice_merges_with_instance_extra_body(mock_openai: Any) -> None:
    """Instance-level extra_body is deep-merged with the guided fragment."""
    mock_client = MagicMock()
    mock_openai.return_value.chat.completions = mock_client
    mock_client.create.return_value = {
        "id": "x",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "yes"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    llm = ChatVLLM(model=MODEL_NAME, extra_body={"top_k": 5})
    llm.with_guided_choice(["yes", "no"]).invoke("test")
    call_kwargs = mock_client.create.call_args.kwargs
    eb = call_kwargs["extra_body"]
    # Both keys must survive in the merged payload
    assert eb["top_k"] == 5
    assert eb["structured_outputs"]["choice"] == ["yes", "no"]


@patch("langchain_vllm._utils.openai.OpenAI")
def test_guided_no_clobber_user_structured_outputs(mock_openai: Any) -> None:
    """User's existing structured_outputs sub-keys are preserved during merge."""
    mock_client = MagicMock()
    mock_openai.return_value.chat.completions = mock_client
    mock_client.create.return_value = {
        "id": "x",
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "yes"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    llm = ChatVLLM(model=MODEL_NAME, extra_body={"structured_outputs": {"foo": 1}})
    llm.with_guided_choice(["yes", "no"]).invoke("test")
    call_kwargs = mock_client.create.call_args.kwargs
    eb = call_kwargs["extra_body"]
    assert eb["structured_outputs"]["foo"] == 1
    assert eb["structured_outputs"]["choice"] == ["yes", "no"]


# ---------------------------------------------------------------------------
# Parser behaviour
# ---------------------------------------------------------------------------


def test_guided_choice_enum_maps_output() -> None:
    """The Enum mapping runnable converts the wire string to the Enum member."""
    llm = ChatVLLM(model=MODEL_NAME)
    runnable = llm.with_guided_choice(_Color)
    # Extract and exercise just the parser chain (second step after the LLM)
    parser_chain = runnable.last  # type: ignore[attr-defined]
    result = parser_chain.invoke("green")
    assert result is _Color.GREEN


def test_guided_choice_enum_strips_whitespace() -> None:
    """Trailing whitespace/newline is stripped before the Enum lookup."""
    llm = ChatVLLM(model=MODEL_NAME)
    parser_chain = llm.with_guided_choice(_Color).last  # type: ignore[attr-defined]
    assert parser_chain.invoke("blue\n") is _Color.BLUE


def test_guided_regex_returns_str_parser() -> None:
    """with_guided_regex uses StrOutputParser as its tail parser."""
    llm = ChatVLLM(model=MODEL_NAME)
    runnable = llm.with_guided_regex(r"\d+")
    parser = runnable.last  # type: ignore[attr-defined]
    assert isinstance(parser, StrOutputParser)


def test_guided_choice_rejects_bare_string() -> None:
    """with_guided_choice raises TypeError when passed a bare string."""
    llm = ChatVLLM(model=MODEL_NAME)
    with pytest.raises(TypeError, match="list of strings"):
        llm.with_guided_choice("yes")  # type: ignore[arg-type]


def test_guided_choice_rejects_empty() -> None:
    """with_guided_choice raises ValueError for an empty list."""
    llm = ChatVLLM(model=MODEL_NAME)
    with pytest.raises(ValueError, match="empty"):
        llm.with_guided_choice([])


def test_guided_regex_rejects_empty() -> None:
    """with_guided_regex raises ValueError for an empty pattern."""
    llm = ChatVLLM(model=MODEL_NAME)
    with pytest.raises(ValueError, match="non-empty"):
        llm.with_guided_regex("")


def test_guided_grammar_rejects_empty() -> None:
    """with_guided_grammar raises ValueError for an empty grammar string."""
    llm = ChatVLLM(model=MODEL_NAME)
    with pytest.raises(ValueError, match="non-empty"):
        llm.with_guided_grammar("")
