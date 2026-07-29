"""Unit tests for VLLM."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_vllm.llms import VLLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def test_initialization_defaults() -> None:
    llm = VLLM(model=MODEL_NAME)
    assert llm.base_url == "http://localhost:8000/v1"
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "EMPTY"
    assert llm._llm_type == "vllm-llm"
    assert llm._client is not None
    assert llm._async_client is not None


def test_model_name_alias() -> None:
    llm = VLLM(model_name="aliased")  # type: ignore[call-arg]
    assert llm.model == "aliased"


def test_default_params() -> None:
    llm = VLLM(model=MODEL_NAME, temperature=0.2, max_tokens=64)
    params = llm._default_params
    assert params["model"] == MODEL_NAME
    assert params["temperature"] == 0.2
    assert params["max_tokens"] == 64
    assert "top_p" not in params


def test_ls_params() -> None:
    llm = VLLM(model=MODEL_NAME, max_tokens=32)
    ls_params = llm._get_ls_params()
    assert ls_params["ls_provider"] == "vllm"
    assert ls_params["ls_max_tokens"] == 32


@patch("langchain_vllm._utils.openai.OpenAI")
def test_generate_maps_choices_to_prompts(mock_openai: Any) -> None:
    """Each prompt should map to its own generation via the choice index."""
    mock_client = MagicMock()
    mock_openai.return_value.completions = mock_client
    mock_client.create.return_value = {
        "model": MODEL_NAME,
        "choices": [
            {"index": 0, "text": "first", "finish_reason": "stop"},
            {"index": 1, "text": "second", "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }
    llm = VLLM(model=MODEL_NAME)
    result = llm.generate(["prompt a", "prompt b"])
    assert result.generations[0][0].text == "first"
    assert result.generations[1][0].text == "second"
    # A single call handles the whole prompt list.
    mock_client.create.assert_called_once()
    assert mock_client.create.call_args.kwargs["prompt"] == ["prompt a", "prompt b"]
