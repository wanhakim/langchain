"""Unit tests for VLLMEmbeddings."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_vllm.embeddings import VLLMEmbeddings

MODEL_NAME = "intfloat/e5-mistral-7b-instruct"


def test_initialization_defaults() -> None:
    embed = VLLMEmbeddings(model=MODEL_NAME)
    assert embed.base_url == "http://localhost:8000/v1"
    assert embed.api_key is not None
    assert embed.api_key.get_secret_value() == "EMPTY"
    assert embed._client is not None
    assert embed._async_client is not None


def test_dimensions_validation() -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        VLLMEmbeddings(model=MODEL_NAME, dimensions=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        VLLMEmbeddings(model=MODEL_NAME, dimensions=-1)


@patch("langchain_vllm._utils.openai.OpenAI")
def test_embed_documents_orders_by_index(mock_openai: Any) -> None:
    """Embeddings returned out of order should be re-sorted by `index`."""
    mock_client = MagicMock()
    mock_openai.return_value.embeddings = mock_client
    mock_client.create.return_value = {
        "data": [
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
        ]
    }
    embed = VLLMEmbeddings(model=MODEL_NAME)
    result = embed.embed_documents(["a", "b"])
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert mock_client.create.call_args.kwargs["input"] == ["a", "b"]
    assert mock_client.create.call_args.kwargs["model"] == MODEL_NAME


@patch("langchain_vllm._utils.openai.OpenAI")
def test_embed_documents_passes_dimensions(mock_openai: Any) -> None:
    mock_client = MagicMock()
    mock_openai.return_value.embeddings = mock_client
    mock_client.create.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
    embed = VLLMEmbeddings(model=MODEL_NAME, dimensions=512)
    embed.embed_documents(["text"])
    assert mock_client.create.call_args.kwargs["dimensions"] == 512


@patch("langchain_vllm._utils.openai.OpenAI")
def test_embed_query(mock_openai: Any) -> None:
    mock_client = MagicMock()
    mock_openai.return_value.embeddings = mock_client
    mock_client.create.return_value = {"data": [{"index": 0, "embedding": [0.7, 0.8]}]}
    embed = VLLMEmbeddings(model=MODEL_NAME)
    assert embed.embed_query("hi") == [0.7, 0.8]


def test_embed_documents_raises_when_client_none() -> None:
    embed = VLLMEmbeddings(model=MODEL_NAME)
    embed._client = None
    with pytest.raises(RuntimeError, match="sync client is not initialized"):
        embed.embed_documents(["test"])


async def test_aembed_documents_orders_by_index() -> None:
    with patch("langchain_vllm._utils.openai.AsyncOpenAI") as mock_async_openai:
        mock_client = MagicMock()
        mock_async_openai.return_value.embeddings = mock_client
        mock_client.create = AsyncMock(
            return_value={
                "data": [
                    {"index": 1, "embedding": [0.4]},
                    {"index": 0, "embedding": [0.1]},
                ]
            }
        )
        embed = VLLMEmbeddings(model=MODEL_NAME)
        result = await embed.aembed_documents(["a", "b"])
        assert result == [[0.1], [0.4]]


async def test_aembed_documents_raises_when_client_none() -> None:
    embed = VLLMEmbeddings(model=MODEL_NAME)
    embed._async_client = None
    with pytest.raises(RuntimeError, match="async client is not initialized"):
        await embed.aembed_documents(["test"])
