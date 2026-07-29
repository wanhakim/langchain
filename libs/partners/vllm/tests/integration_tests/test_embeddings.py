"""Integration tests for VLLMEmbeddings."""

import os

import pytest
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_vllm.embeddings import VLLMEmbeddings

MODEL = os.environ.get("VLLM_TEST_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
BASE_URL = os.environ.get("VLLM_TEST_EMBED_BASE_URL") or ""

# Skip the entire module when no embeddings server URL is configured.
pytestmark = pytest.mark.skipif(
    not BASE_URL,
    reason="VLLM_TEST_EMBED_BASE_URL not set — start an embeddings server first",
)


class TestVLLMEmbeddingsStandard(EmbeddingsIntegrationTests):
    """Standard LangChain embeddings integration test suite."""

    @property
    def embeddings_class(self) -> type[VLLMEmbeddings]:
        return VLLMEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": MODEL, "base_url": BASE_URL or "http://localhost:8001/v1"}


# ---------------------------------------------------------------------------
# Custom integration tests
# ---------------------------------------------------------------------------


def _embeddings() -> VLLMEmbeddings:
    return VLLMEmbeddings(model=MODEL, base_url=BASE_URL or "http://localhost:8001/v1")


def test_embed_query() -> None:
    """embed_query returns a non-empty float vector."""
    vector = _embeddings().embed_query("The meaning of life is 42")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


def test_embed_documents() -> None:
    """embed_documents returns one vector per document, sorted by index."""
    vectors = _embeddings().embed_documents(["Document 1", "Document 2"])
    assert len(vectors) == 2
    assert all(len(v) > 0 for v in vectors)


def test_embed_query_and_documents_same_dim() -> None:
    """Query and document embeddings have the same dimensionality."""
    emb = _embeddings()
    query_vec = emb.embed_query("hello")
    doc_vecs = emb.embed_documents(["hello", "world"])
    assert len(query_vec) == len(doc_vecs[0]) == len(doc_vecs[1])


async def test_aembed_query() -> None:
    """Async embed_query returns a non-empty float vector."""
    vector = await _embeddings().aembed_query("Async embedding test")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


async def test_aembed_documents() -> None:
    """Async embed_documents returns one vector per document."""
    vectors = await _embeddings().aembed_documents(["doc one", "doc two"])
    assert len(vectors) == 2
    assert all(len(v) > 0 for v in vectors)
