"""vLLM embedding models (OpenAI-compatible server)."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.utils import secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from langchain_vllm._utils import (
    build_async_client,
    build_client_params,
    build_sync_client,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLMEmbeddings(BaseModel, Embeddings):
    """vLLM embedding model integration (OpenAI-compatible server).

    Connects to a running vLLM server's `/v1/embeddings` endpoint over HTTP. Start a
    server for an embedding model with `vllm serve <model>` and point `base_url` at
    it.

    Setup:
        ```bash
        pip install -U langchain-vllm
        vllm serve intfloat/e5-mistral-7b-instruct
        ```

    Instantiate:
        ```python
        from langchain_vllm import VLLMEmbeddings

        embed = VLLMEmbeddings(model="intfloat/e5-mistral-7b-instruct")
        ```

    Embed single text:
        ```python
        vector = embed.embed_query("The meaning of life is 42")
        print(vector[:3])
        ```

    Embed multiple texts:
        ```python
        vectors = embed.embed_documents(["Document 1...", "Document 2..."])
        print(len(vectors))
        ```
    """

    model: str = Field(alias="model_name")
    """Name of the embedding model served by the vLLM server."""

    base_url: str = DEFAULT_BASE_URL
    """Base URL of the vLLM OpenAI-compatible server (must include `/v1`)."""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("VLLM_API_KEY", default="EMPTY")
    )
    """API key (defaults to the placeholder `'EMPTY'`; not required locally)."""

    dimensions: int | None = None
    """Number of dimensions for the output embedding vectors.

    If not provided, the model's default embedding dimensionality is used. Only
    honored by models that support `Matryoshka`-style dimension reduction.
    """

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            msg = "`dimensions` must be a positive integer."
            raise ValueError(msg)
        return v

    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to the vLLM server."""

    max_retries: int = 2
    """Maximum number of retries on transient errors."""

    default_headers: dict[str, str] | None = None
    """Headers to send with every request."""

    default_query: dict[str, Any] | None = None
    """Query params to send with every request."""

    extra_body: dict[str, Any] | None = None
    """Extra JSON body params forwarded verbatim to the server."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional params passed through to the embeddings create call."""

    http_client: Any | None = None
    """Optional custom `httpx.Client` for sync requests."""

    http_async_client: Any | None = None
    """Optional custom `httpx.AsyncClient` for async requests."""

    _client: Any = PrivateAttr(default=None)
    """The underlying sync `embeddings` resource."""

    _async_client: Any = PrivateAttr(default=None)
    """The underlying async `embeddings` resource."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Construct the `openai` sync/async clients."""
        api_key = self.api_key.get_secret_value() if self.api_key else "EMPTY"
        params = build_client_params(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
        )
        if self._client is None:
            self._client = build_sync_client(params, self.http_client).embeddings
        if self._async_client is None:
            self._async_client = build_async_client(
                params, self.http_async_client
            ).embeddings
        return self

    def _create_params(self, texts: list[str]) -> dict[str, Any]:
        """Assemble params for the embeddings create call."""
        params: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            **self.model_kwargs,
        }
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        if self.extra_body is not None:
            params["extra_body"] = self.extra_body
        return params

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        """Extract embedding vectors, ordered by their `index`, from a response."""
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        data = sorted(response_dict["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs. Sends raw text (no client-side tokenization)."""
        if not self._client:
            msg = (
                "vLLM sync client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        response = self._client.create(**self._create_params(texts))
        return self._extract_embeddings(response)

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs. Sends raw text (no client-side tokenization)."""
        if not self._async_client:
            msg = (
                "vLLM async client is not initialized. "
                "Make sure the model was properly constructed."
            )
            raise RuntimeError(msg)
        response = await self._async_client.create(**self._create_params(texts))
        return self._extract_embeddings(response)

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
