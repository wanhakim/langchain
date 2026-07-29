"""vLLM large language models (OpenAI-compatible server)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM, LangSmithParams
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import secret_from_env
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self

from langchain_vllm._utils import (
    build_async_client,
    build_client_params,
    build_sync_client,
)
from langchain_vllm._version import __version__

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLM(BaseLLM):
    """vLLM completion model integration (OpenAI-compatible server).

    Connects to a running vLLM server's `/v1/completions` endpoint over HTTP. Start
    a server with `vllm serve <model>` and point `base_url` at it. This does *not*
    load a model in-process.

    Setup:
        ```bash
        pip install -U langchain-vllm
        vllm serve Qwen/Qwen2.5-1.5B-Instruct
        ```

    Key init args — generation params:
        model: str
            Name of the model served by vLLM.
        temperature: float | None
            Sampling temperature.
        max_tokens: int | None
            Maximum number of tokens to generate.

    Key init args — client params:
        base_url: str
            Base URL of the vLLM server (must include `/v1`).
        api_key: SecretStr
            API key (defaults to placeholder `'EMPTY'`; not required locally).

    Instantiate:
        ```python
        from langchain_vllm import VLLM

        llm = VLLM(model="Qwen/Qwen2.5-1.5B-Instruct", temperature=0)
        llm.invoke("The meaning of life is ")
        ```
    """

    model: str = Field(alias="model_name")
    """Name of the model served by the vLLM server."""

    base_url: str = DEFAULT_BASE_URL
    """Base URL of the vLLM OpenAI-compatible server (must include `/v1`)."""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("VLLM_API_KEY", default="EMPTY")
    )
    """API key (defaults to the placeholder `'EMPTY'`; not required locally)."""

    temperature: float | None = None
    """Sampling temperature."""

    max_tokens: int | None = None
    """Maximum number of tokens to generate."""

    top_p: float | None = None
    """Nucleus sampling probability mass."""

    frequency_penalty: float | None = None
    """Penalize new tokens based on their existing frequency."""

    presence_penalty: float | None = None
    """Penalize new tokens based on whether they appear so far."""

    seed: int | None = None
    """Seed for deterministic sampling."""

    stop: list[str] | None = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""

    logprobs: int | None = None
    """Number of log probabilities to return per token (completions endpoint)."""

    n: int | None = None
    """Number of completions to generate for each prompt."""

    best_of: int | None = None
    """Generate `best_of` completions server-side and return the best `n`."""

    streaming: bool = False
    """Whether to stream the results or not."""

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
    """Extra JSON body params forwarded verbatim (e.g. `top_k`, `guided_json`)."""

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional params passed through to the create call."""

    http_client: Any | None = None
    """Optional custom `httpx.Client` for sync requests."""

    http_async_client: Any | None = None
    """Optional custom `httpx.AsyncClient` for async requests."""

    _client: Any = PrivateAttr(default=None)
    """The underlying sync `completions` resource."""

    _async_client: Any = PrivateAttr(default=None)
    """The underlying async `completions` resource."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Construct the `openai` sync/async clients and record the version."""
        self._add_version("langchain-vllm", __version__)
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
            self._client = build_sync_client(params, self.http_client).completions
        if self._async_client is None:
            self._async_client = build_async_client(
                params, self.http_async_client
            ).completions
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "vllm-llm"

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default params for calling the vLLM completions endpoint."""
        exclude_if_none = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "n": self.n,
            "best_of": self.best_of,
            "stop": self.stop or None,
            "extra_body": self.extra_body,
        }
        return {
            "model": self.model,
            **{k: v for k, v in exclude_if_none.items() if v is not None},
            **self.model_kwargs,
        }

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "vllm"
        if max_tokens := kwargs.get("max_tokens", self.max_tokens):
            params["ls_max_tokens"] = max_tokens
        return params

    def _invocation_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Assemble the payload for a completions call (minus `prompt`)."""
        if self.stop is not None and stop is not None:
            msg = "`stop` found in both the input and default params."
            raise ValueError(msg)
        params = {**self._default_params, **kwargs}
        if stop is not None:
            params["stop"] = stop
        return params

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResult:
        params = self._invocation_params(stop=stop, **kwargs)
        # The completions endpoint accepts a list of prompts and returns one choice
        # per prompt, ordered by an `index` field.
        response = self._client.create(prompt=prompts, **params)
        return self._create_llm_result(response, len(prompts))

    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResult:
        params = self._invocation_params(stop=stop, **kwargs)
        response = await self._async_client.create(prompt=prompts, **params)
        return self._create_llm_result(response, len(prompts))

    def _create_llm_result(self, response: Any, num_prompts: int) -> LLMResult:
        """Map a completions response to an `LLMResult`."""
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        choices = response_dict.get("choices") or []
        # Group choices back to their prompt via `index // n`.
        n = self.n or 1
        generations: list[list[Generation]] = [[] for _ in range(num_prompts)]
        for choice in choices:
            prompt_index = choice.get("index", 0) // n
            if prompt_index >= num_prompts:
                prompt_index = 0
            generations[prompt_index].append(
                Generation(
                    text=choice.get("text", ""),
                    generation_info={
                        "finish_reason": choice.get("finish_reason"),
                        "logprobs": choice.get("logprobs"),
                    },
                )
            )
        llm_output = {
            "token_usage": response_dict.get("usage"),
            "model_name": response_dict.get("model", self.model),
        }
        return LLMResult(generations=generations, llm_output=llm_output)

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = self._invocation_params(stop=stop, **kwargs)
        params["stream"] = True
        for raw_chunk in self._client.create(prompt=prompt, **params):
            chunk = raw_chunk if isinstance(raw_chunk, dict) else raw_chunk.model_dump()
            gen_chunk = _completion_chunk_to_generation_chunk(chunk)
            if gen_chunk is None:
                continue
            if run_manager:
                run_manager.on_llm_new_token(gen_chunk.text, chunk=gen_chunk)
            yield gen_chunk

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = self._invocation_params(stop=stop, **kwargs)
        params["stream"] = True
        response = await self._async_client.create(prompt=prompt, **params)
        async for raw_chunk in response:
            chunk = raw_chunk if isinstance(raw_chunk, dict) else raw_chunk.model_dump()
            gen_chunk = _completion_chunk_to_generation_chunk(chunk)
            if gen_chunk is None:
                continue
            if run_manager:
                await run_manager.on_llm_new_token(gen_chunk.text, chunk=gen_chunk)
            yield gen_chunk


def _completion_chunk_to_generation_chunk(chunk: dict) -> GenerationChunk | None:
    """Convert a streamed completions chunk dict to a `GenerationChunk`."""
    choices = chunk.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    generation_info = {}
    if finish_reason := choice.get("finish_reason"):
        generation_info["finish_reason"] = finish_reason
    if logprobs := choice.get("logprobs"):
        generation_info["logprobs"] = logprobs
    return GenerationChunk(
        text=choice.get("text", ""),
        generation_info=generation_info or None,
    )
