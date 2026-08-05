"""vLLM chat models (OpenAI-compatible server)."""

from __future__ import annotations

import enum
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from operator import itemgetter
from typing import Any, Literal, TypeVar, cast, get_args, get_origin, overload

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
    StrOutputParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self, is_typeddict

from langchain_vllm._compat import (
    _convert_delta_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _create_usage_metadata,
)
from langchain_vllm._utils import (
    build_async_client,
    build_client_params,
    build_sync_client,
)
from langchain_vllm._version import __version__

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"


EnumT = TypeVar("EnumT", bound=enum.Enum)


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _extract_guided_choices(
    choices: Sequence[str] | type[enum.Enum],
) -> tuple[list[str], dict[str, Any] | None]:
    """Return ``(wire_choices, enum_map_or_none)`` for a guided-choice constraint.

    Args:
        choices: A non-empty list/tuple of strings, a ``Literal[...]`` alias, or
            an ``enum.Enum`` subclass whose ``.value`` attributes are used as the
            wire labels.

    Returns:
        A pair ``(wire_choices, enum_map)`` where ``wire_choices`` is the list of
        strings to send to the server and ``enum_map`` is a ``{str: EnumMember}``
        reverse-lookup dict (only set when ``choices`` is an Enum class, so the
        returned string can be mapped back to the original member).

    Raises:
        ValueError: If ``choices`` is a bare string, empty, or an unsupported type.
    """
    if isinstance(choices, str):
        msg = (
            "Pass a list of strings, not a single string. "
            "Example: with_guided_choice(['yes', 'no'])"
        )
        raise TypeError(msg)

    # Literal[...] alias — detect via typing internals
    _origin = get_origin(choices)
    _is_literal = _origin is Literal or (
        hasattr(choices, "__origin__")
        and str(getattr(choices, "__origin__", "")) == "Literal"
    )
    if _is_literal:
        args = get_args(choices)
        if not args:
            msg = "choices must not be empty."
            raise ValueError(msg)
        return [str(a) for a in args], None

    # enum.Enum subclass
    if isinstance(choices, type) and issubclass(choices, enum.Enum):
        members = list(choices)
        if not members:
            msg = "choices Enum must have at least one member."
            raise ValueError(msg)
        wire = [str(m.value) for m in members]
        enum_map: dict[str, Any] = {str(m.value): m for m in members}
        return wire, enum_map

    # Sequence[str]
    wire_list = list(choices)  # type: ignore[arg-type]
    if not wire_list:
        msg = "choices must not be empty."
        raise ValueError(msg)
    return wire_list, None


def _deep_merge_extra_body(
    base: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    """Merge two ``extra_body`` dicts, deep-merging the ``structured_outputs`` sub-dict.

    Args:
        base: The base dict (typically from ``self.extra_body``).
        overlay: The overlay dict (typically a guided-decoding fragment).

    Returns:
        A new dict with the ``structured_outputs`` key deep-merged (union of sub-keys,
        overlay wins on conflicts) and all other keys shallow-merged (overlay wins).
    """
    merged = {**base, **overlay}
    if "structured_outputs" in base and "structured_outputs" in overlay:
        merged["structured_outputs"] = {
            **base["structured_outputs"],
            **overlay["structured_outputs"],
        }
    return merged


class ChatVLLM(BaseChatModel):
    r"""vLLM chat model integration.

    Connects to a running vLLM
    [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
    over HTTP. This does *not* load a model in-process; start a server first with
    `vllm serve <model>` and point `base_url` at it.

    Setup:
        Install `langchain-vllm` and start a vLLM server:

        ```bash
        pip install -U langchain-vllm
        vllm serve Qwen/Qwen2.5-1.5B-Instruct
        ```

        No API key is required for a local server.

    Key init args — completion params:
        model: str
            Name of the model served by vLLM (the `--served-model-name`, which
            defaults to the model's Hugging Face repo id).
        temperature: float | None
            Sampling temperature.
        max_tokens: int | None
            Maximum number of tokens to generate.

    Key init args — client params:
        base_url: str
            Base URL of the vLLM OpenAI-compatible server. Must include the `/v1`
            suffix. Defaults to `'http://localhost:8000/v1'`.
        api_key: SecretStr
            API key. Not required for a local server; defaults to the placeholder
            `'EMPTY'`. Set only if the server was launched with
            `vllm serve --api-key <token>`.

    Instantiate:
        ```python
        from langchain_vllm import ChatVLLM

        llm = ChatVLLM(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            temperature=0,
            # base_url="http://localhost:8000/v1",
        )
        ```

    Invoke:
        ```python
        llm.invoke("Hello!")
        ```

    Stream:
        ```python
        for chunk in llm.stream("Hello!"):
            print(chunk.text, end="")
        ```

    Guided decoding:
        vLLM supports **constrained / guided decoding** — forcing the model to emit
        output that matches a fixed set of labels, a regex pattern, or a grammar.
        This has **no equivalent in the OpenAI API** and is the primary reason to use
        this dedicated integration instead of pointing ``ChatOpenAI`` at a local server.

        Force the model to emit exactly one of a set of labels (classification/routing):

        ```python
        classifier = llm.with_guided_choice(["positive", "negative", "neutral"])
        classifier.invoke("I absolutely love this product!")
        # -> "positive"   (guaranteed in the set, no parsing needed)
        ```

        Use an ``enum.Enum`` and get a real member back:

        ```python
        import enum

        class Intent(enum.Enum):
            BILLING = "billing"
            TECHNICAL = "technical"
            SALES = "sales"

        router = llm.with_guided_choice(Intent)
        intent = router.invoke("My card was charged twice")
        # -> Intent.BILLING
        ```

        Force a regex pattern or a grammar:

        ```python
        phone = llm.with_guided_regex(r"(\d{3}) \d{3}-\d{4}")
        sql = llm.with_guided_grammar(gbnf_grammar_str)
        ```

    Structured output:
        Standard JSON-schema structured output uses OpenAI's ``response_format``:

        ```python
        llm.with_structured_output(City, method="json_schema")
        ```

        To route through vLLM's own ``guided_json`` constrained decoder instead:

        ```python
        llm.with_structured_output(City, method="guided_json")
        ```

    Key init args — guided decoding params:
        structured_outputs_format: Literal["structured_outputs", "legacy"]
            Wire format for guided-decoding parameters. ``"structured_outputs"``
            (default) uses the nested ``{"structured_outputs": {...}}`` object
            supported by vLLM ≥ 0.12.0. Set to ``"legacy"`` for older servers that
            still expect top-level ``guided_choice`` / ``guided_regex`` / etc. fields.
    """

    model: str = Field(alias="model_name")
    """Name of the model served by the vLLM server."""

    base_url: str = DEFAULT_BASE_URL
    """Base URL of the vLLM OpenAI-compatible server (must include `/v1`)."""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("VLLM_API_KEY", default="EMPTY")
    )
    """API key for the server.

    Not authentication for a normal local server — the `openai` SDK requires a
    non-empty value, so this defaults to the placeholder `'EMPTY'`. Only relevant
    when the server is launched with `vllm serve --api-key <token>`.
    """

    temperature: float | None = None
    """Sampling temperature."""

    max_tokens: int | None = Field(default=None)
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

    logprobs: bool | None = None
    """Whether to return log probabilities of the output tokens."""

    top_logprobs: int | None = None
    """Number of most likely tokens to return log probabilities for."""

    n: int | None = None
    """Number of chat completions to generate for each prompt."""

    streaming: bool = False
    """Whether to stream the results or not."""

    stream_usage: bool = True
    """Whether to include usage metadata in streaming output.

    When `True`, `stream_options={'include_usage': True}` is sent so the final
    streamed chunk carries token counts (vLLM, like OpenAI, omits usage from
    streamed chunks otherwise).
    """

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
    """Extra JSON body params forwarded verbatim to the server.

    Use this for vLLM-specific sampling params not covered by first-class fields,
    e.g. `top_k`, `repetition_penalty`, `min_p`, `chat_template_kwargs`.

    For guided / constrained decoding, prefer the typed helpers
    `with_guided_choice`, `with_guided_regex`, and `with_guided_grammar`, which
    build the correct wire payload automatically and respect
    `structured_outputs_format`. Raw usage:

    ```python
    # vLLM >= 0.12.0 (default structured_outputs_format)
    llm = ChatVLLM(..., extra_body={"structured_outputs": {"choice": ["yes", "no"]}})

    # vLLM < 0.12.0 (legacy format)
    llm = ChatVLLM(..., extra_body={"guided_choice": ["yes", "no"]})
    ```

    See the vLLM
    [structured outputs docs](https://docs.vllm.ai/en/stable/features/structured_outputs/).
    """

    structured_outputs_format: Literal["structured_outputs", "legacy"] = (
        "structured_outputs"
    )
    """Wire format for vLLM guided-decoding params.

    - `"structured_outputs"` (default): the nested ``{"structured_outputs": {...}}``
      object introduced in vLLM 0.12.0.
    - `"legacy"`: top-level ``guided_choice`` / ``guided_regex`` / ``guided_grammar``
      / ``guided_json`` fields supported by vLLM < 0.12.0.

    The `with_guided_choice`, `with_guided_regex`, `with_guided_grammar`, and
    ``with_structured_output(method="guided_json")`` helpers read this field and emit
    the correct payload automatically.
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional params passed through to the create call."""

    http_client: Any | None = None
    """Optional custom `httpx.Client` for sync requests."""

    http_async_client: Any | None = None
    """Optional custom `httpx.AsyncClient` for async requests."""

    _client: Any = PrivateAttr(default=None)
    """The underlying sync `chat.completions` resource."""

    _async_client: Any = PrivateAttr(default=None)
    """The underlying async `chat.completions` resource."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _build_extra(cls, values: dict[str, Any]) -> Any:
        """Route unknown kwargs into `model_kwargs`."""
        if not isinstance(values, dict):
            return values
        field_names: set[str] = set(cls.model_fields)
        field_names |= {
            field.alias for field in cls.model_fields.values() if field.alias
        }
        extra = dict(values.get("model_kwargs", {}))
        for field_name in list(values):
            if field_name in field_names:
                continue
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

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
            self._client = build_sync_client(params, self.http_client).chat.completions
        if self._async_client is None:
            self._async_client = build_async_client(
                params, self.http_async_client
            ).chat.completions
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-vllm"

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default params for calling the vLLM server."""
        exclude_if_none = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "n": self.n,
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
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="vllm",
            ls_model_name=params.get("model", self.model),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Assemble the request payload for the create call."""
        messages = self._convert_input(input_).to_messages()
        if stop is not None:
            kwargs["stop"] = stop
        payload = {**self._default_params, **kwargs}
        # Deep-merge extra_body so a bound guided-decoding fragment combines with
        # any instance-level extra_body instead of clobbering it.
        if "extra_body" in kwargs and self.extra_body:
            payload["extra_body"] = _deep_merge_extra_body(
                self.extra_body, kwargs["extra_body"]
            )
        payload["messages"] = [_convert_message_to_dict(m) for m in messages]
        return payload

    def _build_guided_extra_body(
        self,
        *,
        kind: Literal["choice", "regex", "grammar", "json"],
        value: Any,
        guided_decoding_backend: str | None,
    ) -> dict[str, Any]:
        """Build the ``extra_body`` fragment for one guided-decoding primitive.

        Args:
            kind: The type of constraint (`"choice"`, `"regex"`, `"grammar"`, or
                `"json"`).
            value: The constraint value (list of strings for choice, string for
                regex/grammar, dict for json).
            guided_decoding_backend: Optional backend selector forwarded as a
                top-level ``extra_body`` key (e.g. ``"xgrammar"``).

        Returns:
            A dict suitable for passing as ``extra_body`` to ``bind``.
        """
        if self.structured_outputs_format == "structured_outputs":
            fragment: dict[str, Any] = {"structured_outputs": {kind: value}}
        else:
            legacy_key = {
                "choice": "guided_choice",
                "regex": "guided_regex",
                "grammar": "guided_grammar",
                "json": "guided_json",
            }[kind]
            fragment = {legacy_key: value}
        if guided_decoding_backend is not None:
            fragment["guided_decoding_backend"] = guided_decoding_backend
        return fragment

    def _wrap_with_include_raw(
        self,
        llm: Runnable,
        output_parser: Runnable,
        *,
        include_raw: bool,
    ) -> Runnable:
        """Wrap ``llm | output_parser`` with the standard ``include_raw`` envelope.

        Args:
            llm: The bound LLM runnable.
            output_parser: The output parser to apply.
            include_raw: If ``True``, return a dict with ``'raw'``, ``'parsed'``,
                and ``'parsing_error'`` keys.

        Returns:
            A `Runnable` that either chains directly or returns the raw/parsed dict.
        """
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """Build a `ChatResult` from a create response."""
        generations = []
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        if response_dict.get("error"):
            raise ValueError(response_dict.get("error"))

        try:
            choices = response_dict["choices"]
        except KeyError as e:
            msg = f"Response missing 'choices' key: {response_dict.keys()}"
            raise KeyError(msg) from e

        if choices is None:
            # vLLM can return null choices on error without populating the error field.
            msg = (
                "Received response with null value for 'choices'. "
                "This can happen when using OpenAI-compatible APIs (e.g., vLLM) "
                "that return a response in an unexpected format. "
                f"Full response keys: {list(response_dict.keys())}"
            )
            raise TypeError(msg)

        token_usage = response_dict.get("usage")
        for res in choices:
            message = _convert_dict_to_message(res["message"])
            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage)
            gen_info = {**(generation_info or {})}
            if res.get("finish_reason") is not None:
                gen_info["finish_reason"] = res["finish_reason"]
            if "logprobs" in res:
                gen_info["logprobs"] = res["logprobs"]
            generations.append(
                ChatGeneration(message=message, generation_info=gen_info or None)
            )
        llm_output = {
            "token_usage": token_usage,
            "model_provider": "vllm",
            "model_name": response_dict.get("model", self.model),
        }
        if "id" in response_dict:
            llm_output["id"] = response_dict["id"]
        return ChatResult(generations=generations, llm_output=llm_output)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Convert a streamed chunk dict to a `ChatGenerationChunk`."""
        token_usage = chunk.get("usage")
        choices = chunk.get("choices", [])
        usage_metadata = _create_usage_metadata(token_usage) if token_usage else None
        if len(choices) == 0:
            # Final usage-only chunk (only AIMessageChunk carries usage_metadata).
            return ChatGenerationChunk(
                message=AIMessageChunk(content="", usage_metadata=usage_metadata),
                generation_info=base_generation_info,
            )

        choice = choices[0]
        if choice["delta"] is None:
            return None

        message_chunk = _convert_delta_to_message_chunk(
            choice["delta"], default_chunk_class
        )
        generation_info = {**base_generation_info} if base_generation_info else {}
        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason
            if model_name := chunk.get("model"):
                generation_info["model_name"] = model_name
        if logprobs := choice.get("logprobs"):
            generation_info["logprobs"] = logprobs
        if usage_metadata and isinstance(message_chunk, AIMessageChunk):
            message_chunk.usage_metadata = usage_metadata
        message_chunk.response_metadata["model_provider"] = "vllm"
        return ChatGenerationChunk(
            message=message_chunk, generation_info=generation_info or None
        )

    def _should_stream_usage(
        self, stream_usage: bool | None = None, **kwargs: Any
    ) -> bool:
        """Determine whether to request usage metadata during streaming."""
        stream_usage_sources = [
            stream_usage,
            kwargs.get("stream_options", {}).get("include_usage"),
            self.model_kwargs.get("stream_options", {}).get("include_usage"),
            self.stream_usage,
        ]
        for source in stream_usage_sources:
            if isinstance(source, bool):
                return source
        return self.stream_usage

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        response = self._client.create(**payload)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        response = await self._async_client.create(**payload)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        if self._should_stream_usage(stream_usage, **kwargs):
            kwargs["stream_options"] = {"include_usage": True}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        is_first_chunk = True
        for raw_chunk in self._client.create(**payload):
            chunk = raw_chunk if isinstance(raw_chunk, dict) else raw_chunk.model_dump()
            generation_chunk = self._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, {} if not is_first_chunk else None
            )
            if generation_chunk is None:
                continue
            default_chunk_class = generation_chunk.message.__class__
            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            is_first_chunk = False
            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        if self._should_stream_usage(stream_usage, **kwargs):
            kwargs["stream_options"] = {"include_usage": True}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        is_first_chunk = True
        response = await self._async_client.create(**payload)
        async for raw_chunk in response:
            chunk = raw_chunk if isinstance(raw_chunk, dict) else raw_chunk.model_dump()
            generation_chunk = self._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, {} if not is_first_chunk else None
            )
            if generation_chunk is None:
                continue
            default_chunk_class = generation_chunk.message.__class__
            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            is_first_chunk = False
            yield generation_chunk

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | Literal["auto", "any", "none"] | bool | None = None,  # noqa: PYI051
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Assumes the served model is compatible with the OpenAI tool-calling API.
        vLLM supports tool calling only when the server is started with
        `--enable-auto-tool-choice` and a matching `--tool-call-parser`.

        Args:
            tools: A list of tool definitions to bind, handled by
                [`convert_to_openai_tool`][langchain_core.utils.function_calling.convert_to_openai_tool].
            tool_choice: Which tool the model should call. One of a tool name,
                `'auto'`, `'none'`, `'any'`/`'required'`/`True` (force a call), a
                `dict`, or `None`/`False` (default behavior).
            strict: If `True`, requests strict schema adherence from the server.
            parallel_tool_calls: Set `False` to disable parallel tool calls.
            kwargs: Passed through to `bind`.
        """
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        tool_names = [t["function"]["name"] for t in formatted_tools if "function" in t]
        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif not isinstance(tool_choice, dict):
                msg = (
                    "Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
                raise ValueError(msg)
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema", "guided_json"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        guided_decoding_backend: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema, as an OpenAI function/tool schema, a JSON
                Schema, a `TypedDict` class, or a Pydantic class.
            method: The steering method, one of:

                - `'function_calling'`: uses tool calling (requires the server to
                    be launched with tool-calling support).
                - `'json_schema'`: uses OpenAI-style
                    `response_format={'type': 'json_schema', ...}`.
                - `'json_mode'`: uses `response_format={'type': 'json_object'}`. You
                    must instruct the model to produce JSON matching the schema.
                - `'guided_json'`: routes through vLLM's constrained ``guided_json``
                    decoder (sent as ``extra_body``). Uses `structured_outputs_format`
                    to pick the correct wire shape for the server version.
            include_raw: If `True`, return a dict with `'raw'`, `'parsed'`, and
                `'parsing_error'` keys.
            strict: If `True`, requests strict schema adherence (function calling).
            guided_decoding_backend: Backend selector for ``method='guided_json'``
                (e.g. ``"xgrammar"``). Forwarded as a top-level ``extra_body`` key.
            kwargs: Additional keyword args are not supported.

        Returns:
            A `Runnable` producing structured output.
        """
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = "schema must be specified when method is 'function_calling'."
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                strict=strict,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[cast("TypeBaseModel", schema)],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=cast("TypeBaseModel", schema))
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                msg = "schema must be specified when method is 'json_schema'."
                raise ValueError(msg)
            if is_pydantic_schema:
                schema = cast("TypeBaseModel", schema)
                if issubclass(schema, BaseModelV1):
                    json_schema = schema.schema()
                else:
                    json_schema = schema.model_json_schema()
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if is_typeddict(schema):
                    json_schema = convert_to_json_schema(schema)
                    if "required" not in json_schema:
                        json_schema["required"] = list(json_schema["properties"].keys())
                else:
                    json_schema = cast("dict", schema)
                output_parser = JsonOutputParser()
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "schema": json_schema,
                },
            }
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": json_schema,
                },
            )
        elif method == "guided_json":
            if schema is None:
                msg = "schema must be specified when method is 'guided_json'."
                raise ValueError(msg)
            if is_pydantic_schema:
                schema = cast("TypeBaseModel", schema)
                if issubclass(schema, BaseModelV1):
                    json_schema = schema.schema()
                else:
                    json_schema = schema.model_json_schema()
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if is_typeddict(schema):
                    json_schema = convert_to_json_schema(schema)
                    if "required" not in json_schema:
                        json_schema["required"] = list(json_schema["properties"].keys())
                else:
                    json_schema = cast("dict", schema)
                output_parser = JsonOutputParser()
            extra_body = self._build_guided_extra_body(
                kind="json",
                value=json_schema,
                guided_decoding_backend=guided_decoding_backend,
            )
            llm = self.bind(
                extra_body=extra_body,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": json_schema,
                },
            )
        else:
            msg = (
                "Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_schema', 'json_mode', or 'guided_json'. Received: '{method}'"
            )
            raise ValueError(msg)

        return self._wrap_with_include_raw(llm, output_parser, include_raw=include_raw)

    @overload
    def with_guided_choice(
        self,
        choices: type[EnumT],
        *,
        include_raw: bool = ...,
        guided_decoding_backend: str | None = ...,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, EnumT]: ...

    @overload
    def with_guided_choice(
        self,
        choices: Sequence[str],
        *,
        include_raw: bool = ...,
        guided_decoding_backend: str | None = ...,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, str]: ...

    def with_guided_choice(
        self,
        choices: Sequence[str] | type[enum.Enum],
        *,
        include_raw: bool = False,
        guided_decoding_backend: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Force the model to emit exactly one of a fixed set of labels.

        This uses vLLM's **guided / constrained decoding** to guarantee the output
        matches one of the provided choices at the token level — no parsing, no
        validation, no "sorry I can't answer that" escapes. This capability has
        **no equivalent in the OpenAI API** and is the primary reason to use this
        integration instead of pointing ``ChatOpenAI`` at a local vLLM server.

        Args:
            choices: A non-empty list/tuple of strings, or an ``enum.Enum`` subclass
                whose ``.value`` attributes are used as the wire labels. When an Enum
                is passed, the returned runnable maps the output string back to the
                corresponding Enum member.
            include_raw: If ``True``, return a dict with ``'raw'``, ``'parsed'``, and
                ``'parsing_error'`` keys instead of the label directly.
            guided_decoding_backend: Optional backend selector forwarded to vLLM
                (e.g. ``"xgrammar"``).
            kwargs: Passed through to ``bind``.

        Returns:
            A ``Runnable`` that returns a ``str`` (or an Enum member when an Enum
            subclass is passed) guaranteed to be one of the provided choices.

        Example:
            Classify sentiment — the model is forced to emit exactly one label:

            ```python
            classifier = llm.with_guided_choice(["positive", "negative", "neutral"])
            classifier.invoke("I absolutely love this product!")
            # -> "positive"
            ```

            Route with an Enum and get a real member back:

            ```python
            import enum

            class Intent(enum.Enum):
                BILLING = "billing"
                TECHNICAL = "technical"
                SALES = "sales"

            router = llm.with_guided_choice(Intent)
            intent = router.invoke("My card was charged twice")
            # -> Intent.BILLING
            ```
        """
        wire_choices, enum_map = _extract_guided_choices(choices)
        extra_body = self._build_guided_extra_body(
            kind="choice",
            value=wire_choices,
            guided_decoding_backend=guided_decoding_backend,
        )
        llm = self.bind(
            extra_body=extra_body,
            ls_structured_output_format={
                "kwargs": {"method": "guided_choice"},
                "schema": {"choice": wire_choices},
            },
            **kwargs,
        )
        if enum_map is not None:
            _map = enum_map
            output_parser: Runnable = StrOutputParser() | RunnableLambda(
                lambda s, m=_map: m[s.strip()]
            )
        else:
            output_parser = StrOutputParser()
        return self._wrap_with_include_raw(llm, output_parser, include_raw=include_raw)

    def with_guided_regex(
        self,
        pattern: str,
        *,
        include_raw: bool = False,
        guided_decoding_backend: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, str]:
        r"""Force the model to emit output that matches a regular expression.

        Uses vLLM's constrained decoding to guarantee every generated token
        stays within the language defined by ``pattern``. This has no equivalent
        in the OpenAI API.

        Args:
            pattern: A regular-expression string. The entire output is constrained
                to match this pattern (the server applies it token-by-token).
            include_raw: If ``True``, return a dict with ``'raw'``, ``'parsed'``,
                and ``'parsing_error'`` keys.
            guided_decoding_backend: Optional backend selector forwarded to vLLM
                (e.g. ``"xgrammar"``).
            kwargs: Passed through to ``bind``.

        Returns:
            A ``Runnable`` that returns a ``str`` guaranteed to match ``pattern``.

        Example:
            Force a phone-number format:

            ```python
            phone = llm.with_guided_regex(r"(\d{3}) \d{3}-\d{4}")
            phone.invoke("Reach me at 4155551234")
            # -> "(415) 555-1234"
            ```
        """
        if not pattern:
            msg = "pattern must be a non-empty string."
            raise ValueError(msg)
        extra_body = self._build_guided_extra_body(
            kind="regex",
            value=pattern,
            guided_decoding_backend=guided_decoding_backend,
        )
        llm = self.bind(
            extra_body=extra_body,
            ls_structured_output_format={
                "kwargs": {"method": "guided_regex"},
                "schema": {"regex": pattern},
            },
            **kwargs,
        )
        return self._wrap_with_include_raw(
            llm, StrOutputParser(), include_raw=include_raw
        )

    def with_guided_grammar(
        self,
        grammar: str,
        *,
        include_raw: bool = False,
        guided_decoding_backend: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, str]:
        """Force the model to emit output that conforms to a context-free grammar.

        Uses vLLM's constrained decoding with a GBNF (GGML BNF) or EBNF grammar
        string. This has no equivalent in the OpenAI API.

        Args:
            grammar: A grammar string in GBNF or EBNF format understood by the
                vLLM server's configured guided-decoding backend.
            include_raw: If ``True``, return a dict with ``'raw'``, ``'parsed'``,
                and ``'parsing_error'`` keys.
            guided_decoding_backend: Optional backend selector forwarded to vLLM
                (e.g. ``"xgrammar"``).
            kwargs: Passed through to ``bind``.

        Returns:
            A ``Runnable`` that returns a ``str`` conforming to ``grammar``.

        Example:
            Restrict output to simple ``SELECT`` SQL:

            ```python
            sql_grammar = '''
            root   ::= "SELECT " column " FROM " table
            column ::= "id" | "name" | "email"
            table  ::= "users" | "orders"
            '''
            sql = llm.with_guided_grammar(sql_grammar)
            sql.invoke("Get every user's email")
            # -> "SELECT email FROM users"
            ```
        """
        if not grammar:
            msg = "grammar must be a non-empty string."
            raise ValueError(msg)
        extra_body = self._build_guided_extra_body(
            kind="grammar",
            value=grammar,
            guided_decoding_backend=guided_decoding_backend,
        )
        llm = self.bind(
            extra_body=extra_body,
            ls_structured_output_format={
                "kwargs": {"method": "guided_grammar"},
                "schema": {"grammar": grammar},
            },
            **kwargs,
        )
        return self._wrap_with_include_raw(
            llm, StrOutputParser(), include_raw=include_raw
        )
