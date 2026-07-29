"""OpenAI SDK client builders for vLLM servers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import openai

if TYPE_CHECKING:
    import httpx


def build_client_params(  # noqa: PLR0913
    *,
    api_key: str,
    base_url: str,
    timeout: Any = None,
    max_retries: int | None = None,
    default_headers: Any = None,
    default_query: Any = None,
) -> dict[str, Any]:
    """Assemble the shared kwargs for the `openai` sync and async clients.

    Args:
        api_key: The API key to authenticate with. For a local vLLM server this is
            typically the placeholder `'EMPTY'`; the `openai` SDK requires a
            non-empty value even when the server does not check it.
        base_url: The base URL of the vLLM OpenAI-compatible server, e.g.
            `'http://localhost:8000/v1'`.
        timeout: Request timeout passed to the client.
        max_retries: Maximum number of retries on transient errors.
        default_headers: Headers sent with every request.
        default_query: Query params sent with every request.

    Returns:
        A dict of client kwargs with `None` values removed.
    """
    params: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
        "max_retries": max_retries,
        "default_headers": default_headers,
        "default_query": default_query,
    }
    return {k: v for k, v in params.items() if v is not None}


def build_sync_client(
    params: dict[str, Any],
    http_client: httpx.Client | None = None,
) -> openai.OpenAI:
    """Build a synchronous `openai.OpenAI` client.

    Args:
        params: Client kwargs from `build_client_params`.
        http_client: Optional custom `httpx.Client`.

    Returns:
        A configured `openai.OpenAI` instance.
    """
    if http_client is not None:
        return openai.OpenAI(**params, http_client=http_client)
    return openai.OpenAI(**params)


def build_async_client(
    params: dict[str, Any],
    http_client: httpx.AsyncClient | None = None,
) -> openai.AsyncOpenAI:
    """Build an asynchronous `openai.AsyncOpenAI` client.

    Args:
        params: Client kwargs from `build_client_params`.
        http_client: Optional custom `httpx.AsyncClient`.

    Returns:
        A configured `openai.AsyncOpenAI` instance.
    """
    if http_client is not None:
        return openai.AsyncOpenAI(**params, http_client=http_client)
    return openai.AsyncOpenAI(**params)
