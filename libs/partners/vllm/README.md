# langchain-vllm

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-vllm?label=%20)](https://pypi.org/project/langchain-vllm/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-vllm)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-vllm)](https://pypistats.org/packages/langchain-vllm)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
uv add langchain-vllm
```

## 🤔 What is this?

This package contains the LangChain integration with [vLLM](https://docs.vllm.ai/).

It connects to a running vLLM [OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
(started with `vllm serve <model>`) over HTTP. Start a server, then point the
integration at it:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

```python
from langchain_vllm import ChatVLLM

llm = ChatVLLM(model="Qwen/Qwen2.5-1.5B-Instruct")
llm.invoke("Hello!")
```

No API key is required for a local server. `base_url` defaults to
`http://localhost:8000/v1`.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/integrations/langchain_vllm/). For conceptual guides, tutorials, and examples on using these classes, see the [LangChain Docs](https://docs.langchain.com/oss/python/integrations/providers/vllm).

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 🧪 Testing

Unit tests (no server required):

```bash
make test
```

Integration tests require a running vLLM server. `docker-compose.xpu.yml` in
`tests/integration_tests/` starts a chat server (port 8000) and an embeddings
server (port 8001) on Intel XPU:

```bash
# Run all integration tests in one command (serve → test → teardown)
make e2e_xpu
```

Or step by step:

```bash
# Start both XPU servers and wait until healthy
make serve_xpu

# Verify servers are up
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models

# Run all integration tests
VLLM_TEST_BASE_URL="http://localhost:8000/v1" \
VLLM_TEST_EMBED_BASE_URL="http://localhost:8001/v1" \
make integration_test

# Tear down
make serve_xpu_down
```

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Resources

- [LangChain Academy](https://academy.langchain.com/) — comprehensive, free courses on LangChain libraries and products, made by the LangChain team
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
