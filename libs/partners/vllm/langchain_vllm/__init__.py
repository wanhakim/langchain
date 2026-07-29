"""LangChain integration for vLLM OpenAI-compatible servers."""

from langchain_vllm._version import __version__
from langchain_vllm.chat_models import ChatVLLM
from langchain_vllm.embeddings import VLLMEmbeddings
from langchain_vllm.llms import VLLM

__all__ = [
    "VLLM",
    "ChatVLLM",
    "VLLMEmbeddings",
    "__version__",
]
