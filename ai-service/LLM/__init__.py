# LLM/__init__.py
from .Client import LLMClient
from .auth import GigaChatTokenProvider, StaticTokenProvider, TokenProvider
from .exceptions import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMTimeoutError,
)
from .structs import ChatRequest, LLMMessage, LLMResponse, LLMUsage, Role

__all__ = [
    # Client
    "LLMClient",
    # Structs
    "ChatRequest",
    "LLMMessage",
    "LLMResponse",
    "LLMUsage",
    "Role",
    # Auth
    "TokenProvider",
    "StaticTokenProvider",
    "GigaChatTokenProvider",
    # Exceptions
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMResponseParseError",
]