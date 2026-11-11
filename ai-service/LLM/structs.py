from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_input(self) -> Dict[str, Any]:
        """Формат элемента input для Responses API."""
        payload: Dict[str, Any] = {
            "type": "message",
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[LLMUsage] = None
    raw: Optional[Any] = None  # исходный response object, для debug
    parsed: Optional[Any] = None  # результат structured output (если запрашивался)


@dataclass(frozen=True)
class ChatRequest:
    """
    Удобный контейнер для запроса. LLMClient превращает его в вызов API.
    """
    messages: List[LLMMessage]
    model: Optional[str] = None
    instructions: Optional[str] = None  # системный промпт для Responses API
    temperature: float = 0.2
    max_output_tokens: Optional[int] = None
    response_format: Optional[Any] = None  # Pydantic-класс для structured output