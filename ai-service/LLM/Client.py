import logging
from typing import Any, List, Optional, Type, TypeVar

import httpx
import openai
from openai import OpenAI
from pydantic import BaseModel

from .auth import TokenProvider
from .exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMTimeoutError,
)
from .structs import ChatRequest, LLMMessage, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Тонкая обёртка над OpenAI Responses API.
    Явно передаёт контекст в виде списка сообщений (input).
    Поддерживает structured output через Pydantic-модели.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        token_provider: Optional[TokenProvider] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        verify_ssl: bool = False, # <-- ДОБАВЛЕНО: отключаем проверку по умолчанию для GigaChat
    ):
        self._default_model = default_model
        self._token_provider = token_provider
        
        # Создаем httpx клиент с отключенной проверкой SSL
        # (или можно передать verify="/path/to/russian_ca.pem", если хотите безопасно)
        http_client = httpx.Client(verify=verify_ssl, timeout=timeout)

        if token_provider is not None:
            self._client = OpenAI(
                base_url=base_url,
                api_key="placeholder",  # Заглушка
                timeout=timeout,
                max_retries=max_retries,
                http_client=http_client, 
            )
        else:
            self._client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                http_client=http_client,
            )

    # ---------- public ----------

    def chat(self, request: ChatRequest) -> LLMResponse:
        """
        Основной метод. Возвращает LLMResponse с текстом и (опционально) parsed.
        """
        
        if self._token_provider:
            self._client._client.headers["Authorization"] = self._auth_header()
        
        model = request.model or self._default_model
        input_items = [msg.to_openai_input() for msg in request.messages]

        # instructions — это системный промпт в Responses API[reference:2].
        # Если пользователь передал SYSTEM-сообщение, выносим его в instructions.
        instructions = request.instructions
        if instructions is None:
            system_msgs = [m for m in request.messages if m.role == "system"]
            if system_msgs:
                instructions = "\n\n".join(m.content for m in system_msgs)
                input_items = [
                    item for item, msg in zip(input_items, request.messages)
                    if msg.role != "system"
                ]

        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "temperature": request.temperature,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if request.max_output_tokens:
            kwargs["max_output_tokens"] = request.max_output_tokens

        try:
            if request.response_format is not None:
                # Structured output через .parse()[reference:3]
                response = self._client.responses.parse(
                    text_format=request.response_format,
                    **kwargs,
                )
                parsed = response.output_parsed
                text = response.output_text
            else:
                response = self._client.responses.create(**kwargs)
                parsed = None
                text = response.output_text
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e
        except openai.APIError as e:
            # 401 при протухшем токене GigaChat — инвалидируем и пробуем ещё раз
            if self._token_provider and "401" in str(e):
                self._token_provider.invalidate()
                self._client._client.headers["Authorization"] = self._auth_header()
                return self.chat(request)  # один повтор
            raise LLMError(str(e)) from e
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Unexpected LLM error: {e}") from e

        if not text:
            raise LLMResponseParseError("LLM returned empty text")

        return LLMResponse(
            text=text,
            model=getattr(response, "model", model),
            finish_reason=getattr(response, "status", None),
            usage=self._extract_usage(response),
            raw=response,
            parsed=parsed,
        )

    # ---------- helpers ----------

    def _auth_header(self) -> str:
        if not self._token_provider:
            return self._client._client.headers.get("Authorization", "")
        token = self._token_provider.get_token()
        return f"Bearer {token}"

    @staticmethod
    def _extract_usage(response: Any) -> Optional[LLMUsage]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return LLMUsage(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )
        
        
if __name__ == "__main__":
    from .auth import GigaChatTokenProvider
    import os
    from dotenv import load_dotenv
    from .structs import Role

    
    if os.path.exists(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\LLM\.env"):
        load_dotenv(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\LLM\.env")
        
    DEFAULT_AUTH_KEY = os.getenv("AI_AUTH_KEY")
    DEFAULT_BASE_URL = os.getenv("AI_BASE_URL", "https://api.giga.chat/v1")
    DEFAULT_SCOPE = DEFAULT_SCOPE = os.getenv("AI_SCOPE", "GIGACHAT_API_PERS")
    DEFAULT_MODEL = os.getenv("AI_MODEL", "GigaChat-2")
    
    provider = GigaChatTokenProvider(
        auth_key=DEFAULT_AUTH_KEY, 
        scope=DEFAULT_SCOPE,
        verify_ssl=False
    )
    client = LLMClient(
        base_url=DEFAULT_BASE_URL,
        token_provider=provider,
        default_model=DEFAULT_MODEL
    )
    
    request1 = ChatRequest(
        messages=[
            LLMMessage(role=Role.SYSTEM, content="Ты — консультант EnergyTool."),
            LLMMessage(role=Role.USER, content="Что такое АСКУЭ?")
        ],
        temperature=0.3
    )
    response1 = client.chat(request=request1)
    
    class AnswerSchema(BaseModel):
        answer: str
        confidence: float
        
    request2 = ChatRequest(
        messages=[
            LLMMessage(role=Role.SYSTEM, content="Ты — консультант EnergyTool."),
            LLMMessage(role=Role.USER, content="Что такое АСКУЭ?")
        ],
        temperature=0.1,
        response_format=AnswerSchema
        
    )
    
    response2 = client.chat(request=request2)
    
    print(response1.text)
    print(response2.parsed)