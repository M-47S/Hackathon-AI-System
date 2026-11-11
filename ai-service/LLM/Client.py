import json
import logging
from typing import Any, Optional, Type, TypeVar

import httpx
import openai
import urllib3
from openai import OpenAI
from pydantic import BaseModel

from .auth import TokenProvider
from .exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMTimeoutError,
)
from .structs import ChatRequest, LLMResponse, LLMUsage

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        token_provider: Optional[TokenProvider] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        verify_ssl: bool = False,
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = default_model
        self._token_provider = token_provider
        self._max_retries = max_retries
        self._http_client = httpx.Client(verify=verify_ssl, timeout=timeout)

    # ---------- public ----------

    def chat(self, request: ChatRequest, _retry_depth: int = 0) -> LLMResponse:
        token = self._token_provider.get_token() if self._token_provider else self._api_key
        if not token:
            raise LLMError("Не передан ни token_provider, ни api_key")

        client = OpenAI(
            base_url=self._base_url,
            api_key=token,
            timeout=self._http_client.timeout,
            max_retries=self._max_retries,
            http_client=self._http_client,
        )

        model = request.model or self._default_model

        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]
        if request.instructions and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": request.instructions})

        # ---- Structured output ----
        want_json = request.response_format is not None
        if want_json:
            schema_json = json.dumps(
                request.response_format.model_json_schema(),
                ensure_ascii=False,
            )
            # ⚠️ Ключевой момент: явно запрещаем модели воспроизводить схему.
            schema_hint = (
                "Ответь ОДНИМ валидным JSON-объектом с реальными данными. "
                "Без markdown-обёрток и пояснений. "
                "Структура полей должна соответствовать схеме ниже, "
                "но заполняй её фактическими значениями. "
                "НЕ повторяй саму схему, НЕ добавляй служебные поля "
                "'properties', 'required', 'type', 'title', '$defs'. "
                "В ответе должны быть только ключи из схемы с их значениями. "
                f"Схема: {schema_json}"
            )
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n{schema_hint}"
            else:
                messages.insert(0, {"role": "system", "content": schema_hint})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_output_tokens:
            kwargs["max_tokens"] = request.max_output_tokens
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except openai.BadRequestError:
                if want_json:
                    kwargs.pop("response_format", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            text = response.choices[0].message.content or ""
            parsed = None
            if want_json and text:
                parsed = self._parse_structured(text, request.response_format)

        except openai.AuthenticationError as e:
            if self._token_provider and _retry_depth < 1:
                logger.warning("Получен 401. Инвалидируем токен и пробуем снова...")
                self._token_provider.invalidate()
                return self.chat(request, _retry_depth=_retry_depth + 1)
            raise LLMError(f"Ошибка авторизации: {e}") from e

        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e

        except openai.APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e

        except openai.APIError as e:
            status_code = getattr(e, "status_code", "unknown")
            body = getattr(e, "body", "unknown")
            logger.error(f"APIError {status_code}: {body}")
            raise LLMError(f"APIError {status_code}: {body}") from e

        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Unexpected LLM error: {e}") from e

        if not text:
            raise LLMResponseParseError("LLM returned empty text")

        return LLMResponse(
            text=text,
            model=getattr(response, "model", model),
            finish_reason=response.choices[0].finish_reason,
            usage=self._extract_usage_chat(response),
            raw=response,
            parsed=parsed,
        )

    # ---------- helpers ----------

    @staticmethod
    def _parse_structured(text: str, model_cls: Type[T]) -> T:
        """
        Парсит JSON-строку в Pydantic-модель.
        Защита от:
          - markdown-обёрток ```json ... ```
          - моделей, которые возвращают саму JSON Schema с заполненными
            значениями внутри 'properties' (типично для GigaChat).
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip("`\n ")

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMResponseParseError(
                f"Ответ не является валидным JSON: {e}\n"
                f"Сырой ответ: {text[:500]}"
            ) from e

        # 🔧 Защита от GigaChat: если пришла "схема" вместо данных —
        # достаём значения из data["properties"].
        if (
            isinstance(data, dict)
            and data.get("type") == "object"
            and isinstance(data.get("properties"), dict)
            and "properties" not in model_cls.model_fields  # не путать с реальным полем
        ):
            logger.warning(
                "Модель вернула JSON Schema вместо данных. "
                "Извлекаю значения из 'properties'."
            )
            data = data["properties"]

        try:
            return model_cls.model_validate(data)
        except Exception as e:
            raise LLMResponseParseError(
                f"Не удалось распарсить structured output: {e}\n"
                f"Сырой ответ: {text[:500]}"
            ) from e

    @staticmethod
    def _extract_usage_chat(response: Any) -> Optional[LLMUsage]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        return LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )
    
if __name__ == "__main__":
    from .auth import GigaChatTokenProvider
    import os
    from dotenv import load_dotenv
    from .structs import Role, LLMMessage

    
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