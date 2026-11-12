import json
import logging
from typing import Any, Optional, Type, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

from structs import (
    LLMError,
    LLMRateLimitError,
    LLMResponseParseError,
    LLMTimeoutError,
    ChatRequest,
    LLMResponse,
    LLMUsage,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = default_model
        self._max_retries = max_retries
        self._timeout = timeout

    def chat(self, request: ChatRequest) -> LLMResponse:
        if not self._api_key:
            raise LLMError("Не передан api_key")

        # Создаем клиент (на платформе можно переиспользовать, но для 100 вопросов это ок)
        client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
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

    @staticmethod
    def _parse_structured(text: str, model_cls: Type[T]) -> T:
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
                f"Ответ не является валидным JSON: {e}\nСырой ответ: {text[:500]}"
            ) from e

        if (
            isinstance(data, dict)
            and data.get("type") == "object"
            and isinstance(data.get("properties"), dict)
            and "properties" not in model_cls.model_fields
        ):
            data = data["properties"]

        try:
            return model_cls.model_validate(data)
        except Exception as e:
            raise LLMResponseParseError(
                f"Не удалось распарсить structured output: {e}\nСырой ответ: {text[:500]}"
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