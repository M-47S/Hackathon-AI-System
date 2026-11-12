# GRADER/llm_client.py
import json
import logging
from typing import Any, Optional, Type, TypeVar

import httpx
import openai
from openai import OpenAI
from pydantic import BaseModel

from structs import (
    LLMError, LLMRateLimitError, LLMResponseParseError, LLMTimeoutError,
    ChatRequest, LLMResponse, LLMUsage,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: str = "qwen/qwen3.8-27b",
        timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = default_model
        self._timeout = timeout
        self._max_retries = max_retries

        if not api_key:
            raise LLMError("LLMClient: не передан api_key")

        # ОДИН httpx.Client с нормальным пулом соединений
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout),
            verify=False,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=60,
                keepalive_expiry=60.0,
            ),
        )
        # ОДИН OpenAI клиент — переиспользуем его всё время
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            http_client=self._http,
        )

    def chat(self, request: ChatRequest) -> LLMResponse:
        model = request.model or self._default_model

        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]
        if request.instructions and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": request.instructions})

        want_json = request.response_format is not None
        if want_json:
            schema_json = json.dumps(
                request.response_format.model_json_schema(), ensure_ascii=False,
            )
            schema_hint = (
                "Ответь ОДНИМ валидным JSON-объектом с реальными данными. "
                "Без markdown-обёрток и пояснений. "
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
                response = self._client.chat.completions.create(**kwargs)
            except openai.BadRequestError:
                if want_json:
                    kwargs.pop("response_format", None)
                    response = self._client.chat.completions.create(**kwargs)
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
            raise LLMError(f"APIError: {getattr(e, 'status_code', '?')}") from e
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

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # _parse_structured и _extract_usage_chat — без изменений
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
            raise LLMResponseParseError(f"Не JSON: {e}") from e
        if (isinstance(data, dict) and data.get("type") == "object"
                and isinstance(data.get("properties"), dict)
                and "properties" not in model_cls.model_fields):
            data = data["properties"]
        try:
            return model_cls.model_validate(data)
        except Exception as e:
            raise LLMResponseParseError(f"Не прошло валидацию: {e}") from e

    @staticmethod
    def _extract_usage_chat(response: Any) -> Optional[LLMUsage]:
        u = getattr(response, "usage", None)
        if u is None:
            return None
        return LLMUsage(
            input_tokens=getattr(u, "prompt_tokens", 0),
            output_tokens=getattr(u, "completion_tokens", 0),
            total_tokens=getattr(u, "total_tokens", 0),
        )