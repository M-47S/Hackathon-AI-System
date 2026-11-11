import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .exceptions import LLMAuthError

logger = logging.getLogger(__name__)

GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_TTL_SECONDS = 30 * 60          # 30 минут по документации
REFRESH_MARGIN_SECONDS = 60            # обновляем за минуту до истечения


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float  # unix timestamp


class TokenProvider:
    """Базовый интерфейс. Для OpenAI-совместимых API токен не нужен."""

    def get_token(self) -> Optional[str]:
        return None

    def invalidate(self) -> None:
        pass


class StaticTokenProvider(TokenProvider):
    """Просто возвращает фиксированный токен (для OpenAI, локального прокси и т.д.)."""

    def __init__(self, token: str):
        self._token = token

    def get_token(self) -> str:
        return self._token


class GigaChatTokenProvider(TokenProvider):
    """
    Потокобезопасный провайдер OAuth-токена GigaChat.
    Токен живёт 30 минут; обновляется автоматически при истечении.
    """

    def __init__(
        self,
        auth_key: str,
        scope: str = "GIGACHAT_API_PERS",
        verify_ssl: bool = True,
    ):
        self._auth_key = auth_key
        self._scope = scope
        self._verify_ssl = verify_ssl
        self._lock = threading.Lock()
        self._cached: Optional[_CachedToken] = None

    def get_token(self) -> str:
        with self._lock:
            if self._cached and self._is_valid(self._cached):
                return self._cached.access_token
            self._cached = self._fetch_new_token()
            return self._cached.access_token

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None

    @staticmethod
    def _is_valid(cached: _CachedToken) -> bool:
        return time.time() < (cached.expires_at - REFRESH_MARGIN_SECONDS)

    def _fetch_new_token(self) -> _CachedToken:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": self._generate_rquid(),
            "Authorization": f"Basic {self._auth_key}",
        }
        data = {"scope": self._scope}

        try:
            resp = requests.post(
                GIGACHAT_OAUTH_URL,
                headers=headers,
                data=data,
                timeout=30,
                verify=self._verify_ssl,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as e:
            raise LLMAuthError(f"GigaChat OAuth request failed: {e}") from e

        token = payload.get("access_token")
        if not token:
            raise LLMAuthError(f"GigaChat OAuth response missing access_token: {payload}")

        # expires_at в ответе — unix timestamp в миллисекундах[reference:1]
        expires_at_ms = payload.get("expires_at")
        if expires_at_ms:
            expires_at = expires_at_ms / 1000.0
        else:
            expires_at = time.time() + DEFAULT_TTL_SECONDS
            logger.warning(
                "GigaChat OAuth response missing expires_at, "
                f"falling back to {DEFAULT_TTL_SECONDS}s TTL"
            )

        logger.info(
            f"GigaChat token refreshed, ttl={expires_at - time.time():.0f}s"
        )
        return _CachedToken(access_token=token, expires_at=expires_at)

    @staticmethod
    def _generate_rquid() -> str:
        import uuid
        return str(uuid.uuid4())