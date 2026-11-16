class LLMError(Exception):
    """Базовая ошибка LLM-слоя."""


class LLMAuthError(LLMError):
    """Ошибка получения/обновления токена."""


class LLMRateLimitError(LLMError):
    """429 или исчерпана квота."""


class LLMTimeoutError(LLMError):
    """Таймаут запроса к провайдеру."""


class LLMResponseParseError(LLMError):
    """Ответ не соответствует ожидаемой структуре."""