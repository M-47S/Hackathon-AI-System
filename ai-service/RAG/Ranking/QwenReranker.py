# RAG/Ranking/QwenReranker.py
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class QwenReranker:
    """HTTP-реранкер через OpenAI-совместимый прокси.

    Автоматически определяет доступный endpoint (/v1/rerank или /rerank)
    при первом успешном вызове и кэширует его.

    При любой ошибке API или некорректном ответе возвращает пустой список —
    вызывающий код (pipeline_search) должен сделать fallback на FAISS/BM25.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "Qwen3-Reranker-8B",
        timeout: float = 30.0,
        verify: bool = True,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._endpoint: Optional[str] = None  
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            verify=verify,
            limits=httpx.Limits(
                max_connections=60,
                max_keepalive_connections=20,
            ),
        )

    def _try_endpoint(self, endpoint: str, payload: dict) -> Optional[dict]:
        """Пробует один endpoint. Возвращает dict или None.

        400/404/405 → скорее всего неверный endpoint, молча идём дальше.
        Остальные ошибки → логируем на DEBUG и возвращаем None.
        """
        try:
            resp = self._client.post(
                f"{self._base_url}{endpoint}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            if resp.status_code in (400, 404, 405):
                return None
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("rerank %s failed: %s", endpoint, e)
            return None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """Возвращает [(idx, score)], отсортированный по убыванию score.

        При любой ошибке возвращает [] (пустой список).
        """
        if not documents:
            return []

        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }

        # Если endpoint уже определён — используем только его.
        # Иначе пробуем по очереди: сначала OpenAI-style, потом «голый».
        candidates = [self._endpoint] if self._endpoint else ["/v1/rerank", "/rerank"]

        for endpoint in candidates:
            data = self._try_endpoint(endpoint, payload)
            if data is None:
                continue
            self._endpoint = endpoint  # кэшируем рабочий endpoint

            # Разные прокси отвечают по-разному
            entries = data.get("results") or data.get("data") or []
            result: List[Tuple[int, float]] = []
            for item in entries:
                try:
                    idx = int(item.get("index", -1))
                    score = float(
                        item.get("relevance_score", item.get("score", 0.0))
                    )
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(documents):
                    result.append((idx, score))
            if result:
                return result

        return []  # pipeline_search сделает FAISS-fallback

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __del__(self):
        self.close()