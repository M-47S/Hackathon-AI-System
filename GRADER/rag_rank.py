# GRADER/rag_rank.py
import logging
import httpx

logger = logging.getLogger(__name__)


class QwenReranker:
    def __init__(self, base_url: str, api_key: str, model: str = "qwen/qwen3-reranker-8b"):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._endpoint: str | None = None  # определим при первом вызове
        self._client = httpx.Client(
            timeout=httpx.Timeout(30.0),
            verify=False,
            limits=httpx.Limits(max_connections=60, max_keepalive_connections=20),
        )

    def _try_endpoint(self, endpoint: str, payload: dict) -> dict | None:
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

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }

        candidates = [self._endpoint] if self._endpoint else ["/v1/rerank", "/rerank"]
        for endpoint in candidates:
            data = self._try_endpoint(endpoint, payload)
            if data is None:
                continue
            self._endpoint = endpoint
            entries = data.get("results") or data.get("data") or []
            result = []
            for item in entries:
                idx = int(item.get("index", -1))
                score = float(item.get("relevance_score", item.get("score", 0.0)))
                if 0 <= idx < len(documents):
                    result.append((idx, score))
            if result:
                return result

        return []   # пустой — pipeline_search сделает FAISS-fallback

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass