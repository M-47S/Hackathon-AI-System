# GRADER/rag_rank.py
import os
import httpx

class QwenReranker:
    def __init__(self, base_url: str, api_key: str, model: str = "Qwen3-Reranker-8B"):
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
            verify=False,
        )
        self._model = model

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
        response = self._client.post(
            "/v1/rerank",
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
        )
        response.raise_for_status()
        data = response.json()
        # формат: {"results": [{"index": 3, "relevance_score": 0.92}, ...]}
        return [(r["index"], r["relevance_score"]) for r in data["results"]]
    
    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __del__(self):
        self.close()