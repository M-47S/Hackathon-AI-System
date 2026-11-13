# RAG/Ranking/LocalReranker.py
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_PATH = r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Ranking\models\BAAI/bge-reranker-v2-m3"


class LocalReranker:
    """
    Локальный Cross-Encoder реранкер на bge-reranker-v2-m3.
    Работает на CPU/GPU без сетевых запросов.

    При ошибке (OOM, GPU crash, некорректный вход) возвращает пустой список —
    вызывающий код (pipeline_search) сделает fallback на FAISS/BM25.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str = "cpu",
        max_length: int = 1024, 
    ):
        self.model_path = model_path
        self._model: Optional[CrossEncoder] = CrossEncoder(
            model_path,
            device=device,
            max_length=max_length,
        )
        logger.info(
            "LocalReranker загружен: %s (device=%s, max_length=%d)",
            model_path, device, max_length,
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Возвращает [(original_index, score)], отсортированный по убыванию score.
        original_index — позиция документа во входном списке documents.
        При ошибке — пустой список (graceful degradation).
        """
        if not documents or self._model is None:
            return []

        try:
            pairs = [(query, doc) for doc in documents]
            scores = self._model.predict(pairs, show_progress_bar=False)
            indexed = list(enumerate(float(s) for s in scores))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return indexed[:top_n]
        except Exception as e:
            logger.warning("LocalReranker.rerank failed: %s", e)
            return []

    def close(self) -> None:
        """Освобождает модель. Полезно при hot-swap или graceful shutdown."""
        self._model = None

    def __del__(self):
        self.close()