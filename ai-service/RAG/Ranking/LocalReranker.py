# RAG/Ranking/LocalReranker.py
from __future__ import annotations
from typing import List, Optional, Tuple
import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_PATH = r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Ranking\models\BAAI/bge-reranker-v2-m3"

class LocalReranker:
    """
    Локальный Cross-Encoder реранкер.
    Не делает сетевых запросов, работает на CPU/GPU.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_path = model_path
        self._model = CrossEncoder(
            model_path,
            device=device,
            max_length=max_length,
        )
        logger.info("LocalReranker загружен: %s (device=%s)", model_path, device)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Возвращает список (original_index, score), отсортированный по убыванию score.
        original_index — позиция документа во входном списке documents.
        """
        if not documents:
            return []

        # CrossEncoder.predict принимает список пар (query, doc)
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # scores — numpy array; превращаем в список (index, score)
        indexed = list(enumerate(float(s) for s in scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_n]