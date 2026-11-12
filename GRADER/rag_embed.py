# GRADER/rag_emded.py
"""
rag_embed.py — RemoteEmbedder для платформы.
Эмбеддинги через OpenAI-совместимый прокси (BAAI/bge-m3).

Заменяет RAG/Embedding/wrapper.py, который работал с локальной HF-моделью.
На хакатоне локальные модели не разворачиваются — используем прокси.
"""
from __future__ import annotations

import logging
from typing import List

import httpx
from openai import OpenAI

from structs import Embedding

log = logging.getLogger(__name__)


class RemoteEmbedder:
    """
    Минимальный duck-typing интерфейс, совместимый с EmbeddingWrapper:
      - _get_text_embedding(text) -> Embedding
      - _get_text_embedding_batch(texts) -> List[Embedding]
      - _get_query_embedding(query) -> Embedding
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "BAAI/bge-m3",
        batch_size: int = 32,
        timeout: float = 60.0,
    ):
        self._client = OpenAI(
            base_url=base_url, 
            api_key=api_key, 
            timeout=timeout,
            http_client=httpx.Client(verify=False, timeout=timeout))
        self._model = model
        self._batch_size = batch_size

    def _embed(self, texts: List[str]) -> List[Embedding]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        # OpenAI SDK возвращает data в том же порядке, что и input
        return [Embedding(d.embedding) for d in response.data]

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed([text])[0]

    def _get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[Embedding]:
        result: List[Embedding] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            result.extend(self._embed(batch))
            if show_progress:
                log.info("embedded %d/%d", min(i + self._batch_size, len(texts)), len(texts))
        return result

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)