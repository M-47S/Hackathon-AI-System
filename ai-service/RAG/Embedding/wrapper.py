import logging
import os
from typing import List, Any
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from ..structs import Embedding

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("EMB_PATH", r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Embedding\models\multilingual-e5-large")


class EmbeddingWrapper(BaseEmbedding):
    """
    Обёртка над HuggingFaceEmbedding для модели intfloat/multilingual-e5-large.
    Инкапсулирует:
      - проверку наличия локальной папки с моделью;
      - автоматическое добавление префиксов passage:/query: для E5;
      - удобные методы для получения эмбеддингов;
      - опциональную установку модели глобально в LlamaIndex (Settings.embed_model).
    """

    model_path: str = MODEL_PATH
    device: str = "cpu" 
    
    _embed_model: HuggingFaceEmbedding = PrivateAttr()
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        device: str = "cpu",
        set_global: bool = True,
        **kwargs: Any
    ):
        super().__init__(
            model_path=os.path.abspath(model_path),
            device=device,
            **kwargs
        
        )
        self.model_path = os.path.abspath(model_path)
        self.device = device

        # Проверяем, что папка с моделью существует и в ней есть config.json
        config_file = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Модель не найдена по пути: {self.model_path}. "
                f"Убедитесь, что папка содержит файлы модели (config.json, model.safetensors и т.д.)."
            )

        logger.info(f"Загрузка модели из {self.model_path} на устройство {self.device}...")

        # Инициализация HuggingFaceEmbedding с префиксами для E5
        self._embed_model: HuggingFaceEmbedding  = HuggingFaceEmbedding(
            model_name=self.model_path,
            device=self.device,
            text_instruction="passage: ",  # для документов
            query_instruction="query: ",   # для запросов
        )

        # Опционально делаем модель глобальной для LlamaIndex
        if set_global:
            Settings.embed_model = self._embed_model
            logger.info("Модель установлена как глобальная (Settings.embed_model).")

        logger.info("Модель успешно загружена.")

    def _aget_query_embedding(self, query: str):
        vector = self._embed_model._aget_query_embedding(query)
        return Embedding(vector)
    
    def _get_query_embedding(self, query: str) -> Embedding:
        """Возвращает эмбеддинг для поискового запроса."""
        vector = self._embed_model.get_query_embedding(query)
        return Embedding(vector)

    def _get_text_embedding(self, text: str) -> Embedding:
        """Возвращает эмбеддинг для документа (чанка)."""
        vector = self._embed_model.get_text_embedding(text)
        return Embedding(vector)

    def _get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[Embedding]:
        """Возвращает эмбеддинги для списка текстов (документов)."""
        return list(
            map(Embedding, self._embed_model.get_text_embedding_batch(
                texts, show_progress=show_progress)
            )
        )
        
    @classmethod
    def class_name(cls) -> str:
        return "EmbeddingWrapper"

    @property
    def model(self) -> HuggingFaceEmbedding:
        """Доступ к объекту HuggingFaceEmbedding (при необходимости)."""
        return self._embed_model
    
if __name__ == "__main__":
    model = EmbeddingWrapper()
    
    query = "Что такое закон Ома?"
    document = "Описывающий характеристики тока в цепи в зависимости от Силы тока, напряжения и сопротивления."
    
    eq = model._get_query_embedding(query)
    ed = model._get_text_embedding(document)
    
    similiriry = (eq*ed)/(eq.norm*ed.norm)
    
    print(similiriry)