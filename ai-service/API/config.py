from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Приоритет загрузки .env:
#   1. API/.env          — единый .env сервиса (новый)
#   2. LLM/.env          — старый .env LLM (backward compat)
#   3. ai-service/.env   — корневой .env
#   4. cwd/.env
_ENV_CANDIDATES = [
    Path(__file__).parent.parent / ".env",           # ai-service/.env (Приоритет)
    Path(__file__).parent / ".env",                  # API/.env 
    Path.cwd() / ".env",
]
for _p in _ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(_p, override=False)
        break


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


class Settings:
    # --- Service ---
    HOST: str = _env("AI_HOST", default="0.0.0.0")
    PORT: int = int(_env("AI_PORT", default="8001") or "8001")
    LOG_LEVEL: str = _env("LOG_LEVEL", default="INFO")
    
    # --- LLM (провайдер-агностично) ---
    LLM_BASE_URL: str = _env(
        "GIGACHAT_BASE_URL", "AI_BASE_URL", "LLM_BASE_URL",
        default="https://api.giga.chat/v1",
    )
    LLM_AUTH_KEY: str | None = _env(
        "GIGACHAT_AUTH_KEY", "AI_AUTH_KEY", "LLM_API_KEY",
    )
    LLM_SCOPE: str = _env("GIGACHAT_SCOPE", "AI_SCOPE", default="GIGACHAT_API_PERS")
    LLM_MODEL: str = _env("GIGACHAT_MODEL", "AI_MODEL", "LLM_MODEL", default="GigaChat-2-MAX")
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 800

    # --- RAG: пути к данным ---
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent # **/ai-service/
    RAG_KB_DIR: Path = Path(_env(
        "RAG_KB_DIR",
        default=str(_PROJECT_ROOT / "Storage" / "Source"),
    ))
    RAG_INDEX_DIR: Path = Path(_env(
        "RAG_INDEX_DIR",
        default=str(_PROJECT_ROOT / "Storage" / "Index"),
    ))
    RAG_TOP_K: int = int(_env("RAG_TOP_K", default="6") or "6")
    RAG_RERANK: bool = (_env("RAG_RERANK", default="true") or "true").lower() in ("1", "true", "yes")
    RAG_EXCLUDE_DOCS: set[str] = {
    s.strip() for s in (_env("RAG_EXCLUDE_DOCS", default="20-portfolio,26-general,00-product-map") or "").split(",") if s.strip()
}
    
    # ── авто-сборка индекса при первом запуске ──
    RAG_AUTO_BUILD: bool = (_env("RAG_AUTO_BUILD", default="true") or "true").lower() in ("1", "true", "yes")
    RAG_CHUNK_SIZE: int = int(_env("RAG_CHUNK_SIZE", default="640") or "640")
    RAG_CHUNK_OVERLAP: int = int(_env("RAG_CHUNK_OVERLAP", default="64") or "64")

    # --- Models - место хранения всех моделей в проекте ---
    MODELS_DIR: str = _env("MODELS_DIR", default= f"C:/Users/0/OneDrive/Desktop/HackAI/ai-service/Models")
    
    # --- Embedder (явные пути) ---
    EMBEDDING_MODEL_NAME: str = _env("EMBEDDING_MODEL_NAME", default= f"multilingual-e5-large")
    EMBEDDING_MODEL_PATH: str = MODELS_DIR + '/' + EMBEDDING_MODEL_NAME
    EMBEDDING_DEVICE: str = _env("EMBEDDING_DEVICE", default="cpu")
    EMBEDDING_BATCH_SIZE: int = int(_env("EMBEDDING_BATCH_SIZE", default="10") or "10")

    # --- Reranker (явные пути) ---
    RERANKER_MODEL_NAME: str = _env("RERANKER_MODEL_NAME", default= f"bge-reranker-v2-m3")
    RERANKER_MODEL_PATH: str = MODELS_DIR + '/' + RERANKER_MODEL_NAME
    RERANKER_DEVICE: str = _env("RERANKER_DEVICE", default="cpu")
    RERANKER_MAX_LENGTH: int = int(_env("RERANKER_MAX_LENGTH", default="1024") or "1024")