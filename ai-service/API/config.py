from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем .env из любого разумного места (первое найденное)
_ENV_CANDIDATES = [
    Path(__file__).parent.parent / "LLM" / ".env",   # ai-service/LLM/.env
    Path(__file__).parent.parent / ".env",           # ai-service/.env
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
    # --- LLM (провайдер-агностично) ---
    LLM_BASE_URL: str = _env(
        "GIGACHAT_BASE_URL", "AI_BASE_URL", "LLM_BASE_URL",
        default="https://api.giga.chat/v1",
    )
    LLM_AUTH_KEY: str | None = _env(
        "GIGACHAT_AUTH_KEY", "AI_AUTH_KEY", "LLM_API_KEY",
    )
    LLM_SCOPE: str = _env("GIGACHAT_SCOPE", "AI_SCOPE", default="GIGACHAT_API_PERS")
    LLM_MODEL: str = _env("GIGACHAT_MODEL", "AI_MODEL", "LLM_MODEL", default="GigaChat-2")
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 800

    # --- RAG ---
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAG_KB_DIR: Path = Path(_env(
        "RAG_KB_DIR",
        default=str(_PROJECT_ROOT / "RAG" / "Store" / "_test_" / "Data Knowledge"),
    ))
    RAG_INDEX_DIR: Path = Path(_env(
        "RAG_INDEX_DIR",
        default=str(_PROJECT_ROOT / "RAG" / "Store" / "Storage"),
    ))
    RAG_TOP_K: int = 4
    RAG_RERANK: bool = True

    # --- Service ---
    HOST: str = _env("AI_HOST", default="0.0.0.0")
    PORT: int = int(_env("AI_PORT", default="8001") or "8001")
    LOG_LEVEL: str = _env("LOG_LEVEL", default="INFO")