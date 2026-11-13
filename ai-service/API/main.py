"""AI Service — точка входа FastAPI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from .config import Settings
from .schemas import AIResponseForm, HealthResponse, RequestToAI
from .service import ChatService

logger = logging.getLogger("ai-service")


# ---------------------------------------------------------------------------
# Lifespan — предзагрузка «тяжёлых» ресурсов
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("=" * 60)
    logger.info("AI Service starting — loading heavy resources...")
    logger.info("=" * 60)

    # --- 1. Embedder ---
    from RAG.Embedding import EmbeddingWrapper
    embedder = EmbeddingWrapper()
    logger.info("Embedder loaded: %s", type(embedder).__name__)

    # --- 2. Reranker (опционально) ---
    reranker = None
    if Settings.RAG_RERANK:
        try:
            from RAG.Ranking import LocalReranker
            reranker = LocalReranker(device="cpu")
            logger.info("Reranker loaded")
        except Exception as e:
            logger.warning("Reranker not loaded (%s) — retrieval without rerank", e)

    # --- 3. RAG index ---
    from RAG.pipelines import pipeline_load
    try:
        doc_store, vec_store = pipeline_load(
            Settings.RAG_INDEX_DIR, Settings.RAG_KB_DIR,
        )
    except FileNotFoundError as e:
        logger.error(
            "RAG index not found at %s. Run: python -m RAG.prepare_index",
            Settings.RAG_INDEX_DIR,
        )
        raise RuntimeError("RAG index missing") from e
    logger.info(
        "RAG index loaded: %d chunks, %d leaf-chunks",
        len(doc_store), len(doc_store.leaf_chunks),
    )

    # --- 4. LLM client ---
    from LLM import LLMClient, GigaChatTokenProvider
    if not Settings.LLM_AUTH_KEY:
        raise RuntimeError("No LLM_AUTH_KEY / AI_AUTH_KEY / GIGACHAT_AUTH_KEY")

    is_gigachat = "giga.chat" in Settings.LLM_BASE_URL
    if is_gigachat:
        provider = GigaChatTokenProvider(
            auth_key=Settings.LLM_AUTH_KEY,
            scope=Settings.LLM_SCOPE,
            verify_ssl=False,
        )
        llm_client = LLMClient(
            base_url=Settings.LLM_BASE_URL,
            token_provider=provider,
            default_model=Settings.LLM_MODEL,
            verify_ssl=False,
        )
    else:
        llm_client = LLMClient(
            base_url=Settings.LLM_BASE_URL,
            api_key=Settings.LLM_AUTH_KEY,
            default_model=Settings.LLM_MODEL,
            verify_ssl=True,
        )
    logger.info("LLM client: base=%s model=%s",
                Settings.LLM_BASE_URL, Settings.LLM_MODEL)

    # --- 5. ChatService ---
    app.state.chat_service = ChatService(
        llm_client=llm_client,
        embedder=embedder,
        doc_store=doc_store,
        vec_store=vec_store,
        reranker=reranker,
        top_k=Settings.RAG_TOP_K,
    )

    logger.info("=" * 60)
    logger.info("AI Service ready at http://%s:%d", Settings.HOST, Settings.PORT)
    logger.info("=" * 60)

    try:
        yield
    finally:
        logger.info("AI Service shutting down...")
        try:
            llm_client.close()
        except Exception:
            pass
        if reranker:
            try:
                reranker.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EnergyTool AI Service",
    version="1.0.0",
    description="ИИ-консультант EnergyTool: RAG + LLM со structured output",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/v1/chat", response_model=AIResponseForm)
def chat(req: RequestToAI) -> AIResponseForm:
    """Основной эндпоинт: контекст → структурированный AI-ответ.

    Sync (`def`) — FastAPI выполнит в threadpool, не блокируя event loop.
    """
    service: ChatService = app.state.chat_service
    return service.handle(req)


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=Settings.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(app, host=Settings.HOST, port=Settings.PORT)