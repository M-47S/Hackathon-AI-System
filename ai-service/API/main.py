"""AI Service — точка входа FastAPI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import faulthandler
import sys
import time
from pathlib import Path

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
    try:
        embedder = EmbeddingWrapper(
            model_name=Settings.EMBEDDING_MODEL_NAME,
            device=Settings.EMBEDDING_DEVICE,
            embed_batch_size=Settings.EMBEDDING_BATCH_SIZE,
        )
        logger.info(
            "Embedder loaded: %s (model=%s, device=%s)",
            type(embedder).__name__,
            Settings.EMBEDDING_MODEL_NAME,
            Settings.EMBEDDING_DEVICE,
        )
    except Exception as e:
        logger.error("Embedder load failed: %s", e)
        logger.error("Check EMBEDDING_MODEL_NAME=%s", Settings.EMBEDDING_MODEL_NAME)
        raise

    # --- 2. Reranker (опционально) ---
    reranker = None
    if Settings.RAG_RERANK:
        try:
            from RAG.Ranking import LocalReranker
            reranker = LocalReranker(
                model_path=Settings.RERANKER_MODEL_PATH,
                device=Settings.RERANKER_DEVICE,
                max_length=Settings.RERANKER_MAX_LENGTH,
            )
            logger.info(
                "Reranker loaded (model=%s, device=%s, max_length=%d)",
                Settings.RERANKER_MODEL_PATH,
                Settings.RERANKER_DEVICE,
                Settings.RERANKER_MAX_LENGTH,
            )
        except Exception as e:
            logger.warning(
                "Reranker not loaded (%s) — retrieval without rerank. "
                "Check RERANKER_MODEL_PATH=%s",
                e, Settings.RERANKER_MODEL_PATH,
            )

    # --- 3. RAG index (auto-build if missing) ---
    from RAG.pipelines import pipeline_build_from_directory, pipeline_load

    def _log(msg: str, level: str = "info") -> None:
        """Логирование с гарантированным flush (uvicorn иногда буферизует)."""
        getattr(logger, level)(msg)
        sys.stderr.flush()
        sys.stdout.flush()

    def _load_or_build_index():
        _log("[3.1] Trying to load RAG index from cache...")
        t_step = time.time()
        try:
            doc_store, vec_store = pipeline_load(
                Settings.RAG_INDEX_DIR, Settings.RAG_KB_DIR,
            )
            _log(
                f"[3.1] RAG index loaded from cache in {time.time()-t_step:.1f}s: "
                f"{len(doc_store)} chunks, {len(doc_store.leaf_chunks)} leaves"
            )
            return doc_store, vec_store
        except FileNotFoundError:
            _log(
                f"[3.1] RAG index not found at {Settings.RAG_INDEX_DIR} "
                f"(elapsed {time.time()-t_step:.1f}s)",
                level="warning",
            )

        if not Settings.RAG_AUTO_BUILD:
            raise RuntimeError(
                f"RAG index missing at {Settings.RAG_INDEX_DIR} "
                f"and RAG_AUTO_BUILD=false. "
                f"Either set RAG_AUTO_BUILD=true or build index manually."
            )

        # --- 3.2 Проверка KB ---
        _log(f"[3.2] Checking KB directory: {Settings.RAG_KB_DIR}")
        t_step = time.time()
        kb_path = Path(Settings.RAG_KB_DIR).resolve()
        _log(f"[3.2] Resolved KB path: {kb_path}")

        if not kb_path.exists():
            raise RuntimeError(
                f"KB directory missing: {kb_path}. "
                f"Set RAG_KB_DIR in API/.env to the folder with .md files."
            )

        _log(f"[3.2] Scanning {kb_path} for .md files (rglob)...")
        md_files = list(kb_path.rglob("*.md"))
        _log(
            f"[3.2] Found {len(md_files)} .md files "
            f"in {time.time()-t_step:.1f}s"
        )
        if not md_files:
            raise RuntimeError(
                f"KB directory exists but has no .md files: {kb_path}"
            )

        # --- 3.3 Импорт chunker (llama_index инициализация) ---
        _log("[3.3] Importing MarkdownChunker (llama_index init, may take 5-10s)...")
        t_step = time.time()
        from RAG.Chunking import MarkdownChunker
        _log(f"[3.3] MarkdownChunker imported in {time.time()-t_step:.1f}s")

        # --- 3.4 Обёртка вокруг embedder для прогресса ---
        _log("[3.4] Wrapping embedder with progress logging...")
        original_batch = embedder._get_text_embedding_batch
        batch_counter = {"n": 0, "total_chars": 0}

        def logged_batch(texts, *args, **kwargs):
            batch_counter["n"] += 1
            n_chars = sum(len(t) for t in texts)
            batch_counter["total_chars"] += n_chars
            _log(
                f"[embed] batch #{batch_counter['n']}: "
                f"{len(texts)} texts, {n_chars} chars"
            )
            t_b = time.time()
            result = original_batch(texts, *args, **kwargs)
            _log(
                f"[embed] batch #{batch_counter['n']} done in "
                f"{time.time()-t_b:.1f}s (total {batch_counter['total_chars']} chars)"
            )
            return result

        embedder._get_text_embedding_batch = logged_batch
        _log("[3.4] Embedder wrapped")

        # --- 3.5 Сборка индекса ---
        _log(
            f"[3.5] Building RAG index from {len(md_files)} .md files "
            f"(chunk={Settings.RAG_CHUNK_SIZE}, overlap={Settings.RAG_CHUNK_OVERLAP}). "
            f"Expected duration: 2-5 min on CPU. "
            f"Progress will appear below per embedding batch."
        )
        t_build = time.time()

        # faulthandler: через 90s покажет traceback всех потоков если сборка висит
        faulthandler.dump_traceback_later(90, repeat=True)

        try:
            stats = pipeline_build_from_directory(
                raw_dir=kb_path,
                index_dir=Settings.RAG_INDEX_DIR,
                embedder=embedder,
                chunker=MarkdownChunker(
                    chunk_size=Settings.RAG_CHUNK_SIZE,
                    chunk_overlap=Settings.RAG_CHUNK_OVERLAP,
                ),
                pattern="**/*.md",
                save=True,
            )
        except Exception as e:
            _log(f"[3.5] RAG index build FAILED after "
                 f"{time.time()-t_build:.1f}s: {e}", level="error")
            raise RuntimeError(f"RAG index build failed: {e}") from e
        finally:
            faulthandler.cancel_dump_traceback_later()

        _log(
            f"[3.5] RAG index built in {time.time()-t_build:.1f}s: "
            f"{stats.summary()}"
        )

        # --- 3.6 Верификация ---
        _log("[3.6] Reloading built index from disk...")
        t_step = time.time()
        doc_store, vec_store = pipeline_load(
            Settings.RAG_INDEX_DIR, Settings.RAG_KB_DIR,
        )
        _log(
            f"[3.6] RAG index verified in {time.time()-t_step:.1f}s: "
            f"{len(doc_store)} chunks, {len(doc_store.leaf_chunks)} leaves"
        )
        return doc_store, vec_store

    try:
        doc_store, vec_store = _load_or_build_index()
    except Exception as e:
        logger.error("RAG index load/build failed: %s", e)
        raise

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