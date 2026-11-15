"""AI Service — точка входа FastAPI."""

from __future__ import annotations

import faulthandler
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
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

    # -----------------------------------------------------------------------
    # Хелпер: гарантированный вывод в консоль (uvicorn перехватывает logging)
    # -----------------------------------------------------------------------
    def _log(msg: str, level: str = "info") -> None:
        prefix = {
            "info": "INFO", "warning": "WARN",
            "error": "ERR ", "debug": "DBG ",
        }.get(level, "INFO")
        print(f"[{prefix}] [RAG-BUILD] {msg}", flush=True)
        try:
            getattr(logger, level)(msg)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # 1. Embedder
    # -----------------------------------------------------------------------
    _log("=" * 60)
    _log("[1] Loading EmbeddingWrapper...")
    t_step = time.time()

    from RAG.Embedding import EmbeddingWrapper
    try:
        embedder = EmbeddingWrapper(
            model_path=Settings.EMBEDDING_MODEL_PATH,
            device=Settings.EMBEDDING_DEVICE,
            embed_batch_size=Settings.EMBEDDING_BATCH_SIZE,
        )
        _log(
            f"[1] Embedder loaded in {time.time()-t_step:.1f}s "
            f"(model={Settings.EMBEDDING_MODEL_PATH}, "
            f"device={Settings.EMBEDDING_DEVICE}, "
            f"batch={Settings.EMBEDDING_BATCH_SIZE})"
        )
    except Exception as e:
        _log(
            f"[1] Embedder load FAILED: {e}. "
            f"Check EMBEDDING_MODEL_PATH={Settings.EMBEDDING_MODEL_PATH}",
            level="error",
        )
        raise

    # -----------------------------------------------------------------------
    # 2. Reranker (опционально)
    # -----------------------------------------------------------------------
    reranker = None
    if Settings.RAG_RERANK:
        _log("[2] Loading Reranker...")
        t_step = time.time()
        try:
            from RAG.Ranking import LocalReranker
            reranker = LocalReranker(
                model_path=Settings.RERANKER_MODEL_PATH,
                device=Settings.RERANKER_DEVICE,
                max_length=Settings.RERANKER_MAX_LENGTH,
            )
            _log(
                f"[2] Reranker loaded in {time.time()-t_step:.1f}s "
                f"(model={Settings.RERANKER_MODEL_PATH}, "
                f"device={Settings.RERANKER_DEVICE}, "
                f"max_length={Settings.RERANKER_MAX_LENGTH})"
            )
        except Exception as e:
            _log(
                f"[2] Reranker not loaded ({e}) — retrieval without rerank. "
                f"Check RERANKER_MODEL_PATH={Settings.RERANKER_MODEL_PATH}",
                level="warning",
            )
    else:
        _log("[2] Reranker disabled (RAG_RERANK=false)")

    # -----------------------------------------------------------------------
    # 3. RAG index (auto-build if missing)
    # -----------------------------------------------------------------------
    from RAG.pipelines import pipeline_build_from_directory, pipeline_load, build_bm25_index

    def _load_or_build_index():
        # --- 3.1 Попытка загрузить готовый индекс ---
        _log(f"[3.1] Trying to load RAG index from cache: {Settings.RAG_INDEX_DIR}")
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

        # --- 3.2 Проверка флага авто-сборки ---
        if not Settings.RAG_AUTO_BUILD:
            raise RuntimeError(
                f"RAG index missing at {Settings.RAG_INDEX_DIR} "
                f"and RAG_AUTO_BUILD=false. "
                f"Either set RAG_AUTO_BUILD=true or build index manually."
            )

        # --- 3.3 Проверка KB ---
        _log(f"[3.2] Checking KB directory: {Settings.RAG_KB_DIR}")
        t_step = time.time()
        kb_path = Path(Settings.RAG_KB_DIR).resolve()
        _log(f"[3.2] Resolved KB path: {kb_path}")

        if not kb_path.exists():
            raise RuntimeError(
                f"KB directory missing: {kb_path}. "
                f"Set RAG_KB_DIR in .env to the folder with .md files."
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

        # --- 3.4 Импорт chunker (llama_index init) ---
        _log("[3.3] Importing MarkdownChunker (llama_index init, may take 5-10s)...")
        t_step = time.time()
        from RAG.Chunking import MarkdownChunker
        _log(f"[3.3] MarkdownChunker imported in {time.time()-t_step:.1f}s")

        # --- 3.5 Обёртка embedder для прогресса ---
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
                f"{time.time()-t_b:.1f}s "
                f"(total {batch_counter['total_chars']} chars)"
            )
            return result

        embedder._get_text_embedding_batch = logged_batch
        _log("[3.4] Embedder wrapped")

        # --- 3.6 Сборка индекса ---
        _log(
            f"[3.5] Building RAG index from {len(md_files)} .md files "
            f"(chunk={Settings.RAG_CHUNK_SIZE}, "
            f"overlap={Settings.RAG_CHUNK_OVERLAP}, "
            f"batch={Settings.EMBEDDING_BATCH_SIZE}, "
            f"device={Settings.EMBEDDING_DEVICE})."
        )
        _log(
            "[3.5] ⏳ Expected duration: 60-180 sec on CPU with "
            "multilingual-e5-large. Progress appears per batch below. "
            "First batch takes 15-30s (model warm-up). DO NOT interrupt."
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
            _log(
                f"[3.5] RAG index build FAILED after "
                f"{time.time()-t_build:.1f}s: {e}",
                level="error",
            )
            raise RuntimeError(f"RAG index build failed: {e}") from e
        finally:
            faulthandler.cancel_dump_traceback_later()

        _log(
            f"[3.5] RAG index built in {time.time()-t_build:.1f}s: "
            f"{stats.summary()}"
        )

        # --- 3.7 Верификация ---
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
        _log(f"[3] RAG index load/build FAILED: {e}", level="error")
        raise

    # -----------------------------------------------------------------------
    # 4. LLM client
    # -----------------------------------------------------------------------
    _log("[4] Initializing LLM client...")
    from LLM import LLMClient, GigaChatTokenProvider

    if not Settings.LLM_AUTH_KEY:
        raise RuntimeError(
            "No LLM_AUTH_KEY / AI_AUTH_KEY / GIGACHAT_AUTH_KEY. "
            "Set it in .env"
        )

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
    _log(
        f"[4] LLM client ready: base={Settings.LLM_BASE_URL} "
        f"model={Settings.LLM_MODEL} provider="
        f"{'GigaChat' if is_gigachat else 'OpenAI-compat'}"
    )
    
    # ---Гибридный поиск ---
    bm25_index = None
    if Settings.USE_HYBRID:
        t0 = time.time()
        bm25_index = build_bm25_index(doc_store)
        logger.info("BM25 index built in %.1fs", time.time() - t0)

    # -----------------------------------------------------------------------
    # 5. ChatService
    # -----------------------------------------------------------------------
    _log("[5] Initializing ChatService...")
    app.state.chat_service = ChatService(
        llm_client=llm_client,
        embedder=embedder,
        doc_store=doc_store,
        vec_store=vec_store,
        reranker=reranker,
        bm25_index=bm25_index,
        use_hybrid=Settings.USE_HYBRID,
        top_k=Settings.RAG_TOP_K,
        exclude_docs=Settings.RAG_EXCLUDE_DOCS
    )
    _log(f"[5] ChatService ready (top_k={Settings.RAG_TOP_K})")

    logger.info("=" * 60)
    logger.info("AI Service ready at http://%s:%d", Settings.HOST, Settings.PORT)
    logger.info("=" * 60)
    print(
        f"\n✅ AI Service ready at http://{Settings.HOST}:{Settings.PORT}\n"
        f"   Swagger: http://{Settings.HOST}:{Settings.PORT}/docs\n"
        f"   Health:  http://{Settings.HOST}:{Settings.PORT}/health\n",
        flush=True,
    )

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