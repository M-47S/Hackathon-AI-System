"""
test.py — комплексная проверка RAG-системы EnergyTool.

Слои:
  1. TestInfrastructure — LLM / Embedding / Reranker (внешние подключения)
  2. TestChunker        — MarkdownChunker (LlamaIndex)
  3. TestStores         — DocumentStore / VectorStore (persistence)
  4. TestPipelineFill   — pipeline_fill_store на реальной KB
  5. TestMetrics        — ground truth + расчёт метрик
  6. TestRetrieval      — качество поиска (hit@5, mrr@5)
  7. TestE2E            — запрос пользователя → retrieval → LLM → ответ
  8. TestEdgeCases      — пустой запрос, сломанный реранкер, нерелевантный запрос

Запуск:
    pytest test.py -v                     # всё
    pytest test.py -v -m "not slow"       # без медленных
    pytest test.py -v -m "smoke"          # только инфраструктура
    pytest test.py -v -k TestE2E          # только E2E
    python test.py                        # standalone, -v -s

Env:
    RAG_KB_DIR, RAG_INDEX_DIR            — пути к KB и индексу
    GIGACHAT_AUTH_KEY / GEMINI_API_KEY / OPENAI_API_KEY  — для LLM-тестов
    RAG_HIT_THRESHOLD (default 0.6), RAG_MRR_THRESHOLD (default 0.35)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pytest

# ---------------------------------------------------------------------------
# Env loading — обязательно ДО фикстур
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _ENV_CANDIDATES = [
        Path(__file__).parent.parent / "LLM" / ".env",   # ai-service/LLM/.env
        Path(__file__).parent / ".env",                   # ai-service/RAG/.env
        Path.cwd() / ".env",                              # cwd/.env
    ]
    for _p in _ENV_CANDIDATES:
        if _p.exists():
            load_dotenv(_p, override=False)
            print(f"  🔑 .env loaded: {_p}")
            break
except ImportError:
    print("  ⚠️  python-dotenv не установлен — читаем из os.environ")


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    """Возвращает первое непустое значение из перечисленных env-переменных."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_KB_DIR = Path(
    r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service"
    r"\RAG\Store\_test_\Data Knowledge"
)
DEFAULT_INDEX_DIR = Path(
    r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service"
    r"\RAG\Store\Storage"
)

KB_DIR = Path(os.getenv("RAG_KB_DIR", str(DEFAULT_KB_DIR)))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))

HIT_THRESHOLD = float(os.getenv("RAG_HIT_THRESHOLD", "0.6"))
MRR_THRESHOLD = float(os.getenv("RAG_MRR_THRESHOLD", "0.35"))



# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedder():
    """EmbeddingWrapper — локальная модель (sentence-transformers)."""
    try:
        from RAG.Embedding import EmbeddingWrapper
        return EmbeddingWrapper()
    except Exception as e:
        pytest.skip(f"EmbeddingWrapper не загрузился: {e}")


@pytest.fixture(scope="session")
def reranker():
    """LocalReranker (CrossEncoder). Если модели нет — None, тесты уйдут в fallback."""
    try:
        from RAG.Ranking import LocalReranker
        return LocalReranker(device="cpu")
    except Exception as e:
        print(f"\n  ⚠️  LocalReranker не загружен: {e} — тесты работают без реранкера")
        return None


@pytest.fixture(scope="session")
def llm_client():
    """
    LLMClient с провайдером из env.

    Порядок приоритетов:
      1. GigaChat  (GIGACHAT_AUTH_KEY | AI_AUTH_KEY)
      2. Gemini    (GEMINI_API_KEY)
      3. OpenAI    (OPENAI_API_KEY)

    Для GigaChat verify_ssl=False обязателен И для LLMClient, И для
    GigaChatTokenProvider — Сбер использует сертификаты Минцифры,
    которых нет в стандартном trust store Python.
    """
    try:
        from LLM import LLMClient, GigaChatTokenProvider
    except Exception as e:
        pytest.skip(f"LLM не импортируется: {e}")

    # --- 1. GigaChat ---
    gc_key = _env_first("GIGACHAT_AUTH_KEY", "AI_AUTH_KEY")
    if gc_key:
        base_url = _env_first(
            "GIGACHAT_BASE_URL", "AI_BASE_URL",
            default="https://api.giga.chat/v1",
        )
        scope = _env_first(
            "GIGACHAT_SCOPE", "AI_SCOPE",
            default="GIGACHAT_API_PERS",
        )
        model = _env_first(
            "GIGACHAT_MODEL", "AI_MODEL",
            default="GigaChat-2",
        )
        print(f"\n  🔧 GigaChat: base_url={base_url} scope={scope} model={model} "
              f"auth_key={gc_key[:8]}…")

        provider = GigaChatTokenProvider(
            auth_key=gc_key,
            scope=scope,
            verify_ssl=False,          # ← ключевая правка
        )
        return LLMClient(
            base_url=base_url,
            token_provider=provider,
            default_model=model,
            verify_ssl=False,
        )

    # --- 2. Gemini ---
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return LLMClient(
            base_url=os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            api_key=gemini_key,
            default_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            verify_ssl=True,
        )

    # --- 3. OpenAI ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return LLMClient(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=openai_key,
            default_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            verify_ssl=True,
        )

    pytest.skip(
        "Нет ключей: GIGACHAT_AUTH_KEY / AI_AUTH_KEY / GEMINI_API_KEY / OPENAI_API_KEY"
    )


@pytest.fixture(scope="session")
def kb_files() -> List[Path]:
    if not KB_DIR.exists():
        pytest.skip(f"KB не найдена: {KB_DIR}")
    files = sorted(KB_DIR.rglob("*.md"))
    if not files:
        pytest.skip(f"KB пуста: {KB_DIR}")
    return files


@pytest.fixture(scope="session")
def indexed_store(embedder, kb_files):
    """
    Один раз за сессию: строит doc_store + vec_store из KB.
    Использует те же параметры чанкера, что и продакшн (640/64).
    """
    from RAG.pipelines import pipeline_fill_store
    from RAG.Store import DocumentStore, VectorStore
    from RAG.Chunking import MarkdownChunker

    doc_store = DocumentStore(documents_root=KB_DIR)
    probe = embedder._get_text_embedding("probe")
    vec_store = VectorStore(dim=probe.size)

    t0 = time.time()
    stats = pipeline_fill_store(
        doc_store, vec_store, embedder,
        paths=kb_files,
        chunker=MarkdownChunker(chunk_size=640, chunk_overlap=64),
    )
    print(f"\n  📚 Индекс за {time.time()-t0:.1f}s: {stats.summary()}")
    return doc_store, vec_store


# ---------------------------------------------------------------------------
# 1. Infrastructure
# ---------------------------------------------------------------------------

class TestInfrastructure:
    """Smoke-тесты внешних зависимостей. Быстрые, ловят битые подключения."""

    @pytest.mark.smoke
    def test_embedding_single(self, embedder):
        emb = embedder._get_text_embedding("Тестовый текст")
        assert emb.size > 0, "Размерность эмбеддинга нулевая"
        assert emb.norm > 0, "Нулевая норма — модель не выдала вектор"

    @pytest.mark.smoke
    def test_embedding_batch(self, embedder):
        texts = ["первый", "второй", "третий"]
        embs = embedder._get_text_embedding_batch(texts)
        assert len(embs) == 3
        assert all(e.size == embs[0].size for e in embs)

    @pytest.mark.smoke
    def test_embedding_query(self, embedder):
        emb = embedder._get_query_embedding("запрос про АСКУЭ")
        assert emb.size > 0

    @pytest.mark.smoke
    def test_llm_chat(self, llm_client):
        from LLM import ChatRequest, LLMMessage, Role
        resp = llm_client.chat(ChatRequest(
            messages=[
                LLMMessage(role=Role.SYSTEM, content="Отвечай коротко."),
                LLMMessage(role=Role.USER, content="Скажи одно слово: ok"),
            ],
            temperature=0.0,
            max_output_tokens=20,
        ))
        assert resp.text and resp.text.strip()
        print(f"\n  💬 LLM: {resp.text!r}")

    @pytest.mark.smoke
    def test_reranker(self, reranker):
        if reranker is None:
            pytest.skip("Реранкер не загружен")
        results = reranker.rerank(
            query="Что такое АСКУЭ?",
            documents=[
                "АСКУЭ — автоматизированная система контроля и учёта электроэнергии",
                "Солнечные панели преобразуют свет в электричество",
                "Курс доллара вырос на 2%",
            ],
            top_n=3,
        )
        assert len(results) == 3
        # Релевантный документ должен быть первым
        assert results[0][0] == 0, (
            f"Реранкер поставил нерелевантный документ первым: {results}"
        )


# ---------------------------------------------------------------------------
# 2. Chunker
# ---------------------------------------------------------------------------

class TestChunker:
    """MarkdownChunker: разбиение, иерархия parent-child, edge cases."""

    def test_basic_chunking(self):
        from RAG.Chunking import MarkdownChunker
        md = (
            "# Документ\n\n"
            "## Раздел 1\nТекст первого раздела.\n\n"
            "## Раздел 2\nТекст второго раздела.\n"
        )
        chunks, leaf_ids, next_id = MarkdownChunker(
            chunk_size=200, chunk_overlap=20
        ).chunk(md, "test_doc")

        assert len(chunks) > 0
        assert len(leaf_ids) > 0
        assert next_id > 0
        assert all(c.chunk_id in [ch.chunk_id for ch in chunks] for c in chunks)

    def test_hierarchy(self):
        from RAG.Chunking import MarkdownChunker
        md = "# A\n## B\nText B.\n### C\nText C."
        chunks, leaf_ids, _ = MarkdownChunker(
            chunk_size=100, chunk_overlap=10
        ).chunk(md, "test_doc")

        leaves = [c for c in chunks if c.is_leaf]
        parents = [c for c in chunks if not c.is_leaf]
        assert leaves, "Нет листьев"
        assert parents, "Нет родителей"
        for leaf in leaves:
            assert leaf.parent_chunk_id is not None, (
                f"Лист {leaf.chunk_id} без parent_chunk_id"
            )

    def test_embedding_text_has_context(self):
        """embedding_text должен содержать имя документа или заголовок раздела."""
        from RAG.Chunking import MarkdownChunker
        md = "# Документ АСКУЭ\n## Тарифы\nТекст про тарифы."
        chunks, leaf_ids, _ = MarkdownChunker(
            chunk_size=200, chunk_overlap=20
        ).chunk(md, "askue_doc")

        leaves = [c for c in chunks if c.is_leaf]
        assert leaves
        assert leaves[0].embedding_text
        # Хотя бы один из маркеров должен быть в embedding_text
        et = leaves[0].embedding_text.lower()
        assert "аскуэ" in et or "тариф" in et or "askue" in et, (
            f"embedding_text без контекста: {leaves[0].embedding_text[:200]}"
        )

    def test_empty_markdown(self):
        from RAG.Chunking import MarkdownChunker
        chunks, leaf_ids, _ = MarkdownChunker().chunk("", "empty")
        assert chunks == []
        assert leaf_ids == []

    def test_long_section_splits(self):
        """Длинная секция должна разбиться на несколько листьев."""
        from RAG.Chunking import MarkdownChunker
        long_text = "Предложение про энергетику. " * 200
        md = f"# Doc\n## Section\n{long_text}"
        chunks, leaf_ids, _ = MarkdownChunker(
            chunk_size=200, chunk_overlap=20
        ).chunk(md, "long_doc")
        assert len(leaf_ids) > 1, f"Ожидалось >1 листа, получено {len(leaf_ids)}"


# ---------------------------------------------------------------------------
# 3. Stores
# ---------------------------------------------------------------------------

class TestStores:
    """Persistence: save/load обоих store'ов + целостность."""

    @pytest.mark.slow
    def test_save_load_roundtrip(self, indexed_store, tmp_path):
        from RAG.pipelines import pipeline_save, pipeline_load

        doc_store, vec_store = indexed_store
        pipeline_save(doc_store, vec_store, tmp_path)

        doc_store2, vec_store2 = pipeline_load(tmp_path, KB_DIR)
        assert len(doc_store2) == len(doc_store)
        assert len(vec_store2) == len(vec_store)
        assert len(doc_store2.leaf_chunks) == len(doc_store.leaf_chunks)
        # Один и тот же chunk_id доступен
        first_leaf = doc_store.leaf_chunks[0]
        loaded = doc_store2.get_chunk(first_leaf.chunk_id)
        assert loaded is not None
        assert loaded.text == first_leaf.text

    def test_small_to_big_parent(self, indexed_store):
        """get_chunk_with_context(PARENT) должен вернуть родительский текст."""
        from RAG.structs import ContextMode

        doc_store, _ = indexed_store
        # Ищем лист, у которого есть родитель
        leaf = next(
            (c for c in doc_store.leaf_chunks if c.parent_chunk_id is not None),
            None,
        )
        if leaf is None:
            pytest.skip("В индексе нет листьев с родителем")

        ctx = doc_store.get_chunk_with_context(leaf.chunk_id, mode=ContextMode.PARENT)
        assert ctx.parent is not None
        assert ctx.context_text == ctx.parent.text
        assert len(ctx.context_text) >= len(leaf.text)


# ---------------------------------------------------------------------------
# 4. Pipeline fill
# ---------------------------------------------------------------------------

class TestPipelineFill:
    """pipeline_fill_store на реальной KB."""

    @pytest.mark.slow
    def test_fill_stats(self, indexed_store):
        doc_store, vec_store = indexed_store
        assert len(doc_store) > 0, "DocumentStore пуст"
        assert len(vec_store) > 0, "VectorStore пуст"
        assert len(doc_store.leaf_chunks) == len(vec_store), (
            f"Листьев {len(doc_store.leaf_chunks)} ≠ векторов {len(vec_store)}"
        )

    def test_all_documents_have_leaves(self, indexed_store):
        doc_store, _ = indexed_store
        for doc in doc_store.documents:
            chunks = doc_store.get_chunks_of_document(doc.document_id)
            leaves = [c for c in chunks if c.is_leaf]
            assert leaves, f"Документ {doc.document_id} без листьев"


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Санити-чеки metrics.py и ground truth."""

    def test_ground_truth_format(self):
        from RAG.metrics import GROUND_TRUTH
        assert len(GROUND_TRUTH) > 0
        for item in GROUND_TRUTH:
            assert "id" in item
            assert "question" in item
            assert "relevant_documents" in item
            assert isinstance(item["relevant_documents"], list)
            assert len(item["relevant_documents"]) > 0, (
                f"Пустой relevant_documents у id={item['id']}"
            )

    def test_metric_computation(self):
        from RAG.metrics import _compute_query_metrics
        m = _compute_query_metrics(
            query_id="1",
            question="test",
            retrieved_docs=["a", "b", "c", "d", "e"],
            relevant_docs=["b", "d"],
            k=5,
        )
        assert m.hit == 1.0
        assert m.recall == 1.0          # оба в top-5
        assert m.mrr == pytest.approx(0.5)   # b на позиции 2
        assert m.found_at == 2
        assert m.precision == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# 6. Retrieval quality
# ---------------------------------------------------------------------------

class TestRetrieval:
    """Качество поиска на GROUND_TRUTH из metrics.py."""

    def _retrieval_fn(self, embedder, indexed_store, reranker):
        from RAG.pipelines import (
            pipeline_search_by_text,
            QUERY_INSTRUCTIONS,
            _adaptive_max_per_document,
        )
        doc_store, vec_store = indexed_store

        def fn(query_text: str, top_k: int) -> List[str]:
            hits = pipeline_search_by_text(
                embedder, doc_store, vec_store, query_text,
                query_instruction=QUERY_INSTRUCTIONS["detailed_ru"],
                top_k=top_k,
                min_score=0.0,
                filter_toc=True,
                max_per_document=_adaptive_max_per_document(query_text),
                reranker=reranker,
                rerank_oversample=5,
            )
            return [h.chunk.document_id for h in hits]

        return fn

    @pytest.mark.slow
    def test_hit_rate(self, embedder, indexed_store, reranker):
        from RAG.metrics import evaluate_retrieval, GROUND_TRUTH

        agg, per_query = evaluate_retrieval(
            retrieval_fn=self._retrieval_fn(embedder, indexed_store, reranker),
            ground_truth=GROUND_TRUTH,
            k=5,
            verbose=False,
        )
        print(f"\n  📊 {agg}")

        # Информативный вывод провалов — если порог не пройден, видно причину
        if agg["hit@5"] < HIT_THRESHOLD:
            fails = [m for m in per_query if m.hit == 0]
            print(f"\n  ❌ hit@5={agg['hit@5']:.2f} < {HIT_THRESHOLD}")
            for m in fails:
                print(f"     id={m.query_id}: retrieved={m.retrieved[:5]} "
                      f"≠ relevant={m.relevant}")

        assert agg["hit@5"] >= HIT_THRESHOLD, (
            f"hit@5={agg['hit@5']:.2f} ниже порога {HIT_THRESHOLD}"
        )
        assert agg["mrr@5"] >= MRR_THRESHOLD, (
            f"mrr@5={agg['mrr@5']:.2f} ниже порога {MRR_THRESHOLD}"
        )

    def test_search_returns_chunks(self, embedder, indexed_store):
        from RAG.pipelines import pipeline_search_by_text
        doc_store, vec_store = indexed_store
        hits = pipeline_search_by_text(
            embedder, doc_store, vec_store,
            "Что такое АСКУЭ?",
            top_k=3,
        )
        assert 0 < len(hits) <= 3
        assert all(h.chunk.text.strip() for h in hits)


# ---------------------------------------------------------------------------
# 7. E2E
# ---------------------------------------------------------------------------

class TestE2E:
    """Полный сценарий: запрос (с историей) → retrieval → LLM → ответ."""

    @pytest.mark.slow
    def test_full_dialog(self, embedder, indexed_store, reranker, llm_client):
        from RAG.pipelines import (
            pipeline_search_with_context_by_text,
            QUERY_INSTRUCTIONS,
        )
        from RAG.structs import ContextMode
        from LLM import ChatRequest, LLMMessage, Role

        doc_store, vec_store = indexed_store
        question = (
            "Пользователь: У нас небольшое производство.\n"
            "Ассистент: Какая у вас цель?\n"
            "Пользователь: Сократить расходы на свет.\n"
            "Ассистент: Есть ли у вас солнечные станции?\n"
            "Пользователь: Ещё нет. Что посоветуете для юга России?"
        )
        last_turn = question.split("Пользователь:")[-1].strip()

        contexts = pipeline_search_with_context_by_text(
            embedder, doc_store, vec_store,
            query_text=last_turn,
            query_instruction=QUERY_INSTRUCTIONS["detailed_ru"],
            rerank_query_text=last_turn,
            mode=ContextMode.PARENT,
            top_k=4,
            reranker=reranker,
            rerank_oversample=3,
            filter_toc=True,
        )
        assert contexts, "Retrieval ничего не вернул"
        print(f"\n  🔍 Найдено {len(contexts)} контекстов: "
              f"{[c.chunk.document_id for c in contexts]}")

        ctx_block = "\n\n---\n\n".join(
            c.context_text[:2000] for c in contexts
        )

        resp = llm_client.chat(ChatRequest(
            messages=[
                LLMMessage(role=Role.SYSTEM, content=(
                    "Ты консультант EnergyTool. Отвечай ТОЛЬКО по контексту. "
                    "До 150 слов, в конце — один уточняющий вопрос."
                )),
                LLMMessage(role=Role.USER, content=(
                    f"Контекст:\n{ctx_block}\n\n"
                    f"Диалог:\n{question}"
                )),
            ],
            temperature=0.1,
            max_output_tokens=400,
        ))
        answer = (resp.text or "").strip()
        print(f"\n  💬 Ответ ({len(answer)} символов): {answer[:250]}…")

        assert answer, "Пустой ответ от LLM"
        assert len(answer) < 2000, f"Слишком длинный ответ: {len(answer)} символов"
        # Консультативный маркер
        keywords = (
            "солнечн", "панел", "станц", "квт", "регион",
            "потреблен", "накопител", "тариф", "уточн", "сетев",
        )
        assert any(k in answer.lower() for k in keywords), (
            f"Ответ не по теме: {answer[:200]}"
        )

    @pytest.mark.slow
    def test_out_of_scope_refusal(self, llm_client):
        from LLM import ChatRequest, LLMMessage, Role
        resp = llm_client.chat(ChatRequest(
            messages=[
                LLMMessage(role=Role.SYSTEM, content=(
                    "Ты консультант EnergyTool. Если вопрос вне темы — "
                    "скажи 'Это вне моей компетенции' и предложи специалиста."
                )),
                LLMMessage(role=Role.USER, content="Сколько стоит iPhone 15?"),
            ],
            temperature=0.0,
            max_output_tokens=100,
        ))
        text = (resp.text or "").lower()
        print(f"\n  💬 Отказ: {resp.text[:200]!r}")
        assert any(k in text for k in (
            "вне", "не компетен", "специалист", "не могу", "не помог"
        )), f"Не отказался от off-topic: {resp.text[:200]}"


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Устойчивость: пустые запросы, битые реранкеры, off-topic."""

    def test_empty_query(self, embedder, indexed_store):
        from RAG.pipelines import pipeline_search_by_text
        doc_store, vec_store = indexed_store
        # Не должно упасть
        hits = pipeline_search_by_text(
            embedder, doc_store, vec_store, "",
            top_k=3, min_score=0.0,
        )
        assert isinstance(hits, list)

    def test_reranker_failure_fallback(self, embedder, indexed_store):
        """Сломанный реранкер → FAISS-fallback, вопрос не теряется."""
        from RAG.pipelines import pipeline_search_by_text

        class BrokenReranker:
            def rerank(self, query, documents, top_n=5):
                raise RuntimeError("simulated failure")

        doc_store, vec_store = indexed_store
        hits = pipeline_search_by_text(
            embedder, doc_store, vec_store,
            "Что такое АСКУЭ?",
            top_k=3, min_score=0.0,
            reranker=BrokenReranker(),
        )
        assert isinstance(hits, list)
        # Все hits должны быть non-reranked
        assert all(not h.reranked for h in hits), (
            "При падении реранкера hits должны иметь reranked=False"
        )

    def test_toc_filter_short_chunks(self):
        """Короткие чанки (<150) не должны считаться TOC."""
        from RAG.pipelines import _is_toc_like
        from RAG.structs import Chunk
        short = Chunk(
            chunk_id=1, document_id="d", text="[link](x) короткий", position=0,
        )
        assert _is_toc_like(short) is False

    def test_toc_filter_real_toc(self):
        """Чанк из markdown-ссылок → TOC."""
        from RAG.pipelines import _is_toc_like
        from RAG.structs import Chunk
        toc_text = " ".join(f"[Section {i}](url{i})" for i in range(20))
        toc = Chunk(
            chunk_id=2, document_id="d", text=toc_text, position=0,
        )
        assert _is_toc_like(toc) is True


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))