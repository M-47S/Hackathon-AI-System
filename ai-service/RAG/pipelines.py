"""
RAG/pipelines.py — сценарный слой RAG.

Связывает вместе:
  - MarkdownChunker   (ленивый импорт — не тянем llama_index в solution.py)
  - EmbeddingWrapper  (ленивый импорт — по той же причине)
  - DocumentStore + VectorStore

Основные функции:
  - pipeline_search(...)               -> List[RetrievedChunk]
  - pipeline_fill_store(...)           -> FillStats

Утилиты:
  - pipeline_search_with_context(...)  — search + Small-to-Big
  - pipeline_save / pipeline_load      — персистентность обоих store'ов
  - pipeline_build_from_directory(...) — one-shot пересборка индекса
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Tuple, Union, Protocol, runtime_checkable

from .structs import (
    Chunk,
    ChunkContext,
    ContextMode,
    Document,
    Embedding,
    VectorHit,
)
from .Store import DocumentStore, VectorStore

if TYPE_CHECKING:
    from .Chunking import MarkdownChunker
    from .Embedding import EmbeddingWrapper


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    """Результат поиска: чанк + его релевантность (cosine)."""
    chunk: Chunk
    score: float
    reranked: bool = False


@dataclass
class FillStats:
    """Статистика pipeline_fill_store."""
    documents_added: int = 0
    chunks_added: int = 0
    leaves_indexed: int = 0
    skipped: List[Tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        s = (
            f"documents={self.documents_added}, "
            f"chunks={self.chunks_added}, "
            f"leaves={self.leaves_indexed}, "
            f"skipped={len(self.skipped)}"
        )
        if self.skipped:
            preview = "; ".join(f"{p} ({r})" for p, r in self.skipped[:3])
            s += f" [{preview}{'...' if len(self.skipped) > 3 else ''}]"
        return s

@runtime_checkable
class RerankerLike(Protocol):
    """
    Duck-type контракт для реранкеров.
    Ему соответствуют и LocalReranker, и QwenReranker — без наследования.
    """
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]: ...

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def pipeline_search(
    doc_store: DocumentStore,
    vec_store: VectorStore,
    query: Embedding,
    top_k: int = 5,
    min_score: Optional[float] = None,
    chunk_filter: Optional[Callable[[Chunk], bool]] = None,
    oversample: int = 3,
    reranker: Optional[RerankerLike] = None,
    query_text: Optional[str] = None,
    rerank_oversample: int = 5,
) -> List[RetrievedChunk]:
    """
    Возвращает top_k релевантных чанков.

    Два режима ранжирования:

    1) Reranker=False (по умолчанию):
       FAISS top-K (при наличии chunk_filter — с oversample).
       score — cosine из FAISS, шкала примерно [-1, 1].

    2) Reranker задан И query_text задан:
       FAISS top-(top_k * rerank_oversample) → фильтр → reranker → top_k.
       score — от реранкера (logit, шкала зависит от модели).
       Поле `reranked=True` позволяет отличить режимы.

    Args:
        query:            эмбеддинг запроса (для FAISS).
        query_text:       исходный текст запроса (для реранкера).
                          Обязателен, если передан reranker.
        reranker:         любой объект с методом rerank(query, documents, top_n).
        rerank_oversample: во сколько раз расширить FAISS-выборку,
                          чтобы реранкеру было из чего выбирать.
                          Для top_k=3 и rerank_oversample=5 → FAISS вернёт 15.
    """
    if top_k <= 0:
        return []

    use_reranker = reranker is not None and query_text is not None
    if reranker is not None and query_text is None:
        logger.warning(
            "pipeline_search: передан reranker, но не передан query_text — "
            "реранкер будет проигнорирован."
        )

    # 1. Сколько запросить у FAISS
    if use_reranker:
        search_k = top_k * max(1, rerank_oversample)
    elif chunk_filter is not None:
        search_k = top_k * max(1, oversample)
    else:
        search_k = top_k

    # 2. FAISS
    hits = vec_store.search(query, top_k=search_k, min_score=min_score)
    if not hits:
        return []

    # 3. Достаём чанки (сохраняя порядок FAISS)
    chunk_map = {c.chunk_id: c for c in doc_store.get_chunks(h.chunk_id for h in hits)}
    ordered_chunks: List[Chunk] = []
    for h in hits:
        chunk = chunk_map.get(h.chunk_id)
        if chunk is None:
            logger.warning(
                "pipeline_search: chunk_id=%s отсутствует в DocumentStore "
                "(рассинхронизация индексов?)", h.chunk_id,
            )
            continue
        ordered_chunks.append(chunk)

    # 4. Hard-фильтр (до реранкера — сужаем кандидатов)
    if chunk_filter is not None:
        ordered_chunks = [c for c in ordered_chunks if chunk_filter(c)]

    if not ordered_chunks:
        return []

    # 5. Reranker (soft-ranking)
    if use_reranker:
        reranked = reranker.rerank(
            query=query_text,
            documents=[c.text for c in ordered_chunks],
            top_n=top_k,
        )
        return [
            RetrievedChunk(chunk=ordered_chunks[idx], score=float(score), reranked=True)
            for idx, score in reranked
        ]

    # 6. Fallback: score привязан к chunk_id (корректно при наличии фильтра)
    hit_score = {h.chunk_id: h.score for h in hits}
    return [
        RetrievedChunk(chunk=c, score=hit_score[c.chunk_id], reranked=False)
        for c in ordered_chunks[:top_k]
    ]


def pipeline_search_with_context(
    doc_store: DocumentStore,
    vec_store: VectorStore,
    query: Embedding,
    mode: ContextMode = ContextMode.PARENT,
    window_size: int = 1,
    top_k: int = 5,
    min_score: Optional[float] = None,
    chunk_filter: Optional[Callable[[Chunk], bool]] = None,
    oversample: int = 3,
    reranker: Optional[RerankerLike] = None,
    query_text: Optional[str] = None,
    rerank_oversample: int = 5,
) -> List[ChunkContext]:
    """
    То же, что pipeline_search, но возвращает ChunkContext с поднятым
    контекстом (parent / window) для промпта LLM.
    """
    hits = pipeline_search(
        doc_store, vec_store, query,
        top_k=top_k,
        min_score=min_score,
        chunk_filter=chunk_filter,
        oversample=oversample,
        reranker=reranker,
        query_text=query_text,
        rerank_oversample=rerank_oversample,
    )
    return [
        doc_store.get_chunk_with_context(
            hit.chunk.chunk_id, mode=mode, window_size=window_size
        )
        for hit in hits
    ]
# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def pipeline_fill_store(
    doc_store: DocumentStore,
    vec_store: VectorStore,
    embedder: "EmbeddingWrapper",
    paths: Iterable[Union[Path, str]],
    chunker: Optional["MarkdownChunker"] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    read_encoding: str = "utf-8",
) -> FillStats:
    """
    Читает markdown-документы, режет на чанки, эмбеддит листья
    и складывает в DocumentStore + VectorStore.

    chunk_id выделяются через doc_store.next_chunk_id — счётчик
    живёт внутри DocumentStore и синхронизируется в add_chunks().
    document_id разрешается через doc_store.resolve_document_id()
    (при коллизии stem файла будет добавлен суффикс _1, _2, ...).

    Атомарность на уровне документа: commit в оба store'а только
    после успешного embedding. Если embedding упал — ничего не
    добавлено, счётчик не сдвинулся.
    """
    if chunker is None:
        from .Chunking import MarkdownChunker
        chunker = MarkdownChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    stats = FillStats()

    for path_like in paths:
        path = Path(path_like)

        if not path.exists() or not path.is_file():
            logger.warning("pipeline_fill_store: не файл: %s", path)
            stats.skipped.append((str(path), "not a file"))
            continue

        # --- 1. Чтение ---
        try:
            text = path.read_text(encoding=read_encoding)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("pipeline_fill_store: read error %s: %s", path, e)
            stats.skipped.append((str(path), f"read error: {e}"))
            continue

        if not text.strip():
            stats.skipped.append((str(path), "empty"))
            continue

        # --- 2. document_id с разрешением коллизий ---
        preferred_id = path.stem
        document_id = doc_store.resolve_document_id(preferred_id)
        if document_id != preferred_id:
            logger.info(
                "pipeline_fill_store: коллизия document_id: %s -> %s",
                preferred_id, document_id,
            )

        # --- 3. Чанкинг (в памяти) ---
        # start_id берём из счётчика doc_store — это ключевой момент
        try:
            chunks, leaf_ids, _ = chunker.chunk(
                markdown_text=text,
                document_id=document_id,
                start_id=doc_store.next_chunk_id,
            )
        except Exception as e:
            logger.exception("pipeline_fill_store: chunking error %s", path)
            stats.skipped.append((str(path), f"chunking error: {e}"))
            continue

        if not chunks:
            stats.skipped.append((str(path), "no chunks"))
            continue

        leaf_set = set(leaf_ids)
        leaf_chunks = [c for c in chunks if c.chunk_id in leaf_set]
        if not leaf_chunks:
            stats.skipped.append((str(path), "no leaves"))
            continue

        # --- 4. Embedding (в памяти) ---
        try:
            embeddings = embedder._get_text_embedding_batch(
                [c.embedding_text  for c in leaf_chunks]
            )
        except Exception as e:
            logger.exception("pipeline_fill_store: embedding error %s", path)
            stats.skipped.append((str(path), f"embedding error: {e}"))
            continue

        if len(embeddings) != len(leaf_chunks):
            stats.skipped.append((
                str(path),
                f"embedding count mismatch: {len(embeddings)} vs {len(leaf_chunks)}",
            ))
            continue

        # --- 5. Commit (атомарно на уровне документа) ---
        try:
            document = Document(
                document_id=document_id,
                filename=path.name,
                relative_path=_relative_to(path, doc_store.documents_root),
                mime_type=_guess_mime(path),
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
            doc_store.add_document(document)
            doc_store.add_chunks(chunks)             
            vec_store.add(
                embedding=embeddings,
                idx=[c.chunk_id for c in leaf_chunks],
            )
        except Exception as e:
            logger.exception("pipeline_fill_store: commit error %s", path)
            stats.skipped.append((str(path), f"commit error: {e}"))
            continue

        stats.documents_added += 1
        stats.chunks_added += len(chunks)
        stats.leaves_indexed += len(leaf_chunks)

        logger.info(
            "pipeline_fill_store: [%s] +%d chunks, +%d leaves",
            document_id, len(chunks), len(leaf_chunks),
        )

    logger.info("pipeline_fill_store done: %s", stats.summary())
    return stats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def pipeline_save(
    doc_store: DocumentStore,
    vec_store: VectorStore,
    storage_dir: Union[Path, str],
) -> None:
    """Сохраняет оба хранилища в одну директорию."""
    doc_store.save(storage_dir)
    vec_store.save(storage_dir)


def pipeline_load(
    storage_dir: Union[Path, str],
    documents_root: Union[Path, str],
) -> Tuple[DocumentStore, VectorStore]:
    """Загружает оба хранилища из одной директории."""
    doc_store = DocumentStore.load(
        storage_dir=storage_dir, documents_root=documents_root
    )
    vec_store = VectorStore.load(storage_dir=storage_dir)
    return doc_store, vec_store


# ---------------------------------------------------------------------------
# Convenience: full rebuild from a directory
# ---------------------------------------------------------------------------

def pipeline_build_from_directory(
    raw_dir: Union[Path, str],
    index_dir: Union[Path, str],
    embedder: "EmbeddingWrapper",
    chunker: Optional["MarkdownChunker"] = None,
    pattern: str = "**/*.md",
    save: bool = True,
) -> FillStats:
    """
    Полная пересборка индекса из директории. Удобно для prepare_index.py.

    Порядок:
      1. Создаём пустые DocumentStore + VectorStore.
      2. Определяем размерность через probe-эмбеддинг.
      3. Прогоняем файлы через pipeline_fill_store.
      4. Сохраняем в index_dir.
    """
    raw_dir = Path(raw_dir)
    index_dir = Path(index_dir)

    doc_store = DocumentStore(documents_root=raw_dir)

    # Размерность определяем через probe-эмбеддинг.
    probe = embedder._get_text_embedding("probe")
    vec_store = VectorStore(dim=probe.size)

    paths = sorted(raw_dir.glob(pattern))
    logger.info(
        "pipeline_build_from_directory: %d файлов в %s", len(paths), raw_dir
    )

    stats = pipeline_fill_store(
        doc_store, vec_store, embedder, paths, chunker=chunker,
    )

    if save:
        pipeline_save(doc_store, vec_store, index_dir)
        logger.info("pipeline_build_from_directory: сохранено в %s", index_dir)

    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative_to(path: Path, root: Path) -> str:
    """Относительный путь от root; fallback — имя файла."""
    try:
        return str(path.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return path.name


def _guess_mime(path: Path) -> str:
    return {
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
        ".docx": (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    }.get(path.suffix.lower(), "application/octet-stream")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manual run / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Запуск:  python -m RAG.pipelines
    import sys
    import time

    DATA_DIR = Path(
            r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Store\_test_\Data Knowledge"
    )
    INDEX_DIR = Path(
        r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Store\Storage"
    )

    # ---------- 1. Проверка входной папки ----------
    if not DATA_DIR.exists():
        print(f"❌ Не найдена папка с данными:\n   {DATA_DIR}")
        sys.exit(1)

    md_files = sorted(DATA_DIR.rglob("*.md"))
    print(f"📂 Папка: {DATA_DIR}")
    print(f"   .md файлов: {len(md_files)}")

    if not md_files:
        print("❌ В папке нет .md файлов.")
        sys.exit(1)

    # ---------- 1.5. Диагностика: есть ли «АСКУЭ» в базе ----------
    print("\n🔎 Проверка термина «АСКУЭ» в базе знаний:")
    hits_askue = 0
    for f in md_files:
        n = f.read_text(encoding="utf-8").lower().count("аскуэ")
        if n:
            hits_askue += n
            print(f"   • {f.name}: {n} вхождений")
    if hits_askue == 0:
        print("   ⚠️ Термина «АСКУЭ» в базе нет — retrieval его не найдёт.")
        print("   ⚠️ Стоит ожидать семантический поиск через «облачная платформа учёта».")
    else:
        print(f"   ✅ Всего вхождений: {hits_askue}")

    # ---------- 2. Embedder ----------
    print("\n🔧 Загрузка EmbeddingWrapper...")
    from .Embedding import EmbeddingWrapper  # noqa: E402

    try:
        embedder = EmbeddingWrapper()
    except FileNotFoundError as e:
        print(f"❌ Не удалось загрузить модель: {e}")
        sys.exit(1)

    probe_dim = embedder._get_text_embedding("probe").size
    print(f"   Модель загружена, размерность: {probe_dim}")

    # ---------- 3. Сборка индекса ----------
    print(f"\n🚀 Индексирование в {INDEX_DIR}")
    stats = pipeline_build_from_directory(
        raw_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        embedder=embedder,
        pattern="**/*.md",
        save=True,
    )
    print(f"✅ {stats.summary()}")
    if stats.skipped:
        for p, reason in stats.skipped:
            print(f"   ⚠️ {Path(p).name}: {reason}")

    # ---------- 4. Загрузка ----------
    print("\n🔍 Загрузка индекса с диска...")
    doc_store, vec_store = pipeline_load(INDEX_DIR, DATA_DIR)
    n_docs = len(list(doc_store.documents))
    n_chunks = len(doc_store)
    n_leaves = len(doc_store.leaf_chunks)
    n_vectors = len(vec_store)
    print(f"   Документов:       {n_docs}")
    print(f"   Чанков:           {n_chunks} "
          f"(родителей {n_chunks - n_leaves}, листьев {n_leaves})")
    print(f"   Векторов в FAISS: {n_vectors}")
    print(f"   next_chunk_id:    {doc_store.next_chunk_id}")
    assert n_vectors == n_leaves, (
        f"❌ Рассинхронизация: {n_vectors} векторов vs {n_leaves} листьев"
    )
    print("   ✅ Векторы ↔ листья синхронизированы")

    # ---------- 5. Локальный реранкер ----------
    print("\n🎯 Загрузка LocalReranker...")
    reranker = None
    try:
        from .Ranking import LocalReranker  # noqa: E402
        t0 = time.perf_counter()
        reranker = LocalReranker(device="cpu")
        print(f"   ✅ Загружен за {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"   ⚠️ Не удалось загрузить реранкер: {e}")
        print("   ⚠️ Дальше — режим без реранкера.")

    # ---------- 6. Тестовые запросы ----------
    # Сначала «проблемный» АСКУЭ, потом реальные из demo_questions.csv
    test_queries = [
        "Что такое АСКУЭ и как подключить счётчики?",
        "У нас несколько магазинов. Можно смотреть их расход в одном кабинете?",
        "В счётчиках стоят симки другого оператора. Для вашей платформы их надо менять?",
        "Хочу видеть расход электричества отдельно по цехам.",
    ]

    def _print_hits(title: str, hits: List[RetrievedChunk]) -> None:
        print(f"\n   {title}:")
        if not hits:
            print("      ⚠️ пусто")
            return
        for i, h in enumerate(hits, 1):
            section = h.chunk.section_title or "—"
            preview = h.chunk.text[:100].replace("\n", " ").strip()
            tag = "rerank" if h.reranked else "cos"
            print(f"      {i}. [{tag:6s} {h.score:+.3f}] "
                  f"{h.chunk.document_id} / {section}")
            print(f"         {preview}...")

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"❓ {q}")
        q_emb = embedder._get_query_embedding(q)

        # Без реранкера: FAISS top-5
        t0 = time.perf_counter()
        base_hits = pipeline_search(
            doc_store, vec_store, q_emb,
            top_k=5, min_score=0.0,
        )
        dt = (time.perf_counter() - t0) * 1000
        _print_hits(f"FAISS top-5 ({dt:.0f} ms)", base_hits)

        # С реранкером
        if reranker is not None:
            t0 = time.perf_counter()
            rer_hits = pipeline_search(
                doc_store, vec_store, q_emb,
                top_k=5, min_score=0.0,
                reranker=reranker,
                query_text=q,
                rerank_oversample=5,  # FAISS вернёт 25, реранкер сузит до 5
            )
            dt = (time.perf_counter() - t0) * 1000
            _print_hits(f"Reranker top-5 ({dt:.0f} ms)", rer_hits)

            # Показать, куда переехал топ-1 из FAISS
            if base_hits and rer_hits:
                faiss_top = base_hits[0].chunk.chunk_id
                positions = [i for i, h in enumerate(rer_hits, 1)
                             if h.chunk.chunk_id == faiss_top]
                pos = positions[0] if positions else "вне топ-5"
                print(f"\n   ℹ️ FAISS-топ1 (id={faiss_top}) "
                      f"после реранкера: {pos}")

    # ---------- 7. Small-to-Big (пример с parent) ----------
    print("\n" + "=" * 70)
    print("🔗 Small-to-Big (PARENT) — на запросе про АСКУЭ:")
    q = "Что такое АСКУЭ и как подключить счётчики?"
    q_emb = embedder._get_query_embedding(q)
    ctxs = pipeline_search_with_context(
        doc_store, vec_store, q_emb,
        mode=ContextMode.PARENT, top_k=3,
        reranker=reranker, query_text=q,
    )
    for i, ctx in enumerate(ctxs, 1):
        has_parent = "да" if ctx.parent is not None else "нет"
        print(
            f"   {i}. chunk_id={ctx.chunk.chunk_id} "
            f"({ctx.chunk.document_id}/{ctx.chunk.section_title or '—'}), "
            f"parent={has_parent}, "
            f"context_len={len(ctx.context_text)}"
        )

    print("\n🎉 Готово.")