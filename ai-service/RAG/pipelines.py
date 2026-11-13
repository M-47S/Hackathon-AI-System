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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    Any,
    runtime_checkable,
)

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

@runtime_checkable
class EmbedderLike(Protocol):
    """
    Duck-type контракт для эмбеддеров'.
    Ему соответствует EmbeddingWrapper.
    """
    def _get_query_embedding(
        self, 
        query: str
    ) -> Embedding: ...
    
    def _get_query_embedding_batch(
            self, 
            queries: List[str]
    ) -> List[Embedding]: ...
    
    def _get_text_embedding(
        self, 
        text: str
    ) -> Embedding: ...
    
    def _get_text_embedding_batch(
        self, 
        texts: List[str]
    ) -> List[Embedding]: ...

@runtime_checkable
class ChunkerLike(Protocol):
    """Duck-type контракт для чанкеров. 
    Соответствует MarkdownChunker.
    """
    def chunk(
        self,
        markdown_text: str,
        document_id: str,
        start_id: int = 0,
    ) -> Tuple[List[Chunk], List[int], int]: ...

    

_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")

# Подсказки «продуктового» запроса — для них поднимаем лимит на документ,
# чтобы целевой документ был представлен 3-4 чанками (важно для Small-to-Big).
_PRODUCT_HINTS = (
    "аскуэ", "оптов", "солнечн", "потер", "распознаван",
    "фото", "платформ", "учёт", "учет", "счётчик", "счетчик",
    "генерац", "оре", "опт ", "оре ",
)


QUERY_INSTRUCTIONS: dict[str, str] = {
    "none": "",
    "short_ru": "Поиск в базе знаний по энергетике и продуктам МТС EnergyTool: ",
    "detailed_ru": (
        "Найди в базе знаний МТС EnergyTool информацию, "
        "которая отвечает на вопрос клиента: "
    ),
    "bge_official": "Represent this sentence for searching relevant passages: ",
}


DEFAULT_QUERY_INSTRUCTION = QUERY_INSTRUCTIONS["detailed_ru"]

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
    oversample: int = 5,
    reranker: Optional[RerankerLike] = None,
    query_text: Optional[str] = None,
    rerank_oversample: int = 5,
    max_per_document: Optional[int] = None,
    filter_toc: bool = True,
    exclude_documents: Optional[set] = None,   # ← NEW из Grader
) -> List[RetrievedChunk]:
    """
    Возвращает top_k релевантных чанков.

    Два режима ранжирования:

    1) Reranker=False (по умолчанию):
       FAISS top-K (при наличии chunk_filter — с oversample).
       score — cosine из FAISS, шкала примерно [-1, 1].

    2) Reranker задан И query_text задан:
       FAISS top-(top_k * rerank_oversample) → фильтр → TOC-фильтр
       → диверсификация → reranker → top_k.
       score — от реранкера (logit, шкала зависит от модели).
       Поле `reranked=True` позволяет отличить режимы.

    Args:
        query:             эмбеддинг запроса (для FAISS).
        query_text:        исходный текст запроса (для реранкера и adaptive).
        reranker:          любой объект с методом rerank(query, documents, top_n).
        rerank_oversample: во сколько раз расширить FAISS-выборку.
        max_per_document:  не более N чанков с одного document_id.
                           None — авто (2 для общих, 4 для продуктовых).
                           0 — отключить диверсификацию.
        filter_toc:        отсекать чанки-оглавления (markdown-ссылки).
        exclude_documents: set document_id, которые нужно исключить
                           из выдачи (для шумных обзоров вроде «20-portfolio»).
    """
    if top_k <= 0:
        return []

    use_reranker = reranker is not None and query_text is not None
    if reranker is not None and query_text is None:
        logger.warning(
            "pipeline_search: передан reranker, но не передан query_text — "
            "реранкер будет проигнорирован."
        )

    # Авто-выбор лимита на документ, если явно не задан
    if max_per_document is None:
        max_per_document = _adaptive_max_per_document(query_text or "")

    # 1. Сколько запросить у FAISS
    if use_reranker:
        search_k = top_k * max(1, rerank_oversample)
    elif chunk_filter is not None or filter_toc or exclude_documents:
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

    # 4a. Hard-фильтр (пользовательский)
    if chunk_filter is not None:
        ordered_chunks = [c for c in ordered_chunks if chunk_filter(c)]

    # 4b. TOC-фильтр (автоматический): отсекает чанки-оглавления
    if filter_toc:
        ordered_chunks = [c for c in ordered_chunks if not _is_toc_like(c)]

    # 4c. Blacklist документов (шумные обзоры)
    if exclude_documents:
        ordered_chunks = [
            c for c in ordered_chunks if c.document_id not in exclude_documents
        ]

    if not ordered_chunks:
        return []

    # 5. Диверсификация по document_id
    ordered_chunks = _diversify_by_document(
        ordered_chunks,
        max_per_document=max_per_document,
        min_required=0 if use_reranker else top_k,
    )

    # 6. Reranker (soft-ranking) с graceful fallback
    if use_reranker:
        reranked = []
        try:
            reranked = reranker.rerank(
                query=query_text,
                documents=[c.text for c in ordered_chunks],
                top_n=top_k,
            )
        except Exception as e:
            logger.warning("Reranker упал (%s), fallback на FAISS scores", e)

        if reranked:
            # Защита от мусорных индексов от реранкера
            return [
                RetrievedChunk(chunk=ordered_chunks[idx], score=float(score), reranked=True)
                for idx, score in reranked
                if 0 <= idx < len(ordered_chunks)
            ]

        # FALLBACK: FAISS scores
        hit_score = {h.chunk_id: h.score for h in hits}
        return [
            RetrievedChunk(chunk=c, score=hit_score[c.chunk_id], reranked=False)
            for c in ordered_chunks[:top_k]
        ]

    # 7. Fallback: score по chunk_id
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
    max_per_document: Optional[int] = None,
    filter_toc: bool = True,
    exclude_documents: Optional[set] = None, 
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
        max_per_document=max_per_document,
        filter_toc=filter_toc,
        exclude_documents=exclude_documents,
    )
    return [
        doc_store.get_chunk_with_context(
            hit.chunk.chunk_id, mode=mode, window_size=window_size
        )
        for hit in hits
    ]


def pipeline_search_by_text(
    embedder: EmbedderLike,
    doc_store: DocumentStore,
    vec_store: VectorStore,
    query_text: str,
    query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    rerank_query_text: Optional[str] = None,   # ← NEW из Grader
    **search_kwargs,
) -> List[RetrievedChunk]:
    """
    Высокоуровневая обёртка над pipeline_search.

    Разделение запросов:
      - эмбеддинг:   instruction + query_text  (богатый контекст для bge-m3)
      - реранкер:    rerank_query_text, если задан, иначе query_text
                     (чистая формулировка пользователя для cross-encoder)

    Разделение нужно, когда query_text — «расширенный» LLM-запрос
    (с раскрытыми аббревиатурами), а реранкер лучше работает
    с естественной формулировкой.
    """
    text_for_embedding = (
        f"{query_instruction}{query_text}" if query_instruction else query_text
    )
    query_embedding = embedder._get_query_embedding(text_for_embedding)
    return pipeline_search(
        doc_store=doc_store,
        vec_store=vec_store,
        query=query_embedding,
        query_text=rerank_query_text if rerank_query_text is not None else query_text,
        **search_kwargs,
    )


def pipeline_search_with_context_by_text(
    embedder: EmbedderLike,
    doc_store: DocumentStore,
    vec_store: VectorStore,
    query_text: str,
    query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    rerank_query_text: Optional[str] = None,   # ← NEW из Grader
    **search_kwargs,
) -> List[ChunkContext]:
    """Аналог pipeline_search_by_text для Small-to-Big (ChunkContext)."""
    text_for_embedding = (
        f"{query_instruction}{query_text}" if query_instruction else query_text
    )
    query_embedding = embedder._get_query_embedding(text_for_embedding)
    return pipeline_search_with_context(
        doc_store=doc_store,
        vec_store=vec_store,
        query=query_embedding,
        query_text=rerank_query_text if rerank_query_text is not None else query_text,
        **search_kwargs,
    )

# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------

def pipeline_fill_store(
    doc_store: DocumentStore,
    vec_store: VectorStore,
    embedder: EmbedderLike,
    paths: Iterable[Union[Path, str]],
    chunker: Optional[ChunkerLike] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    read_encoding: str = "utf-8",
) -> FillStats:
    """
    Читает markdown-документы, режет на чанки, эмбеддит листья
    и складывает в DocumentStore + VectorStore.

    Атомарность на уровне документа: commit в оба store'а только
    после успешного embedding.
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
                [c.embedding_text for c in leaf_chunks]
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
    embedder: EmbedderLike,
    chunker: Optional[ChunkerLike] = None,
    pattern: str = "**/*.md",
    save: bool = True,
) -> FillStats:
    """
    Полная пересборка индекса из директории. Удобно для prepare_index.py.
    """
    raw_dir = Path(raw_dir)
    index_dir = Path(index_dir)

    doc_store = DocumentStore(documents_root=raw_dir)
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


def _diversify_by_document(
    chunks: List[Chunk],
    max_per_document: int = 2,
    min_required: int = 0,
) -> List[Chunk]:
    """
    Ограничивает число чанков с одного document_id.

    Двухуровневая логика:
      1) «Ядро» — по max_per_document чанков с каждого документа.
      2) Если ядра меньше min_required — добираем из «переполнения».
    """
    if max_per_document <= 0 or not chunks:
        return chunks

    primary: List[Chunk] = []
    overflow: List[Chunk] = []
    counts: dict[str, int] = {}

    for c in chunks:
        n = counts.get(c.document_id, 0)
        if n < max_per_document:
            counts[c.document_id] = n + 1
            primary.append(c)
        else:
            overflow.append(c)

    if min_required > 0 and len(primary) < min_required:
        need = min_required - len(primary)
        primary.extend(overflow[:need])

    return primary


def _is_toc_like(chunk: Chunk, threshold: float = 0.65) -> bool:   # ← 0.5 → 0.65
    """
    True, если чанк — в основном markdown-ссылки (оглавление, навигация).

    Дополнительно: короткие чанки (<150 символов) НЕ считаются TOC — это
    обычно короткие продуктовые секции (заголовок + 1-2 строки), которые
    baseline-подход сохраняет в выдаче.
    """
    text = chunk.text
    if not text:
        return True
    if len(text) < 150:                # ← NEW из Grader
        return False
    link_chars = sum(len(m.group(0)) for m in _LINK_RE.finditer(text))
    return link_chars / len(text) > threshold


def _adaptive_max_per_document(query_text: str, default: int = 2) -> int:
    """
    Для «продуктовых» запросов лимит 4, для остальных default (2).
    """
    q = (query_text or "").lower()
    if any(hint in q for hint in _PRODUCT_HINTS):
        return 4
    return default


# ---------------------------------------------------------------------------
# Manual run / smoke test — оставляем как есть
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time
    import statistics

    DATA_DIR = Path(
        r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service"
        r"\RAG\Store\_test_\Data Knowledge"
    )
    INDEX_DIR = Path(
        r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service"
        r"\RAG\Store\Storage"
    )

    if not DATA_DIR.exists():
        print(f"❌ Не найдена папка: {DATA_DIR}")
        sys.exit(1)

    md_files = sorted(DATA_DIR.rglob("*.md"))
    print(f"📂 Папка: {DATA_DIR}")
    print(f"   .md файлов: {len(md_files)}")
    if not md_files:
        sys.exit(1)

    # ---------- Диагностика АСКУЭ ----------
    print("\n🔎 Термин «АСКУЭ» в базе:")
    total_askue = 0
    for f in md_files:
        n = f.read_text(encoding="utf-8").lower().count("аскуэ")
        if n:
            total_askue += n
            print(f"   • {f.name}: {n}")
    if total_askue == 0:
        print("   ⚠️ 0 вхождений — retrieval не найдёт термин буквально,")
        print("   ⚠️ но instruction prefix может помочь семантически.")

    # ---------- Embedder ----------
    print("\n🔧 Загрузка EmbeddingWrapper...")
    from .Embedding import EmbeddingWrapper  # noqa: E402

    try:
        embedder = EmbeddingWrapper()
    except FileNotFoundError as e:
        print(f"❌ Не удалось загрузить модель: {e}")
        sys.exit(1)

    probe_dim = embedder._get_text_embedding("probe").size
    print(f"   dim={probe_dim}")

    # ---------- Индексация ----------
    print(f"\n🚀 Индексирование → {INDEX_DIR}")
    stats = pipeline_build_from_directory(
        raw_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        embedder=embedder,
        pattern="**/*.md",
        save=True,
    )
    print(f"✅ {stats.summary()}")

    # ---------- Загрузка ----------
    doc_store, vec_store = pipeline_load(INDEX_DIR, DATA_DIR)
    n_leaves = len(doc_store.leaf_chunks)
    n_vectors = len(vec_store)
    print(f"   Чанков={len(doc_store)}, листьев={n_leaves}, векторов={n_vectors}")
    assert n_vectors == n_leaves

    # ---------- TOC-диагностика ----------
    toc_chunks = [c for c in doc_store.leaf_chunks if _is_toc_like(c)]
    print(f"\n🧹 TOC-подобных листьев: {len(toc_chunks)} (отсекаются из выдачи)")

    # ---------- Реранкер ----------
    print("\n🎯 Загрузка LocalReranker...")
    reranker = None
    try:
        from .Ranking import LocalReranker  # noqa: E402
        t0 = time.perf_counter()
        reranker = LocalReranker(device="cpu")
        print(f"   ✅ за {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"   ⚠️ не загружен: {e}")

    # ---------- Хелперы печати ----------
    def _docs(hits):
        return [h.chunk.document_id for h in hits]

    def _print_hits(title, hits):
        print(f"\n   {title}:")
        if not hits:
            print("      ⚠️ пусто")
            return
        for i, h in enumerate(hits, 1):
            section = (h.chunk.section_title or "—")[:40]
            preview = h.chunk.text[:90].replace("\n", " ").strip()
            tag = "rerank" if h.reranked else "cos"
            print(f"      {i}. [{tag:6s} {h.score:+.3f}] "
                  f"{h.chunk.document_id} / {section}")
            print(f"         {preview}...")

    # ---------- A/B: инструкции на проблемном запросе ----------
    probe_query = "Что такое АСКУЭ и как подключить счётчики?"
    print("\n" + "=" * 70)
    print(f"🧪 A/B-тест Instruction Prefix")
    print(f"❓ {probe_query}")
    m_probe = _adaptive_max_per_document(probe_query)
    print(f"   adaptive max_per_document={m_probe}")

    doc_sets: dict[str, list[str]] = {}

    for name, instr in QUERY_INSTRUCTIONS.items():
        try:
            hits = pipeline_search_by_text(
                embedder, doc_store, vec_store, probe_query,
                query_instruction=instr,
                top_k=5, min_score=0.0,
                filter_toc=True, max_per_document=m_probe,
            )
        except Exception as e:
            print(f"\n   ❌ {name}: {e}")
            continue
        doc_sets[name] = _docs(hits)
        _print_hits(f"Instruction = {name!r}", hits)

    # ---------- Сводка: какие документы всплыли ----------
    print("\n   📊 Итог по документам:")
    for name, docs in doc_sets.items():
        print(f"      {name:12s} → {docs}")

    # ---------- A/B + reranker (только для detailed_ru) ----------
    if reranker is not None:
        print("\n" + "-" * 70)
        print("🎯 С реранкером (instruction = 'detailed_ru'):")
        t0 = time.perf_counter()
        hits_rer = pipeline_search_by_text(
            embedder, doc_store, vec_store, probe_query,
            query_instruction=QUERY_INSTRUCTIONS["detailed_ru"],
            top_k=5, min_score=0.0,
            filter_toc=True, max_per_document=m_probe,
            reranker=reranker, rerank_oversample=5,
        )
        dt = (time.perf_counter() - t0) * 1000
        _print_hits(f"Reranker ({dt:.0f} ms)", hits_rer)
        print(f"\n      Документы: {_docs(hits_rer)}")

    # ---------- Прогон на реальных запросах ----------
    real_queries = [
        "У нас несколько магазинов. Можно смотреть их расход в одном кабинете?",
        "В счётчиках стоят симки другого оператора. Для нас это проблема?",
        "Хочу видеть расход электричества отдельно по цехам.",
    ]

    print("\n" + "=" * 70)
    print("📋 Реальные запросы (instruction = 'detailed_ru')")

    timings: list[float] = []
    for q in real_queries:
        print(f"\n❓ {q}")
        t0 = time.perf_counter()
        hits = pipeline_search_by_text(
            embedder, doc_store, vec_store, q,
            query_instruction=QUERY_INSTRUCTIONS["detailed_ru"],
            top_k=5, min_score=0.0,
            filter_toc=True,
            reranker=reranker, rerank_oversample=5,
        )
        dt = (time.perf_counter() - t0) * 1000
        timings.append(dt)
        _print_hits(f"top-5 ({dt:.0f} ms)", hits)

    if timings:
        print(f"\n📊 Тайминги: median={statistics.median(timings):.0f} ms, "
              f"на 100 запросов (4 воркера) ≈ "
              f"{statistics.median(timings) * 100 / 4 / 1000:.1f} с")

    print("\n🎉 Готово.")