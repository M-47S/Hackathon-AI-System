# GRADER/pipelines.py
"""
RAG/pipelines.py — сценарный слой RAG.

Связывает вместе:
  - MarkdownChunker (ленивый импорт — не тянем llama_index в solution.py)
  - EmbedderLike    (duck-typing: EmbeddingWrapper локально, RemoteEmbedder на платформе)
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
    runtime_checkable,
)

from structs import (
    Chunk,
    ChunkContext,
    ContextMode,
    Document,
    Embedding,
    VectorHit,
)
from vector_store import VectorStore
from document_store import DocumentStore

if TYPE_CHECKING:
    from chunker import MarkdownChunker


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
    Duck-type контракт для эмбеддеров.
    Ему соответствуют:
      - EmbeddingWrapper (локальная HF-модель, для dev)
      - RemoteEmbedder   (OpenAI-совместимый прокси, для платформы)
    Никаких импортов конкретных классов здесь не нужно.
    """
    def _get_text_embedding(self, text: str) -> Embedding: ...
    def _get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[Embedding]: ...
    def _get_query_embedding(self, query: str) -> Embedding: ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")

# Подсказки «продуктового» запроса — для них поднимаем лимит на документ,
# чтобы целевой документ был представлен 3-4 чанками (важно для Small-to-Big).
_PRODUCT_HINTS = (
    "аскуэ", "оптов", "солнечн", "потер", "распознаван",
    "фото", "платформ", "учёт", "учет", "счётчик", "счетчик",
    "генерац", "оре", "опт ", "оре ",
)


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
                           Работает всегда, независимо от реранкера.
    """
    if top_k <= 0:
        return []

    use_reranker = reranker is not None and query_text is not None
    if reranker is not None and query_text is None:
        logger.warning(
            "pipeline_search: передан reranker, но не передан query_text — "
            "реранкер будет проигнорирован."
        )

    if max_per_document is None:
        max_per_document = _adaptive_max_per_document(query_text or "")

    # 1. Сколько запросить у FAISS
    if use_reranker:
        search_k = top_k * max(1, rerank_oversample)
    elif chunk_filter is not None or filter_toc:
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

    # 4a. Hard-фильтр
    if chunk_filter is not None:
        ordered_chunks = [c for c in ordered_chunks if chunk_filter(c)]

    # 4b. TOC-фильтр
    if filter_toc:
        ordered_chunks = [c for c in ordered_chunks if not _is_toc_like(c)]

    if not ordered_chunks:
        return []

    # 5. Диверсификация
    ordered_chunks = _diversify_by_document(
        ordered_chunks,
        max_per_document=max_per_document,
        min_required=0 if use_reranker else top_k,
    )

    # 6. Reranker
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

    # 7. Fallback
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
) -> List[ChunkContext]:
    """То же, что pipeline_search, но с контекстом (parent / window)."""
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
    embedder: EmbedderLike,          # ← было: "EmbeddingWrapper"
    paths: Iterable[Union[Path, str]],
    chunker: Optional["MarkdownChunker"] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    read_encoding: str = "utf-8",
) -> FillStats:
    """
    Читает markdown-документы, режет на чанки, эмбеддит листья
    и складывает в DocumentStore + VectorStore.

    embedder — любой объект с методами _get_text_embedding / 
    _get_text_embedding_batch / _get_query_embedding:
      - EmbeddingWrapper  (локальная HF-модель, dev)
      - RemoteEmbedder    (прокси bge-m3, платформа)
    Тип не важен — duck-typing через протокол EmbedderLike.
    """
    if chunker is None:
        from chunker import MarkdownChunker
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

        # --- 2. document_id ---
        preferred_id = path.stem
        document_id = doc_store.resolve_document_id(preferred_id)
        if document_id != preferred_id:
            logger.info(
                "pipeline_fill_store: коллизия document_id: %s -> %s",
                preferred_id, document_id,
            )

        # --- 3. Чанкинг ---
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

        # --- 4. Embedding ---
        # embedding_text может быть None (старый индекс / баг chunker'а) —
        # fallback на обычный text.
        texts_to_embed = [
            (c.embedding_text or c.text) for c in leaf_chunks
        ]
        try:
            embeddings = embedder._get_text_embedding_batch(texts_to_embed)
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

        # --- 5. Commit ---
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
    doc_store.save(storage_dir)
    vec_store.save(storage_dir)


def pipeline_load(
    storage_dir: Union[Path, str],
    documents_root: Union[Path, str],
) -> Tuple[DocumentStore, VectorStore]:
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
    embedder: EmbedderLike,          # ← было: "EmbeddingWrapper"
    chunker: Optional["MarkdownChunker"] = None,
    pattern: str = "**/*.md",
    save: bool = True,
) -> FillStats:
    raw_dir = Path(raw_dir)
    index_dir = Path(index_dir)

    doc_store = DocumentStore(documents_root=raw_dir)

    # Размерность через probe-эмбеддинг — работает и для RemoteEmbedder.
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
    """Ограничивает число чанков с одного document_id."""
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


def _is_toc_like(chunk: Chunk, threshold: float = 0.35, min_links: int = 2) -> bool:
    """
    True, если чанк — в основном markdown-ссылки (оглавление, навигация).
    Такие чанки плохи для RAG: много ключевых слов, но нет содержания.
    """
    text = chunk.text
    if not text:
        return True
    links = list(_LINK_RE.finditer(text))

    # ≥2 ссылок — это точно навигация
    if len(links) >= min_links:
        return True

    link_chars = sum(len(m.group(0)) for m in links)
    return link_chars / len(text) > threshold


def _adaptive_max_per_document(query_text: str, default: int = 2) -> int:
    """
    Для «продуктовых» запросов лимит выше (4), чтобы целевой документ
    был представлен несколькими чанками (нужно для Small-to-Big).
    Для общих запросов — default (2).
    """
    q = (query_text or "").lower()
    if any(hint in q for hint in _PRODUCT_HINTS):
        return 4
    return default