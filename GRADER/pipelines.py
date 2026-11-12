"""
pipelines.py — сценарный слой RAG (для платформы Hack&Change).

Кладётся в КОРЕНЬ ZIP вместе с solution.py, structs.py, document_store.py,
vector_store.py, chunker.py, llm_client.py, rag_embed.py, rag_rank.py.

Все импорты — ПЛОСКИЕ (без относительных путей). Зависимости от
sentence-transformers, torch, llama-index ОТСУТСТВУЮТ: на платформе
эмбеддинги и реранкер идут через OpenAI-совместимый прокси.

Основные функции:
  pipeline_fill_store(...)                -> FillStats
  pipeline_search(...)                    -> List[RetrievedChunk]
  pipeline_search_with_context(...)       -> List[ChunkContext]
  pipeline_search_by_text(...)            -> List[RetrievedChunk]         ← Instruction Prefix
  pipeline_search_with_context_by_text(...) -> List[ChunkContext]         ← Instruction Prefix
  pipeline_save / pipeline_load           — персистентность
  pipeline_build_from_directory           — one-shot пересборка
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

from structs import Chunk, ChunkContext, ContextMode, Document, Embedding, VectorHit
from document_store import DocumentStore
from vector_store import VectorStore

if TYPE_CHECKING:
    from chunker import MarkdownChunker
    # Embedder не импортируем — используем duck-typing:
    # любой объект с методами _get_text_embedding / _get_query_embedding /
    # _get_text_embedding_batch подходит (RemoteEmbedder или EmbeddingWrapper).

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    """Результат поиска: чанк + его релевантность."""
    chunk: Chunk
    score: float
    reranked: bool = False


@dataclass
class FillStats:
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
    """Duck-type контракт: подходит QwenReranker и любой другой."""
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]: ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")

# Подсказки «продуктового» запроса: для них лимит на документ выше,
# чтобы целевой документ был представлен 3–4 чанками (важно для Small-to-Big).
_PRODUCT_HINTS = (
    "аскуэ", "оптов", "солнечн", "потер", "распознаван",
    "фото", "платформ", "учёт", "учет", "счётчик", "счетчик",
    "генерац", "оре", "опт ", "оре ",
)

# Instruction Prefix (retrieval instructions).
# Применяется ТОЛЬКО в _by_text-версиях: инструкция идёт в эмбеддинг,
# а в реранкер передаётся ЧИСТЫЙ query_text (без префикса).
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
    oversample: int = 3,
    reranker: Optional[RerankerLike] = None,
    query_text: Optional[str] = None,
    rerank_oversample: int = 3,
    max_per_document: Optional[int] = None,
    filter_toc: bool = True,
    exclude_documents: Optional[set] = None,
) -> List[RetrievedChunk]:
    """
    Возвращает top_k релевантных чанков.

    Режим 1 (reranker=None): score = cosine из FAISS, шкала ≈ [-1, 1].
    Режим 2 (reranker + query_text): реранкер переупорядочивает кандидатов,
        score — logit реранкера, `reranked=True`.

    Args:
        exclude_documents: set document_id, которые нужно исключить
            из выдачи. Полезно для шумных обзорных документов
            (например, "20-portfolio"), если они лезут во все запросы.
    """
    if top_k <= 0:
        return []

    use_reranker = reranker is not None and query_text is not None
    if reranker is not None and query_text is None:
        logger.warning(
            "pipeline_search: передан reranker без query_text — "
            "реранкер проигнорирован."
        )

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

    # 3. Достаём чанки
    chunk_map = {c.chunk_id: c for c in doc_store.get_chunks(h.chunk_id for h in hits)}
    ordered_chunks: List[Chunk] = []
    for h in hits:
        chunk = chunk_map.get(h.chunk_id)
        if chunk is None:
            logger.warning(
                "pipeline_search: chunk_id=%s отсутствует в DocumentStore", h.chunk_id
            )
            continue
        ordered_chunks.append(chunk)

    # 4a. Пользовательский hard-фильтр
    if chunk_filter is not None:
        ordered_chunks = [c for c in ordered_chunks if chunk_filter(c)]

    # 4b. TOC-фильтр
    if filter_toc:
        ordered_chunks = [c for c in ordered_chunks if not _is_toc_like(c)]

    # 4c. Blacklist документов (для шумных обзоров типа 20-portfolio)
    if exclude_documents:
        ordered_chunks = [
            c for c in ordered_chunks if c.document_id not in exclude_documents
        ]

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
    """То же, что pipeline_search, но возвращает ChunkContext (Small-to-Big)."""
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
    embedder,
    doc_store: DocumentStore,
    vec_store: VectorStore,
    query_text: str,
    query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    rerank_query_text: Optional[str] = None,   # ← NEW
    **search_kwargs,
) -> List[RetrievedChunk]:
    """
    Instruction Prefix + поиск.

    - Эмбеддинг:   instruction + query_text
    - Рерэнкер:    rerank_query_text, если задан, иначе query_text

    Разделение нужно, когда query_text — «расширенный» LLM-запрос
    (с раскрытыми аббревиатурами), а реранкер лучше работает
    с естественной формулировкой пользователя.
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
    embedder,
    doc_store: DocumentStore,
    vec_store: VectorStore,
    query_text: str,
    query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    rerank_query_text: Optional[str] = None,   # ← NEW
    **search_kwargs,
) -> List[ChunkContext]:
    """
    Instruction Prefix + Small-to-Big.

    - Эмбеддинг:   instruction + query_text
    - Рерэнкер:    rerank_query_text, если задан, иначе query_text
    """
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
    embedder,
    paths: Iterable[Union[Path, str]],
    chunker: Optional["MarkdownChunker"] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    read_encoding: str = "utf-8",
) -> FillStats:
    """
    Читает .md, режет на чанки, эмбеддит листья, пишет в оба store'а.
    Атомарность на уровне документа: commit только после успешного embedding.
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

        try:
            text = path.read_text(encoding=read_encoding)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("pipeline_fill_store: read error %s: %s", path, e)
            stats.skipped.append((str(path), f"read error: {e}"))
            continue

        if not text.strip():
            stats.skipped.append((str(path), "empty"))
            continue

        preferred_id = path.stem
        document_id = doc_store.resolve_document_id(preferred_id)
        if document_id != preferred_id:
            logger.info(
                "pipeline_fill_store: коллизия document_id: %s -> %s",
                preferred_id, document_id,
            )

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
# Convenience: full rebuild
# ---------------------------------------------------------------------------

def pipeline_build_from_directory(
    raw_dir: Union[Path, str],
    index_dir: Union[Path, str],
    embedder,
    chunker: Optional["MarkdownChunker"] = None,
    pattern: str = "**/*.md",
    save: bool = True,
) -> FillStats:
    """Полная пересборка индекса из директории."""
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
    """Ограничивает N чанков на document_id, чтобы длинный документ не доминировал."""
    if max_per_document <= 0 or not chunks:
        return chunks

    primary: List[Chunk] = []
    overflow: List[Chunk] = []
    counts: dict = {}

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


def _is_toc_like(chunk: Chunk, threshold: float = 0.65) -> bool: 
    text = chunk.text
    if not text:
        return True
    # Дополнительно: короткие чанки (<150 символов) не считаем TOC
    if len(text) < 150:
        return False
    link_chars = sum(len(m.group(0)) for m in _LINK_RE.finditer(text))
    return link_chars / len(text) > threshold


def _adaptive_max_per_document(query_text: str, default: int = 2) -> int:
    """Для продуктовых запросов лимит 4, для остальных default (2)."""
    q = (query_text or "").lower()
    if any(hint in q for hint in _PRODUCT_HINTS):
        return 4
    return default