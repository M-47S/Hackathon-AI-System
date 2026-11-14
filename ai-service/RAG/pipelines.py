"""
RAG/pipelines.py — сценарный слой RAG (с поддержкой гибридного поиска).

Гибридный поиск:
  - FAISS (векторный) + BM25 (лексический)
  - Fusion через Reciprocal Rank Fusion (RRF, k=60)
  - Отключается флагом use_hybrid=False (default: чистый FAISS)
"""

from __future__ import annotations

import hashlib
import logging
import math
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
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]: ...


@runtime_checkable
class EmbedderLike(Protocol):
    def _get_query_embedding(self, query: str) -> Embedding: ...
    def _get_query_embedding_batch(self, queries: List[str]) -> List[Embedding]: ...
    def _get_text_embedding(self, text: str) -> Embedding: ...
    def _get_text_embedding_batch(self, texts: List[str]) -> List[Embedding]: ...


@runtime_checkable
class ChunkerLike(Protocol):
    def chunk(
        self,
        markdown_text: str,
        document_id: str,
        start_id: int = 0,
    ) -> Tuple[List[Chunk], List[int], int]: ...


_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
_BM25_TOKEN_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)

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
# BM25 (лексический поиск)
# ---------------------------------------------------------------------------

class BM25Index:
    """Простой BM25 (Okapi BM25) по leaf-чанкам DocumentStore.

    Не требует сторонних библиотек. Строится один раз, потом — быстрый поиск.

    Параметры k1 и b — стандартные из литературы:
      k1 = 1.5 — насыщение term frequency
      b  = 0.75 — нормализация по длине документа

    Пример использования:
        index = BM25Index(doc_store.leaf_chunks)
        results = index.search("Меркурий 230", top_k=15)
        # → [(chunk_id, bm25_score), ...] по убыванию score
    """

    def __init__(
        self,
        chunks: List[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._k1 = k1
        self._b = b
        self._chunks = list(chunks)
        self._doc_tokens: dict[int, list[str]] = {}
        self._doc_tf: dict[int, dict[str, int]] = {}
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._build()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Простая токенизация: lowercase, слова из ≥3 букв/цифр."""
        return _BM25_TOKEN_RE.findall(text.lower().replace("ё", "е"))

    def _build(self) -> None:
        total_len = 0
        for c in self._chunks:
            tokens = self._tokenize(c.text or "")
            self._doc_tokens[c.chunk_id] = tokens
            total_len += len(tokens)

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._doc_tf[c.chunk_id] = tf

            for t in tf.keys():
                self._df[t] = self._df.get(t, 0) + 1

        n = len(self._chunks)
        self._avgdl = (total_len / n) if n > 0 else 0.0

        logger.info(
            "BM25Index built: %d chunks, %d unique terms, avgdl=%.1f",
            n, len(self._df), self._avgdl,
        )

    def search(self, query: str, top_k: int = 25) -> List[Tuple[int, float]]:
        """Возвращает [(chunk_id, bm25_score), ...] по убыванию score."""
        if not self._chunks or not query.strip():
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        n = len(self._chunks)
        avgdl = self._avgdl or 1.0

        scores: dict[int, float] = {}
        for c in self._chunks:
            tf = self._doc_tf.get(c.chunk_id, {})
            dl = len(self._doc_tokens.get(c.chunk_id, []))
            if dl == 0:
                continue

            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                df = self._df.get(qt, 0)
                # IDF с +1 чтобы избежать отрицательных значений
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                num = f * (self._k1 + 1.0)
                denom = f + self._k1 * (1.0 - self._b + self._b * dl / avgdl)
                score += idf * (num / denom)

            if score > 0:
                scores[c.chunk_id] = score

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return ranked


def build_bm25_index(doc_store: DocumentStore) -> BM25Index:
    """Строит BM25 по всем leaf-чанкам DocumentStore."""
    leaves = doc_store.leaf_chunks
    if not leaves:
        logger.warning("build_bm25_index: нет leaf-чанков")
        return BM25Index([])
    return BM25Index(leaves)


def rrf_fuse(
    rankings: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Принимает несколько ранжирований [(chunk_id, score), ...],
    возвращает объединённое [(chunk_id, rrf_score), ...] по убыванию RRF.

    k=60 — стандартное значение из оригинальной статьи Cormack et al.
    Score-и исходных ранжирований игнорируются, важно только место в списке.
    """
    rrf_scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: -x[1])


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
    exclude_documents: Optional[set] = None,
    # ── NEW: гибридный поиск ──
    bm25_index: Optional[BM25Index] = None,
    use_hybrid: bool = False,
) -> List[RetrievedChunk]:
    """
    Возвращает top_k релевантных чанков.

    Режимы:
      1) Чистый FAISS (use_hybrid=False) — как раньше.
      2) Hybrid (use_hybrid=True + bm25_index):
         FAISS top-N + BM25 top-N → RRF fusion → фильтры → reranker → top_k.
    """
    if top_k <= 0:
        return []

    use_reranker = reranker is not None and query_text is not None
    if reranker is not None and query_text is None:
        logger.warning(
            "pipeline_search: передан reranker, но не передан query_text — "
            "реранкер будет проигнорирован."
        )

    # Hybrid активен только если передан bm25_index и query_text
    hybrid_active = (
        use_hybrid
        and bm25_index is not None
        and query_text is not None
    )
    if use_hybrid and not hybrid_active:
        logger.warning(
            "pipeline_search: use_hybrid=True, но %s — fallback на чистый FAISS.",
            "нет bm25_index" if bm25_index is None else "нет query_text",
        )

    # Авто-выбор лимита на документ, если явно не задан
    if max_per_document is None:
        max_per_document = _adaptive_max_per_document(query_text or "")

    # 1. Сколько запросить у каждого поисковика
    if use_reranker:
        search_k = top_k * max(1, rerank_oversample)
    elif chunk_filter is not None or filter_toc or exclude_documents or hybrid_active:
        search_k = top_k * max(1, oversample)
    else:
        search_k = top_k

    # 2. Retrieval: FAISS-only или Hybrid (FAISS + BM25 → RRF)
    hit_score: dict[int, float] = {}
    ordered_chunks: List[Chunk] = []

    if hybrid_active:
        # --- FAISS ---
        faiss_hits = vec_store.search(query, top_k=search_k, min_score=min_score)
        faiss_ranking = [(h.chunk_id, h.score) for h in faiss_hits]

        # --- BM25 ---
        bm25_ranking = bm25_index.search(query_text or "", top_k=search_k)

        if not faiss_ranking and not bm25_ranking:
            return []

        # --- RRF fusion ---
        fused = rrf_fuse([faiss_ranking, bm25_ranking])
        fused_ids = [cid for cid, _ in fused]
        hit_score = dict(fused)

        chunk_map = {
            c.chunk_id: c
            for c in doc_store.get_chunks(fused_ids)
        }
        ordered_chunks = [
            chunk_map[cid] for cid in fused_ids if cid in chunk_map
        ]

        logger.debug(
            "pipeline_search [hybrid]: faiss=%d, bm25=%d, fused=%d",
            len(faiss_ranking), len(bm25_ranking), len(ordered_chunks),
        )
    else:
        # --- Чистый FAISS (как было) ---
        hits = vec_store.search(query, top_k=search_k, min_score=min_score)
        if not hits:
            return []

        chunk_map = {
            c.chunk_id: c
            for c in doc_store.get_chunks(h.chunk_id for h in hits)
        }
        for h in hits:
            chunk = chunk_map.get(h.chunk_id)
            if chunk is None:
                logger.warning(
                    "pipeline_search: chunk_id=%s отсутствует в DocumentStore",
                    h.chunk_id,
                )
                continue
            ordered_chunks.append(chunk)
        hit_score = {h.chunk_id: h.score for h in hits}

    # 3. Фильтры
    if chunk_filter is not None:
        ordered_chunks = [c for c in ordered_chunks if chunk_filter(c)]

    if filter_toc:
        ordered_chunks = [c for c in ordered_chunks if not _is_toc_like(c)]

    if exclude_documents:
        ordered_chunks = [
            c for c in ordered_chunks if c.document_id not in exclude_documents
        ]

    if not ordered_chunks:
        return []

    # 4. Диверсификация
    ordered_chunks = _diversify_by_document(
        ordered_chunks,
        max_per_document=max_per_document,
        min_required=0 if use_reranker else top_k,
    )

    # 5. Reranker (с fallback)
    if use_reranker:
        reranked = []
        try:
            reranked = reranker.rerank(
                query=query_text,
                documents=[c.text for c in ordered_chunks],
                top_n=top_k,
            )
        except Exception as e:
            logger.warning("Reranker упал (%s), fallback на retrieval scores", e)

        if reranked:
            return [
                RetrievedChunk(
                    chunk=ordered_chunks[idx],
                    score=float(score),
                    reranked=True,
                )
                for idx, score in reranked
                if 0 <= idx < len(ordered_chunks)
            ]

        # FALLBACK: scores от retrieval (FAISS или RRF)
        return [
            RetrievedChunk(
                chunk=c,
                score=hit_score.get(c.chunk_id, 0.0),
                reranked=False,
            )
            for c in ordered_chunks[:top_k]
        ]

    # 6. Без reranker: top_k по retrieval score
    return [
        RetrievedChunk(
            chunk=c,
            score=hit_score.get(c.chunk_id, 0.0),
            reranked=False,
        )
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
    bm25_index: Optional[BM25Index] = None,
    use_hybrid: bool = False,
) -> List[ChunkContext]:
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
        bm25_index=bm25_index,
        use_hybrid=use_hybrid,
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
    rerank_query_text: Optional[str] = None,
    bm25_index: Optional[BM25Index] = None,
    use_hybrid: bool = False,
    **search_kwargs,
) -> List[RetrievedChunk]:
    text_for_embedding = (
        f"{query_instruction}{query_text}" if query_instruction else query_text
    )
    query_embedding = embedder._get_query_embedding(text_for_embedding)
    return pipeline_search(
        doc_store=doc_store,
        vec_store=vec_store,
        query=query_embedding,
        query_text=rerank_query_text if rerank_query_text is not None else query_text,
        bm25_index=bm25_index,
        use_hybrid=use_hybrid,
        **search_kwargs,
    )


def pipeline_search_with_context_by_text(
    embedder: EmbedderLike,
    doc_store: DocumentStore,
    vec_store: VectorStore,
    query_text: str,
    query_instruction: Optional[str] = DEFAULT_QUERY_INSTRUCTION,
    rerank_query_text: Optional[str] = None,
    bm25_index: Optional[BM25Index] = None,
    use_hybrid: bool = False,
    **search_kwargs,
) -> List[ChunkContext]:
    text_for_embedding = (
        f"{query_instruction}{query_text}" if query_instruction else query_text
    )
    query_embedding = embedder._get_query_embedding(text_for_embedding)
    return pipeline_search_with_context(
        doc_store=doc_store,
        vec_store=vec_store,
        query=query_embedding,
        query_text=rerank_query_text if rerank_query_text is not None else query_text,
        bm25_index=bm25_index,
        use_hybrid=use_hybrid,
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
    embedder: EmbedderLike,
    chunker: Optional[ChunkerLike] = None,
    pattern: str = "**/*.md",
    save: bool = True,
) -> FillStats:
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


def _is_toc_like(chunk: Chunk, threshold: float = 0.65) -> bool:
    text = chunk.text
    if not text:
        return True
    if len(text) < 150:
        return False
    link_chars = sum(len(m.group(0)) for m in _LINK_RE.finditer(text))
    return link_chars / len(text) > threshold


def _adaptive_max_per_document(query_text: str, default: int = 2) -> int:
    q = (query_text or "").lower()
    if any(hint in q for hint in _PRODUCT_HINTS):
        return 4
    return default