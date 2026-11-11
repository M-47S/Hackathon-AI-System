# RAG/Chunkers/MarkdownChunker.py
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from llama_index.core import Document as LIDocument
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import NodeRelationship

from ..structs import Chunk


logger = logging.getLogger(__name__)


class MarkdownChunker:
    """LlamaIndex-парсер Markdown → список Chunk с иерархией.

    Возвращает:
    - chunks: все чанки (родители + листья), с int chunk_id;
    - leaf_ids: ID листьев (то, что идёт в FAISS);
    - next_id: следующий свободный chunk_id (для сквозной нумерации).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.md_parser = MarkdownNodeParser()
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        markdown_text: str,
        document_id: str,
        start_id: int = 0,
    ) -> Tuple[List[Chunk], List[int], int]:
        if not isinstance(markdown_text, str):
            raise TypeError(
                f"markdown_text должен быть str, получено "
                f"{type(markdown_text).__name__}"
            )
        if not markdown_text.strip():
            logger.warning("Пустой Markdown, возвращаю пустую коллекцию")
            return [], [], start_id

        source_doc = LIDocument(
            text=markdown_text,
            metadata={"source": document_id},
        )
        parent_nodes = self.md_parser.get_nodes_from_documents([source_doc])
        
        doc_title = _extract_document_title(markdown_text)

        # Шаг 1. Присваиваем int ID каждому родителю, запоминаем маппинг
        # node_id (str) → chunk_id (int)
        next_id = start_id
        id_map: dict[str, int] = {}
        parent_chunks: list[Chunk] = []

        # Порядок позиций: родители нумеруются по порядку в документе,
        # листья получат position = parent.position * 10000 + i,
        # чтобы при сортировке листья шли в правильном порядке.
        for pos, parent in enumerate(parent_nodes):
            chunk_id = next_id
            next_id += 1
            id_map[parent.node_id] = chunk_id

            # Родитель может сам иметь PARENT (для вложенных заголовков).
            # MarkdownNodeParser выставляет это через relationships.
            parent_rel = parent.relationships.get(NodeRelationship.PARENT)
            parent_chunk_id = id_map.get(parent_rel.node_id) if parent_rel else None

            section_title = _extract_section_title(parent)

            parent_chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=parent.text,
                position=pos * 10000,   # см. ниже
                section_title=section_title,
                parent_chunk_id=parent_chunk_id,
            ))

        # Шаг 2. Режем каждого родителя на листья
        leaf_chunks: list[Chunk] = []

        for parent in parent_nodes:
            parent_chunk_id = id_map[parent.node_id]
            parent_chunk = next(c for c in parent_chunks if c.chunk_id == parent_chunk_id)

            child_nodes = self.text_splitter.get_nodes_from_documents(
                [LIDocument(text=parent.text)]
            )

            for i, child in enumerate(child_nodes):
                chunk_id = next_id
                next_id += 1
                id_map[child.node_id] = chunk_id

                leaf_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=child.text,
                    position=parent_chunk.position + i + 1,
                    section_title=parent_chunk.section_title,
                    parent_chunk_id=parent_chunk_id,
                    is_leaf=True,
                    embedding_text=_make_embedding_text(
                        doc_title=doc_title,
                        section_title=parent_chunk.section_title,
                        text=child.text,
                    ),
                ))

        all_chunks = parent_chunks + leaf_chunks
        leaf_ids = [c.chunk_id for c in leaf_chunks]

        logger.info(
            f"[{document_id}] создано {len(parent_chunks)} родителей и "
            f"{len(leaf_chunks)} листьев"
        )
        return all_chunks, leaf_ids, next_id


def _extract_section_title(node) -> str | None:
    """LlamaIndex кладёт в metadata['header_path'] что-то вроде
    '/Документ/Раздел 2/Подраздел'. Возвращаем последний непустой сегмент.
    """
    header_path = (node.metadata or {}).get("header_path")
    if not header_path:
        return None
    parts = [p.strip() for p in str(header_path).split("/") if p.strip()]
    return parts[-1] if parts else None

import re

def _extract_document_title(markdown_text: str) -> str:
    """
    Первый H1 в markdown — самое читаемое название документа.
    Fallback: первая непустая строка без #.
    """
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    # fallback: первая непустая строка
    for line in markdown_text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


def _make_embedding_text(
    doc_title: str,
    section_title: Optional[str],
    text: str,
) -> str:
    """
    Контекстный префикс для эмбеддинга.
    Формат: '<Документ> — <Раздел>:\n<текст>'
    """
    parts = []
    if doc_title:
        parts.append(doc_title)
    if section_title:
        parts.append(section_title)
    header = " — ".join(parts)
    return f"{header}:\n{text}" if header else text