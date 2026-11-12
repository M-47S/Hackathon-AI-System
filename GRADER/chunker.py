# GRADER/chunker.py
"""
Чистый Python-чанкер Markdown → Chunk с иерархией Small-to-Big.

Не зависит от llama-index, tiktoken, sentence-transformers.
Не делает сетевых вызовов — работает в изолированной среде платформы.
"""
import logging
import re
from typing import List, Optional, Tuple

from structs import Chunk

logger = logging.getLogger(__name__)


# Markdown heading: # Title / ## Title / ### Title
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$', re.MULTILINE)
# Sentence boundary: punctuation followed by whitespace + capital letter
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[А-ЯЁA-Z«"(])')


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_markdown_sections(
    markdown_text: str,
) -> List[Tuple[Optional[str], str, int]]:
    """
    Разбивает markdown на секции по заголовкам # .. ######.

    Возвращает список (section_title, section_body, heading_level).
    Контент до первого заголовка → отдельная секция с title=None, level=0.
    Пустые секции (заголовок без текста) возвращаются с пустым body —
    вызывающий код их отфильтрует.
    """
    matches = list(_HEADING_RE.finditer(markdown_text))

    if not matches:
        return [(None, markdown_text.strip(), 0)]

    sections: List[Tuple[Optional[str], str, int]] = []

    # Преамбула до первого заголовка
    first_start = matches[0].start()
    if first_start > 0:
        preamble = markdown_text[:first_start].strip()
        if preamble:
            sections.append((None, preamble, 0))

    # Каждая секция: от конца заголовка до начала следующего
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[body_start:body_end].strip()
        sections.append((title, body, level))

    return sections


def _split_into_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения по . ! ? с сохранением пунктуации."""
    if not text.strip():
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _simple_split(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> List[str]:
    """
    Разбивает текст на чанки по ~max_chars символов с перекрытием.

    Старается сохранять границы предложений. Если одно предложение
    длиннее max_chars — режется принудительно (редкий случай).
    """
    if not text.strip():
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = _split_into_sentences(text)
    if not sentences:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        # Одиночное предложение длиннее max_chars — режем посимвольно
        if sent_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for j in range(0, sent_len, max_chars):
                chunks.append(sent[j:j + max_chars])
            continue

        # Добавление этого предложения превысит лимит?
        if current and current_len + sent_len + 1 > max_chars:
            chunks.append(" ".join(current))

            # Перекрытие: последние предложения из закрытого чанка
            overlap: List[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap
            current_len = overlap_len

        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Contextual retrieval helpers
# ---------------------------------------------------------------------------

def _extract_document_title(markdown_text: str) -> str:
    """Первый H1, либо первая непустая строка без #."""
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    for line in markdown_text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


def _make_embedding_text(doc_title, section_title, text, doc_filename=""):
    parts = []
    if doc_title: parts.append(doc_title)
    if section_title: parts.append(section_title)
    if doc_filename: parts.append(doc_filename)   # NEW
    header = " — ".join(parts)
    return f"{header}:\n{text}" if header else text
        


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarkdownChunker:
    """
    Чистый Python-чанкер Markdown с иерархией Small-to-Big.

    - Родители: секции по заголовкам # ## ### (без дальнейшего дробления).
    - Листья: родители, разбитые на куски ~chunk_size токенов с перекрытием.
      Именно листья идут в FAISS.
    - parent_chunk_id у листа указывает на его секцию;
      у родителя — на секцию более высокого уровня (H3 → H2 → H1).

    Параметры chunk_size / chunk_overlap интерпретируются как «токены»
    с приблизительным пересчётом 1 токен ≈ 4 символа (для русского).
    """

    # В MarkdownChunker.__init__:
    def __init__(
        self,
        chunk_size: int = 640,      # токенов, ≈2560 символов (было 384)
        chunk_overlap: int = 64,    # токенов, ≈256 символов (было 30)
    ):
        self._max_chars = max(400, chunk_size * 4)
        self._overlap_chars = max(0, chunk_overlap * 4)

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

        doc_title = _extract_document_title(markdown_text)
        sections = _parse_markdown_sections(markdown_text)

        next_id = start_id
        parent_chunks: List[Chunk] = []
        leaf_chunks: List[Chunk] = []

        # Иерархия по уровню заголовка: level_parents[2] = chunk_id последнего H2
        # Это даёт корректный parent_chunk_id для вложенных секций (H3 → H2).
        level_parents: dict[int, int] = {}

        for pos, (section_title, section_body, level) in enumerate(sections):
            if not section_body:
                # Заголовок без текста — пропускаем, но запоминаем уровень
                # только если у него будет содержание в дочерних секциях —
                # это сложно отследить; проще пропустить пустые.
                continue

            # Определяем parent_chunk_id из иерархии заголовков
            parent_chunk_id: Optional[int] = None
            if level > 0:
                for lvl in range(level - 1, 0, -1):
                    if lvl in level_parents:
                        parent_chunk_id = level_parents[lvl]
                        break

            # Создаём родительский чанк (текст = тело секции без заголовка)
            parent_id = next_id
            next_id += 1

            parent_chunks.append(Chunk(
                chunk_id=parent_id,
                document_id=document_id,
                text=section_body,
                position=pos * 10000,
                section_title=section_title,
                parent_chunk_id=parent_chunk_id,
                is_leaf=False,
            ))

            # Обновляем иерархию
            if level > 0:
                level_parents[level] = parent_id
                # Сбрасываем уровни глубже текущего
                for lvl in [l for l in level_parents if l > level]:
                    del level_parents[lvl]

            # Разбиваем родителя на листья
            leaf_texts = _simple_split(
                section_body,
                max_chars=self._max_chars,
                overlap_chars=self._overlap_chars,
            )

            for i, leaf_text in enumerate(leaf_texts):
                leaf_id = next_id
                next_id += 1

                leaf_chunks.append(Chunk(
                    chunk_id=leaf_id,
                    document_id=document_id,
                    text=leaf_text,
                    position=pos * 10000 + i + 1,
                    section_title=section_title,
                    parent_chunk_id=parent_id,
                    is_leaf=True,
                    embedding_text=_make_embedding_text(
                        doc_title=doc_title,
                        section_title=section_title,
                        text=leaf_text,
                        doc_filename=document_id,  # используем как fallback, если path недоступен
                    ),
                ))

        all_chunks = parent_chunks + leaf_chunks
        leaf_ids = [c.chunk_id for c in leaf_chunks]

        logger.info(
            "[%s] создано %d родителей и %d листьев",
            document_id, len(parent_chunks), len(leaf_chunks),
        )
        return all_chunks, leaf_ids, next_id