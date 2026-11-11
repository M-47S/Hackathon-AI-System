import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------- Embedding ----------

@total_ordering
class Embedding:
    def __init__(self, vector: Union[List[float], ndarray]):
        self._vector = np.array(vector, dtype=np.float32).reshape(-1)
        self._size = len(self._vector)

    def __len__(self) -> int:
        return self._size

    def __add__(self, other: "Embedding") -> "Embedding":
        if not isinstance(other, Embedding):
            return NotImplemented
        if self._size != other._size:
            raise ValueError(
                f"Нельзя сложить векторы разных размерностей: "
                f"{self._size} vs {other._size}"
            )
        return Embedding(self._vector + other.vector)

    def __matmul__(self, other: "Embedding") -> float:
        if not isinstance(other, Embedding):
            return NotImplemented
        if self._size != other._size:
            raise ValueError(
                f"Нельзя перемножить векторы разных размерностей: "
                f"{self._size} vs {other._size}"
            )
        return float(np.sum(self._vector * other.vector))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Embedding):
            return NotImplemented
        return (
            self._vector.shape == other._vector.shape
            and np.allclose(self._vector, other._vector, atol=1e-6)
        )

    def __lt__(self, other: "Embedding") -> bool:
        if not isinstance(other, Embedding):
            return NotImplemented
        if self._size != other._size:
            raise ValueError(
                f"Нельзя сравнить векторы разных размерностей: "
                f"{self._size} vs {other._size}"
            )
        return self.norm < other.norm

    def __hash__(self) -> int:
        return hash(self._vector.tobytes())

    def normalized(self) -> "Embedding":
        norm = self.norm
        if norm == 0:
            return Embedding(self._vector)
        return Embedding(self._vector / norm)

    @property
    def size(self) -> int:
        return self._size

    @property
    def norm(self) -> float:
        return float(np.sqrt(np.sum(self._vector ** 2)))

    @property
    def vector(self) -> ndarray:
        return self._vector

    def _format_vector(self, max_items: int = 8, precision: int = 4) -> str:
        v = self._vector
        opts = dict(precision=precision, separator=", ", suppress_small=True)
        if len(v) <= max_items:
            return np.array2string(v, **opts)
        half = max_items // 2
        head = np.array2string(v[:half], **opts)[:-1]
        tail = np.array2string(v[-half:], **opts)[1:]
        return f"{head} ... {tail}"

    def __repr__(self) -> str:
        return (
            f"Embedding(\n"
            f"  vector: {self._format_vector()}\n"
            f"  size:   {self._size}\n"
            f"  norm:   {self.norm:.6f}\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------- Retrieval ----------

@dataclass(frozen=True)
class VectorHit:
    chunk_id: int
    score: float


# ---------- Documents & Chunks ----------

@dataclass(frozen=True)
class Document:
    document_id: str
    filename: str
    relative_path: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve_path(self, documents_root: Path) -> Path:
        return (Path(documents_root) / self.relative_path).resolve()

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Document":
        return cls(
            document_id=raw["document_id"],
            filename=raw["filename"],
            relative_path=raw["relative_path"],
            mime_type=raw.get("mime_type"),
            size_bytes=raw.get("size_bytes"),
            sha256=raw.get("sha256"),
            extra=dict(raw.get("extra") or {}),
        )


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    document_id: str
    text: str
    position: int
    section_title: Optional[str] = None
    parent_chunk_id: Optional[int] = None
    page: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=raw["chunk_id"],
            document_id=raw["document_id"],
            text=raw["text"],
            position=raw["position"],
            section_title=raw.get("section_title"),
            parent_chunk_id=raw.get("parent_chunk_id"),
            page=raw.get("page"),
            char_start=raw.get("char_start"),
            char_end=raw.get("char_end"),
        )


# ---------- Small-to-Big ----------

class ContextMode(str, Enum):
    STRICT = "strict"   # только сам чанк
    PARENT = "parent"   # текст родителя
    WINDOW = "window"   # соседи ± window_size


@dataclass(frozen=True)
class ChunkContext:
    chunk: Chunk
    context_text: str
    parent: Optional[Chunk] = None
    window: tuple = ()  # tuple[Chunk, ...], но без forward-ref проблемы