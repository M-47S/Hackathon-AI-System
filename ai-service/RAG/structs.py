from numpy import ndarray, array
import numpy as np
from dataclasses import dataclass, field, asdict
from llama_index.core.schema import TextNode, NodeRelationship
from typing import List, Dict, Union, Self, Optional, Any
from enum import Enum
from pathlib import Path
import json
from functools import total_ordering

@total_ordering
class Embedding:
    def __init__(self, vector: Union[List[float], ndarray]):
       self._vector = np.array(vector, dtype=np.float32).reshape(-1)
       self._size = len(self._vector)

    def __len__(self) -> int:
        return self._size 
    
    def __add__(self, other: Self) -> Self:
        if not isinstance(other, Embedding):
            raise TypeError("Vectors must have same types - 'Embedding'")
        return Embedding(self._vector + other.vector)
        
    def __matmul__(self, other: 'Embedding') -> float:
        # ИСПРАВЛЕНО: __matmul__ для скалярного произведения (@)
        if not isinstance(other, Embedding):
            return NotImplemented
        return float(np.sum(self._vector * other.vector))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Embedding):
            return NotImplemented
        return (
            self._vector.shape == other._vector.shape
            and np.allclose(self._vector, other._vector, atol=1e-6)
        )

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, Embedding):
            raise TypeError("Vectors must have same types - 'Embedding'")
        if self._size != other._size:
            raise ValueError(
                f"Cannot compare Embeddings of different dimensions: "
                f"{self._size} vs {other._size}"
            )
        return self.norm < other.norm

    def __hash__(self) -> int:
        return hash(self._vector.tobytes())
    
    def normalized(self) -> 'Embedding':
        """Возвращает L2-нормализованную копию (для косинусного сходства)."""
        norm = self.norm
        if norm == 0:
            return Embedding(self._vector)
        return Embedding(self._vector / norm)
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def norm(self) -> float:
        return float(np.sqrt(np.sum(self._vector**2)))   # ← евклидова норма
    
    @property
    def vector(self):
        return self._vector
    
    def _format_vector(self, max_items: int = 8, precision: int = 4) -> str:
        """Строка-представление вектора с обрезкой, если он длинный."""
        v = self._vector
        opts = dict(precision=precision, separator=', ', suppress_small=True)

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


BASE_PNAME = Path("parents.json")
BASE_CNAME = Path("children.json")

class ChunkCollection:
    """
    Контейнер для хранения чанков.
    Разделяет родительские и дочерние чанки для удобства.
    """
    def __init__(self):
        self._parents: List[TextNode] = []
        self._children: List[TextNode] = []
        
        #Cловарь для мгновенного поиска родителя по ID: {parent_id: TextNode}
        self._parent_map: Dict[str, TextNode] = {}

    def __len__(self) -> int:
        return len(self._children)

    def __save_file(self, ListNode: List[TextNode], fpath: Path):
        data = [node.to_dict() for node in ListNode]
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def __load_file(self, fpath: Path) -> List[TextNode]:
        if not fpath.exists():
            raise FileNotFoundError(f"Файл не найден: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [TextNode.from_dict(item) for item in raw]
    
    def add_children(self, nodes: Union[List[TextNode], TextNode]) -> None:
        if isinstance(nodes, list):
            self._children.extend(nodes)
        else:
            self._children.append(nodes)
            
    def add_parent(self, node: TextNode) -> None:
        self._parents.append(node)
        
        self._parent_map[node.node_id] = node
        
    def get_parent_by_child(self, child_node: TextNode) ->  TextNode | None:
        """Возвращает родительский чанк для заданного дочернего."""
        
        parent_rel = child_node.relationships.get(NodeRelationship.PARENT)
        
        if parent_rel:
            result = self._parent_map.get(parent_rel.node_id)
        else:
            result = None
            
        return result
    
    def save_collection(self, path_to_dir: Path | str):
        if isinstance(path_to_dir, str):
            path_to_dir: Path = Path(path_to_dir)
            
        ppath = path_to_dir / BASE_PNAME
        ppath.touch()
        self.__save_file(self._parents, ppath)
        
        cpath = path_to_dir / BASE_CNAME
        cpath.touch()
        self.__save_file(self._children, cpath)

    @classmethod
    def load_collection(cls, path_to_dir: Path | str) -> "ChunkCollection":
        """Загружает коллекцию из папки с parents.json и children.json."""
        path_to_dir = Path(path_to_dir)
        collection = cls()

        for parent in collection.__load_file(path_to_dir / BASE_PNAME):
            collection.add_parent(parent)

        collection.add_children(collection.__load_file(path_to_dir / BASE_CNAME))
        return collection
    
    @property
    def children(self):
        return self._children
    
    @property
    def parents(self):
        return self._parents
    
@dataclass(frozen=True)
class VectorHit:
    chunk_id: int
    score: float


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


class ContextMode(str, Enum):
    STRICT = "strict"   # только сам чанк
    PARENT = "parent"   # текст родителя (Small-to-Big)
    WINDOW = "window"   # соседи ± window_size


@dataclass(frozen=True)
class ChunkContext:
    """Что реально идёт в промпт LLM + метаданные для debug-режима."""
    chunk: Chunk
    context_text: str
    parent: Optional[Chunk] = None
    window: tuple[Chunk, ...] = ()