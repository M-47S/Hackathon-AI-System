# GRADER/document_store.py
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
import json

from structs import Chunk, ChunkContext, ContextMode, Document


BASE_DOCUMENTS = Path("documents.json")
BASE_CHUNKS = Path("chunks.json")


class DocumentStore:
    """
    Хранит метаданные документов (оригиналы — на диске) и чанки.

    - Документы НЕ загружаются автоматически в память: только через
      load_document_bytes().
    - Чанки хранятся в памяти (dict по chunk_id) и сохраняются в JSON.
    - chunk_id совпадает с ID в VectorStore.
    - Внутренний счётчик _next_chunk_id — следующий свободный chunk_id.
      Он синхронизируется автоматически при add_chunks() и load().
    - resolve_document_id() выдаёт свободный document_id, если preferred
      уже занят (например, два файла с одинаковым stem).

    Small-to-Big: маленькие чанки-листья индексируются в FAISS,
    родительские чанки лежат только здесь и служат контекстом.
    """

    def __init__(self, documents_root: Union[Path, str]):
        self._documents_root = Path(documents_root)
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[int, Chunk] = {}
        self._chunks_by_document: Dict[str, List[int]] = {}
        self._next_chunk_id: int = 0

    # ---------------- Документы ----------------

    def add_document(self, document: Document) -> None:
        if document.document_id in self._documents:
            raise ValueError(f"Документ {document.document_id} уже добавлен")
        self._documents[document.document_id] = document

    def add_documents(self, documents: Iterable[Document]) -> None:
        for d in documents:
            self.add_document(d)

    def get_document(self, document_id: str) -> Optional[Document]:
        return self._documents.get(document_id)

    def resolve_document_id(self, preferred: str) -> str:
        """
        Возвращает свободный document_id: preferred, либо preferred_1,
        _2, ... — первый незанятый. Чисто строковая логика, без счётчика.
        """
        if preferred not in self._documents:
            return preferred
        n = 1
        while f"{preferred}_{n}" in self._documents:
            n += 1
        return f"{preferred}_{n}"

    def load_document_bytes(
        self, document_id: str, max_bytes: Optional[int] = None
    ) -> bytes:
        """Читает оригинал с диска. Единственный способ получить содержимое."""
        doc = self._documents.get(document_id)
        if doc is None:
            raise KeyError(f"Документ {document_id} не найден")
        path = doc.resolve_path(self._documents_root)
        if not path.exists():
            raise FileNotFoundError(f"Оригинал отсутствует на диске: {path}")
        if max_bytes is not None:
            with open(path, "rb") as f:
                return f.read(max_bytes)
        return path.read_bytes()

    # ---------------- ID allocation ----------------

    @property
    def next_chunk_id(self) -> int:
        """Следующий свободный chunk_id для передачи в chunker.start_id."""
        return self._next_chunk_id

    # ---------------- Чанки ----------------

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """
        Атомарно добавляет чанки:
          1) сначала валидирует ВСЕ,
          2) затем коммитит всё,
          3) затем синхронизирует счётчик _next_chunk_id.
        Если хоть один chunk_id занят или ссылается на неизвестный
        документ — ничего не добавляется.
        """
        chunks = list(chunks)  # материализуем, чтобы итерировать дважды

        # --- validate all ---
        for c in chunks:
            if c.chunk_id in self._chunks:
                raise ValueError(f"Чанк {c.chunk_id} уже существует")
            if c.document_id not in self._documents:
                raise ValueError(
                    f"Чанк {c.chunk_id} ссылается на несуществующий документ "
                    f"{c.document_id}"
                )

        # --- commit all ---
        for c in chunks:
            self._chunks[c.chunk_id] = c
            self._chunks_by_document.setdefault(c.document_id, []).append(c.chunk_id)

        # --- sync counter ---
        if chunks:
            self._next_chunk_id = max(
                self._next_chunk_id,
                max(c.chunk_id for c in chunks) + 1,
            )

    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: Iterable[int]) -> List[Chunk]:
        return [c for cid in chunk_ids if (c := self._chunks.get(cid)) is not None]

    def get_chunks_of_document(self, document_id: str) -> List[Chunk]:
        ids = self._chunks_by_document.get(document_id, [])
        return self.get_chunks(ids)

    # ---------------- Small-to-Big ----------------

    def get_chunk_with_context(
        self,
        chunk_id: int,
        mode: ContextMode = ContextMode.PARENT,
        window_size: int = 1,
    ) -> ChunkContext:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(f"Чанк {chunk_id} не найден")

        if mode == ContextMode.STRICT:
            return ChunkContext(chunk=chunk, context_text=chunk.text)

        if mode == ContextMode.PARENT:
            if chunk.parent_chunk_id is None:
                return ChunkContext(chunk=chunk, context_text=chunk.text)
            parent = self._chunks.get(chunk.parent_chunk_id)
            if parent is None:
                return ChunkContext(chunk=chunk, context_text=chunk.text)
            return ChunkContext(
                chunk=chunk, context_text=parent.text, parent=parent
            )

        if mode == ContextMode.WINDOW:
            neighbors = self._get_neighbors(chunk, window_size)
            return ChunkContext(
                chunk=chunk,
                context_text="\n\n".join(c.text for c in neighbors),
                window=tuple(neighbors),
            )

        raise ValueError(f"Неизвестный режим: {mode}")

    def _get_neighbors(self, chunk: Chunk, window_size: int) -> List[Chunk]:
        siblings = self.get_chunks_of_document(chunk.document_id)
        siblings.sort(key=lambda c: c.position)
        idx = next(
            (i for i, c in enumerate(siblings) if c.chunk_id == chunk.chunk_id),
            None,
        )
        if idx is None:
            return [chunk]
        lo = max(0, idx - window_size)
        hi = min(len(siblings), idx + window_size + 1)
        return siblings[lo:hi]

    # ---------------- Персистентность ----------------

    def save(self, storage_dir: Union[Path, str]) -> None:
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)

        docs = [asdict(d) for d in self._documents.values()]
        (storage_dir / BASE_DOCUMENTS).write_text(
            json.dumps(docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        chunks = [asdict(c) for c in self._chunks.values()]
        (storage_dir / BASE_CHUNKS).write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        storage_dir: Union[Path, str],
        documents_root: Union[Path, str],
    ) -> "DocumentStore":
        storage_dir = Path(storage_dir)
        docs_path = storage_dir / BASE_DOCUMENTS
        chunks_path = storage_dir / BASE_CHUNKS
        if not docs_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Не найдены {BASE_DOCUMENTS} и/или {BASE_CHUNKS} в {storage_dir}"
            )

        store = cls(documents_root=documents_root)
        store.add_documents(
            Document.from_dict(raw)
            for raw in json.loads(docs_path.read_text(encoding="utf-8"))
        )
        # add_chunks сам восстановит _next_chunk_id из загруженных данных
        store.add_chunks(
            Chunk.from_dict(raw)
            for raw in json.loads(chunks_path.read_text(encoding="utf-8"))
        )
        return store

    # ---------------- Валидация и утилиты ----------------

    def validate(self) -> List[str]:
        """Возвращает список проблем. Пустой список = всё корректно."""
        errors: List[str] = []
        for c in self._chunks.values():
            if c.document_id not in self._documents:
                errors.append(
                    f"chunk {c.chunk_id}: unknown document {c.document_id}"
                )
            if c.parent_chunk_id is not None and c.parent_chunk_id not in self._chunks:
                errors.append(
                    f"chunk {c.chunk_id}: missing parent {c.parent_chunk_id}"
                )
        return errors

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: int) -> bool:
        return chunk_id in self._chunks

    @property
    def documents_root(self) -> Path:
        return self._documents_root

    @property
    def documents(self) -> Iterable[Document]:
        return self._documents.values()

    @property
    def chunks(self) -> Iterable[Chunk]:
        return self._chunks.values()
    
    @property
    def leaf_chunks(self) -> List[Chunk]:
        """Чанки, которые проиндексированы в VectorStore (is_leaf=True)."""
        return [c for c in self._chunks.values() if c.is_leaf]


if __name__ == "__main__":
    import shutil
    import tempfile

    passed, failed = 0, 0

    def run(name, fn):
        global passed, failed
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1

    def fresh_store():
        tmp = Path(tempfile.mkdtemp())
        return tmp, DocumentStore(documents_root=tmp)

    def test_add_get_document():
        tmp, store = fresh_store()
        try:
            (tmp / "doc1.txt").write_text("hello world", encoding="utf-8")
            store.add_document(Document(
                document_id="d1", filename="doc1.txt", relative_path="doc1.txt",
            ))
            assert store.get_document("d1") is not None
            assert store.load_document_bytes("d1") == b"hello world"
            assert store.load_document_bytes("d1", max_bytes=5) == b"hello"
        finally:
            shutil.rmtree(tmp)

    def test_add_chunks_and_get():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="d1", filename="x", relative_path="x"))
            store.add_chunks([
                Chunk(chunk_id=1, document_id="d1", text="a", position=0),
                Chunk(chunk_id=2, document_id="d1", text="b", position=1),
            ])
            assert len(store) == 2
            assert store.get_chunk(1).text == "a"
            assert [c.chunk_id for c in store.get_chunks([2, 1])] == [2, 1]
            assert len(store.get_chunks([999])) == 0
        finally:
            shutil.rmtree(tmp)

    def test_duplicate_chunk_raises():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="d1", filename="x", relative_path="x"))
            store.add_chunks([Chunk(chunk_id=1, document_id="d1", text="a", position=0)])
            try:
                store.add_chunks([Chunk(chunk_id=1, document_id="d1", text="b", position=0)])
            except ValueError:
                return
            raise AssertionError("ожидался ValueError на дубликат chunk_id")
        finally:
            shutil.rmtree(tmp)

    def test_add_chunks_atomic():
        """Если в пачке есть дубликат — не должно добавиться НИЧЕГО."""
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="d1", filename="x", relative_path="x"))
            store.add_chunks([Chunk(chunk_id=5, document_id="d1", text="old", position=0)])
            try:
                store.add_chunks([
                    Chunk(chunk_id=1, document_id="d1", text="new1", position=1),
                    Chunk(chunk_id=5, document_id="d1", text="dup",  position=2),  # конфликт
                    Chunk(chunk_id=2, document_id="d1", text="new2", position=3),
                ])
            except ValueError:
                # ничего из пачки не должно добавиться
                assert 1 not in store
                assert 2 not in store
                assert 5 in store
                return
            raise AssertionError("ожидался ValueError")
        finally:
            shutil.rmtree(tmp)

    def test_next_chunk_id_sync():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="d1", filename="x", relative_path="x"))
            assert store.next_chunk_id == 0
            store.add_chunks([Chunk(chunk_id=0, document_id="d1", text="a", position=0)])
            assert store.next_chunk_id == 1
            store.add_chunks([
                Chunk(chunk_id=5, document_id="d1", text="b", position=1),
                Chunk(chunk_id=6, document_id="d1", text="c", position=2),
            ])
            assert store.next_chunk_id == 7
            # добавление с меньшими id не откатывает счётчик
            store.add_chunks([Chunk(chunk_id=3, document_id="d1", text="d", position=3)])
            assert store.next_chunk_id == 7
        finally:
            shutil.rmtree(tmp)

    def test_resolve_document_id():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="solar", filename="x", relative_path="x"))
            assert store.resolve_document_id("solar")   == "solar_1"
            assert store.resolve_document_id("askue")   == "askue"
            store.add_document(Document(document_id="solar_1", filename="x", relative_path="x"))
            assert store.resolve_document_id("solar")   == "solar_2"
        finally:
            shutil.rmtree(tmp)

    def test_parent_context():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="d1", filename="x", relative_path="x"))
            store.add_chunks([
                Chunk(chunk_id=1, document_id="d1", text="PARENT BIG", position=0),
                Chunk(chunk_id=2, document_id="d1", text="child1", position=1, parent_chunk_id=1),
                Chunk(chunk_id=3, document_id="d1", text="child2", position=2, parent_chunk_id=1),
            ])
            ctx = store.get_chunk_with_context(2, mode=ContextMode.PARENT)
            assert ctx.chunk.chunk_id == 2
            assert ctx.parent.chunk_id == 1
            assert ctx.context_text == "PARENT BIG"
            ctx = store.get_chunk_with_context(1, mode=ContextMode.PARENT)
            assert ctx.parent is None
            assert ctx.context_text == "PARENT BIG"
        finally:
            shutil.rmtree(tmp)

    def test_window_context():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="d1", filename="x", relative_path="x"))
            store.add_chunks(
                Chunk(chunk_id=i, document_id="d1", text=f"t{i}", position=i)
                for i in range(1, 6)
            )
            ctx = store.get_chunk_with_context(3, mode=ContextMode.WINDOW, window_size=1)
            assert [c.chunk_id for c in ctx.window] == [2, 3, 4]
            assert ctx.context_text == "t2\n\nt3\n\nt4"
            ctx = store.get_chunk_with_context(1, mode=ContextMode.WINDOW, window_size=1)
            assert [c.chunk_id for c in ctx.window] == [1, 2]
        finally:
            shutil.rmtree(tmp)

    def test_save_load_roundtrip():
        tmp, store = fresh_store()
        try:
            (tmp / "doc1.md").write_text("# hi", encoding="utf-8")
            store.add_document(Document(
                document_id="d1", filename="doc1.md",
                relative_path="doc1.md", mime_type="text/markdown",
            ))
            store.add_chunks([
                Chunk(chunk_id=1, document_id="d1", text="parent", position=0, section_title="Intro"),
                Chunk(chunk_id=2, document_id="d1", text="child",  position=1, parent_chunk_id=1),
            ])
            store.save(tmp)

            store2 = DocumentStore.load(tmp, documents_root=tmp)
            assert len(store2) == 2
            assert store2.get_chunk(2).parent_chunk_id == 1
            assert store2.get_chunk(1).section_title == "Intro"
            assert store2.get_document("d1").mime_type == "text/markdown"
            assert store2.load_document_bytes("d1") == b"# hi"
            assert store2.next_chunk_id == 3  # счётчик восстановлен из данных
            ctx = store2.get_chunk_with_context(2, mode=ContextMode.PARENT)
            assert ctx.context_text == "parent"
        finally:
            shutil.rmtree(tmp)
    
    def test_validate():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(document_id="d1", filename="x", relative_path="x"))
            store.add_chunks([Chunk(
                chunk_id=1, document_id="d1", text="a", position=0,
                parent_chunk_id=999,
            )])
            errors = store.validate()
            assert len(errors) == 1 and "999" in errors[0]
        finally:
            shutil.rmtree(tmp)

    def test_missing_document_file():
        tmp, store = fresh_store()
        try:
            store.add_document(Document(
                document_id="d1", filename="ghost.md", relative_path="ghost.md"))
            try:
                store.load_document_bytes("d1")
            except FileNotFoundError:
                return
            raise AssertionError("ожидался FileNotFoundError")
        finally:
            shutil.rmtree(tmp)

    print("DocumentStore unit tests")
    print("-" * 40)
    run("add/get document",              test_add_get_document)
    run("add chunks, get by id",         test_add_chunks_and_get)
    run("duplicate chunk raises",        test_duplicate_chunk_raises)
    run("add_chunks is atomic",          test_add_chunks_atomic)
    run("next_chunk_id auto-sync",       test_next_chunk_id_sync)
    run("resolve_document_id",           test_resolve_document_id)
    run("small-to-big: parent",          test_parent_context)
    run("small-to-big: window",          test_window_context)
    run("save/load roundtrip",           test_save_load_roundtrip)
    run("validate dangling parent",      test_validate)
    run("missing original raises",       test_missing_document_file)
    print("-" * 40)
    print(f"Итого: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)