from faiss import IndexFlatIP, IndexIDMap2, write_index, read_index
from pathlib import Path
import numpy as np
from ..structs import Embedding, VectorHit
from typing import List, Tuple, Union, Optional
import warnings

BASE_NAME = Path("vector_data.faiss")
TOP_K: int = 5 

class VectorStore:
    def __init__(self, dim: int):
        self._dim = dim
        self._index: IndexIDMap2 = IndexIDMap2(IndexFlatIP(dim))

    def __len__(self) -> int:
        return self._index.ntotal

    def _check_dim(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"Ожидались векторы формы (N, {self._dim}), получено {vectors.shape}")

    def add(
        self, 
        embedding: Union[Embedding, List[Embedding]], 
        idx: Union[int, List[int], np.ndarray]
    ) -> None:
        """Добавить эмбеддинги с явными ID (ID должны совпадать с DocumentStore)."""
        if isinstance(embedding, list):
            vectors = np.vstack([emb.normalized().vector for emb in embedding])
        else:
            vectors = embedding.normalized().vector.reshape(1, -1)

        vectors = vectors.astype('float32', copy=False)  # уже float32, no-op
        
        self._check_dim(vectors)

        ids = np.asarray(idx if not isinstance(idx, int) else [idx], dtype="int64")
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(f"Количество векторов ({vectors.shape[0]}) и ID ({ids.shape[0]}) не совпадает")

        self._index.add_with_ids(vectors, ids)

    def search(
        self,
        query: Embedding,  # Убрали List[Embedding]
        top_k: int = TOP_K,
        min_score: Optional[float] = None,
    ) -> List[VectorHit]:
        """Поиск по ОДНОМУ эмбеддингу."""
        if len(self) == 0:
            return []

        # Всегда делаем форму (1, D)
        query_vector = query.normalized().vector.reshape(1, -1)
        self._check_dim(query_vector)

        scores, ids = self._index.search(query_vector, top_k)

        hits: List[VectorHit] = []
        # scores и ids имеют форму (1, top_k), поэтому scores[0] и ids[0] — это просто один список
        for score, cid in zip(scores[0], ids[0]):
            if cid == -1:
                continue
            if min_score is not None and float(score) < min_score:
                continue
            hits.append(VectorHit(chunk_id=int(cid), score=float(score)))
        return hits

    def search_batch(
        self,
        queries: List[Embedding],
        top_k: int = TOP_K,
        min_score: Optional[float] = None,
    ) -> List[List[VectorHit]]:
        if len(self) == 0 or not queries:
            return []

        query_vectors = np.vstack([q.normalized().vector for q in queries])
        self._check_dim(query_vectors)

        scores, ids = self._index.search(query_vectors, top_k)
        
        all_hits = []
        # Проходимся по всем N запросам
        for q_idx in range(len(queries)):
            hits = []
            for score, cid in zip(scores[q_idx], ids[q_idx]):
                if cid == -1:
                    continue
                if min_score is not None and float(score) < min_score:
                    continue
                hits.append(VectorHit(chunk_id=int(cid), score=float(score)))
            all_hits.append(hits)
            
        return all_hits
    
    def get(self, idx: Union[int, List[int]]) -> Union[Embedding, List[Embedding]]:
        """Восстановить эмбеддинг(и) по user ID. Debug-only."""
        if isinstance(idx, list):
            return [Embedding(self._index.reconstruct(int(i))) for i in idx]
        return Embedding(self._index.reconstruct(int(idx)))

    def save(self, storage_dir: Union[Path, str]) -> None:
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        write_index(self._index, str(storage_dir / BASE_NAME))

    @classmethod
    def load(cls, storage_dir: Union[Path, str]) -> "VectorStore":
        storage_dir = Path(storage_dir)
        fpath = storage_dir / BASE_NAME
        if not fpath.exists():
            raise FileNotFoundError(f"FAISS index not found: {fpath}")

        index = read_index(str(fpath))
        if not isinstance(index, IndexIDMap2):
            raise ValueError(
                f"Ожидался IndexIDMap2, получен {type(index).__name__}. "
                f"Пересоберите индекс с add_with_ids на IndexIDMap2."
            )
        obj = cls(dim=index.d)
        obj._index = index
        return obj

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def index(self):
        return self._index
    
if __name__ == "__main__":
    import tempfile
    import shutil

    np.random.seed(47)
    DIM = 47

    # ---------- lightweight test harness ----------
    passed, failed = 0, 0

    def assert_close(a: float, b: float, tol: float = 1e-5, msg: str = ""):
        if abs(a - b) > tol:
            raise AssertionError(f"{msg}: {a:.6f} vs {b:.6f} (tol={tol})")

    def run(name, fn):
        global passed, failed
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed += 1

    # ---------- helpers ----------
    def random_unit_embs(n: int, dim: int = DIM) -> list[Embedding]:
        """Случайные векторы, близкие к равномерно распределённым по сфере.
        randn — чтобы косинус между независимыми векторами был около 0,
        а не ~0.75, как было бы с uniform[0,1]."""
        return [Embedding(np.random.randn(dim)) for _ in range(n)]

    # ---------- tests ----------
    def test_basic_search():
        embs = random_unit_embs(5)
        store = VectorStore(DIM)
        store.add(embs, list(range(5)))

        assert len(store) == 5, f"len={len(store)}"

        hits = store.search(embs[0], top_k=3)
        assert len(hits) == 3, f"expected 3 hits, got {len(hits)}"
        assert hits[0].chunk_id == 0, f"top hit is {hits[0].chunk_id}, expected 0"
        assert_close(hits[0].score, 1.0, 1e-4, "self-similarity должен быть ~1")

    def test_min_score_filters():
        embs = random_unit_embs(5)
        store = VectorStore(DIM)
        store.add(embs, list(range(5)))

        # Порог 0.5: случайные randn-векторы с cos ~0 отсекаются,
        # остаётся только точное совпадение с запросом
        hits = store.search(embs[0], top_k=5, min_score=0.5)
        assert len(hits) == 1, (
            f"expected 1 hit, got {len(hits)}: "
            f"{[(h.chunk_id, round(h.score, 3)) for h in hits]}"
        )
        assert hits[0].chunk_id == 0

        # Без порога — все 5
        assert len(store.search(embs[0], top_k=5)) == 5

    def test_cosine_similarity_semantics():
        """Проверяем, что Inner Product работает как cosine:
        векторы с одинаковым направлением и разной длиной дают score ≈ 1.
        Противоположный — ≈ -1. Если бы не было нормализации — упало бы."""
        base = np.random.randn(DIM)
        v_pos_small = Embedding(base * 1.0)
        v_pos_large = Embedding(base * 10.0)
        v_negative = Embedding(-base * 5.0)

        store = VectorStore(DIM)
        store.add([v_pos_small, v_pos_large, v_negative], [10, 20, 30])

        hits = store.search(v_pos_small, top_k=3)
        scores = {h.chunk_id: h.score for h in hits}

        assert_close(scores[10], 1.0, 1e-4, "same direction, small")
        assert_close(scores[20], 1.0, 1e-4, "same direction, large")
        assert_close(scores[30], -1.0, 1e-4, "opposite direction")

    def test_save_load_preserves_behavior():
        tmpdir = Path(tempfile.mkdtemp())
        try:
            embs = random_unit_embs(5)
            store1 = VectorStore(DIM)
            store1.add(embs, list(range(5)))
            store1.save(tmpdir)

            store2 = VectorStore.load(tmpdir)
            assert len(store2) == len(store1), (
                f"len mismatch: {len(store1)} vs {len(store2)}"
            )

            # Сравниваем поведение, а не внутреннее состояние
            h1 = store1.search(embs[0], top_k=3)
            h2 = store2.search(embs[0], top_k=3)
            assert [h.chunk_id for h in h1] == [h.chunk_id for h in h2], (
                f"id order mismatch: {[h.chunk_id for h in h1]} vs "
                f"{[h.chunk_id for h in h2]}"
            )
            for a, b in zip(h1, h2):
                assert_close(a.score, b.score, 1e-6, f"score for id={a.chunk_id}")
        finally:
            shutil.rmtree(tmpdir)

    def test_get_by_id():
        raw = random_unit_embs(3)
        store = VectorStore(DIM)
        store.add(raw, [100, 200, 300])

        e_single = store.get(100)
        assert isinstance(e_single, Embedding)
        np.testing.assert_allclose(
            e_single.vector,
            raw[0].normalized().vector,
            atol=1e-6,
            err_msg="get(100) должен вернуть нормализованный raw[0]",
        )

    def test_empty_store():
        store = VectorStore(DIM)
        assert len(store) == 0
        assert store.search(Embedding(np.random.randn(DIM)), top_k=5) == []

    def test_top_k_limit():
        embs = random_unit_embs(10)
        store = VectorStore(DIM)
        store.add(embs, list(range(10)))

        # Меньше ntotal
        assert len(store.search(embs[0], top_k=3)) == 3

        # Больше ntotal: FAISS вернёт -1, VectorStore должен отфильтровать
        hits = store.search(embs[0], top_k=20)
        assert len(hits) == 10, f"expected 10 (ntotal), got {len(hits)}"
        assert all(h.chunk_id != -1 for h in hits)

    def test_dimension_validation():
        store = VectorStore(DIM)
        wrong = Embedding(np.random.randn(DIM + 1))
        try:
            store.add(wrong, 0)
        except ValueError:
            return
        raise AssertionError("add должен был упасть на неверной размерности")

    # ---------- run ----------
    print("VectorStore unit tests")
    print("-" * 40)
    run("basic search",                  test_basic_search)
    run("min_score filters",             test_min_score_filters)
    run("cosine (not inner product)",    test_cosine_similarity_semantics)
    run("save/load preserves behavior",  test_save_load_preserves_behavior)
    run("get by id",                     test_get_by_id)
    run("empty store",                   test_empty_store)
    run("top_k limit",                   test_top_k_limit)
    run("dimension validation",          test_dimension_validation)
    print("-" * 40)
    print(f"Итого: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)