"""
RAG/metrics.py — офлайн-оценка качества retrieval.

Метрики:
  - Recall@k      — доля релевантных документов, попавших в top-k
  - Precision@k   — доля релевантных среди top-k
  - Hit@k         — есть ли хотя бы один релевантный в top-k
  - MRR           — 1 / позиция первого релевантного (усреднённо)

Ground truth: вручную размеченные пары (вопрос → релевантные document_id)
на основе demo_questions.csv и файлов базы знаний.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
# Релевантность: какие document_id (stem .md-файла) отвечают на вопрос.
# Если вопрос касается нескольких тем — перечислены несколько документов.
# Порядок внутри списка не важен.

GROUND_TRUTH: List[Dict] = [
    {
        "id": 1,
        "question": "У нас крыша на два ската — восток и запад. Панели можно поставить на оба?",
        "relevant_documents": ["23-solar"],
    },
    {
        "id": 2,
        "question": "На крыше почти нет места. А солнечные панели на фасад ставите?",
        "relevant_documents": ["23-solar"],
    },
    {
        "id": 3,
        "question": "Можно сначала поставить солнечную станцию на 60 кВт, а потом расширить?",
        "relevant_documents": ["23-solar", "01-numbers-and-indicators"],
    },
    {
        "id": 4,
        "question": "У нас старая шиферная крыша. Стоит вообще рассматривать солнечные панели?",
        "relevant_documents": ["23-solar", "02-product-scope-and-limits"],
    },
    {
        "id": 5,
        "question": "В счётчиках стоят симки другого оператора. Для вашей платформы их надо менять?",
        "relevant_documents": ["22-askue", "13-metering"],
    },
    {
        "id": 6,
        "question": "Хочу видеть расход электричества отдельно по цехам. В вашем кабинете так можно?",
        "relevant_documents": ["22-askue"],
    },
    {
        "id": 7,
        "question": "У нас несколько магазинов. Можно смотреть их расход в одном кабинете?",
        "relevant_documents": ["22-askue", "20-portfolio"],
    },
    {
        "id": 8,
        "question": "У нас двухтарифные счётчики. По фото сервис оба показания разберёт?",
        "relevant_documents": ["25-meter-reading", "13-metering"],
    },
    {
        "id": 9,
        "question": "Сервис по фотографии сможет прочитать год выпуска счётчика?",
        "relevant_documents": ["25-meter-reading"],
    },
    {
        "id": 10,
        "question": "Мы стали больше работать вечером, и счёт вырос. Можно разобраться, из-за чего?",
        "relevant_documents": ["14-tariffs", "11-price-categories"],
    },
    {
        "id": 11,
        "question": "Нам предлагают компенсировать реактивную энергию. Сначала надо что-то измерить?",
        "relevant_documents": ["17-faq", "12-power-thresholds"],
    },
    {
        "id": 12,
        "question": "Хотим разобраться, подходит ли нам оптовый рынок. Вы сначала проверите наш учёт?",
        "relevant_documents": ["21-orem", "01-numbers-and-indicators"],
    },
    {
        "id": 13,
        "question": "Если расчёт покажет, что проект нам невыгоден, вы так и скажете?",
        "relevant_documents": ["21-orem", "03-calculation-and-requests"],
    },
    {
        "id": 14,
        "question": (
            "Пользователь: У нас небольшое производство, хотим поставить солнечную станцию.\n"
            "Ассистент: Какая у вас основная цель: сократить расходы или иметь резерв при отключениях?\n"
            "Пользователь: И то и другое. Но больше всего электричества тратим вечером.\n"
            "Ассистент: Тогда важно учесть время потребления и отдельно рассмотреть накопители.\n"
            "Пользователь: А обычная сетевая станция с вечерней нагрузкой вообще имеет смысл?"
        ),
        "relevant_documents": ["23-solar", "02-product-scope-and-limits"],
    },
    {
        "id": 15,
        "question": (
            "Пользователь: Хотим подключить к вашей платформе приборы в бизнес-центре.\n"
            "Ассистент: Какая задача: общий учёт или расходы отдельно по арендаторам?\n"
            "Пользователь: По арендаторам. У большинства счётчики уже стоят.\n"
            "Ассистент: Понял. Каким способом сейчас собираете показания?\n"
            "Пользователь: Кто-то присылает фото, а по некоторым ходит техник. С чего нам начать?"
        ),
        "relevant_documents": ["22-askue", "25-meter-reading"],
    },
    {
        "id": 16,
        "question": (
            "Пользователь: Мы электросетевая компания. Хотим лучше находить потери.\n"
            "Ассистент: Для этого есть сервис, который помогает определить точки для выездной проверки. Какие данные учёта у вас есть?\n"
            "Пользователь: По одним приборам почасовые, по другим только суточные.\n"
            "Ассистент: Понял. Какой архив накоплен?\n"
            "Пользователь: Примерно за два года. Таких данных хватит, чтобы начать?"
        ),
        "relevant_documents": ["24-losses"],
    },
    {
        "id": 17,
        "question": (
            "Пользователь: У нас завод, расходы на свет сильно выросли.\n"
            "Ассистент: Для начала нужно разделить рост потребления и изменение цены. Вы сравнивали счета за одинаковые периоды?\n"
            "Пользователь: Потребление выросло чуть-чуть, а сумма заметно.\n"
            "Ассистент: А какая у вас ценовая категория?\n"
            "Пользователь: Кажется, четвёртая. Что вообще стоит проверить в счёте?"
        ),
        "relevant_documents": ["14-tariffs", "11-price-categories", "10-market"],
    },
    {
        "id": 18,
        "question": (
            "Пользователь: Хотим понять, сколько будет стоить автоматический учёт на наших объектах.\n"
            "Ассистент: Стоимость зависит от количества приборов и конфигурации. Сколько объектов у вас?\n"
            "Пользователь: Три склада, всего около сорока счётчиков.\n"
            "Ассистент: Можно подготовить предварительную оценку по данным объектов.\n"
            "Пользователь: Я пока не готов оставлять телефон. Можете объяснить, что понадобится для оценки?"
        ),
        "relevant_documents": ["03-calculation-and-requests", "22-askue", "01-numbers-and-indicators"],
    },
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class QueryMetrics:
    query_id: str
    question: str
    retrieved: List[str]      # document_id в порядке выдачи
    relevant: List[str]
    recall: float
    precision: float
    hit: float
    mrr: float
    found_at: Optional[int]   # позиция первого релевантного (1-based) или None

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_ground_truth_csv(path: str | Path = "retrieval_ground_truth.csv") -> None:
    """Сохраняет GROUND_TRUTH в CSV для ручного просмотра/правки."""
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "relevant_documents"])
        writer.writeheader()
        for item in GROUND_TRUTH:
            writer.writerow({
                "id": item["id"],
                "question": item["question"],
                # несколько документов — через точку с запятой (внутри одного поля)
                "relevant_documents": ";".join(item["relevant_documents"]),
            })
    logger.info("Ground truth сохранён в %s", path)


def load_ground_truth_csv(path: str | Path) -> List[Dict]:
    """Загружает ground truth из CSV (формат: id, question, relevant_documents)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")
    rows: List[Dict] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": row["id"],
                "question": row["question"],
                "relevant_documents": [
                    x.strip() for x in row["relevant_documents"].split(";") if x.strip()
                ],
            })
    return rows


def _compute_query_metrics(
    query_id: str,
    question: str,
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int,
) -> QueryMetrics:
    retrieved = retrieved_docs[:k]
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved)

    # Recall: доля релевантных, попавших в top-k
    if relevant_set:
        recall = len(retrieved_set & relevant_set) / len(relevant_set)
    else:
        recall = 0.0

    # Precision: доля релевантных среди top-k
    precision = (
        len([d for d in retrieved if d in relevant_set]) / len(retrieved)
        if retrieved else 0.0
    )

    # Hit: 1 если хотя бы один релевантный в top-k, иначе 0
    hit = 1.0 if (retrieved_set & relevant_set) else 0.0

    # MRR: 1 / позиция первого релевантного
    found_at: Optional[int] = None
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            found_at = i
            break
    mrr = 1.0 / found_at if found_at else 0.0

    return QueryMetrics(
        query_id=query_id,
        question=question,
        retrieved=retrieved,
        relevant=relevant_docs,
        recall=recall,
        precision=precision,
        hit=hit,
        mrr=mrr,
        found_at=found_at,
    )


def evaluate_retrieval(
    retrieval_fn: Callable[[str, int], Sequence[str]],
    ground_truth: Optional[List[Dict]] = None,
    k: int = 5,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[QueryMetrics]]:
    """
    Оценивает retrieval по ground truth.

    Args:
        retrieval_fn: функция (query_text, top_k) -> список document_id.
        ground_truth: список словарей с полями id, question, relevant_documents.
                      Если None — используется встроенный GROUND_TRUTH.
        k:            размер top-k (по умолчанию 5).
        verbose:      печатать прогресс по каждому запросу.

    Returns:
        (агрегированные метрики, per-query детали)
    """
    if ground_truth is None:
        ground_truth = GROUND_TRUTH

    results: List[QueryMetrics] = []

    for item in ground_truth:
        qid = str(item.get("id", "?"))
        question = item["question"]
        relevant = item["relevant_documents"]

        try:
            retrieved_docs = list(retrieval_fn(question, k))
        except Exception as e:
            logger.exception("retrieval_fn упал на query_id=%s: %s", qid, e)
            retrieved_docs = []

        m = _compute_query_metrics(qid, question, retrieved_docs, relevant, k)
        results.append(m)

        if verbose:
            first = f"@{m.found_at}" if m.found_at else "—"
            print(
                f"   [{qid:>3s}] recall={m.recall:.2f} "
                f"hit={m.hit:.0f} MRR={m.mrr:.3f} first_rel={first} "
                f"top{k}={m.retrieved[:k]}"
            )

    n = len(results)
    if n == 0:
        return {"recall": 0.0, "precision": 0.0, "hit": 0.0, "mrr": 0.0}, []

    agg = {
        f"recall@{k}": sum(r.recall for r in results) / n,
        f"precision@{k}": sum(r.precision for r in results) / n,
        f"hit@{k}": sum(r.hit for r in results) / n,
        f"mrr@{k}": sum(r.mrr for r in results) / n,
    }
    return agg, results


def print_summary(agg: Dict[str, float], k: int) -> None:
    print("\n" + "=" * 60)
    print(f"📊 Итог по retrieval (n={len(GROUND_TRUTH)} запросов, k={k})")
    for name, value in agg.items():
        print(f"   {name:20s}: {value:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from .pipelines import (
        pipeline_search,
        QUERY_INSTRUCTIONS,
        _adaptive_max_per_document,
    )

    # Папки с данными и индексом (можно поменять)
    DATA_DIR = Path(
        r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service"
        r"\RAG\Store\_test_\Data Knowledge"
    )
    INDEX_DIR = Path(
        r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service"
        r"\RAG\Store\Storage"
    )

    # 1. Сохраняем ground truth в CSV (полезно, чтобы посмотреть глазами)
    save_ground_truth_csv(INDEX_DIR / "retrieval_ground_truth.csv")

    # 2. Загружаем индекс и модели
    print("🔧 Загрузка индекса и моделей...")
    from .Embedding import EmbeddingWrapper
    from .pipelines import pipeline_load

    embedder = EmbeddingWrapper()
    doc_store, vec_store = pipeline_load(INDEX_DIR, DATA_DIR)

    # Реранкер (опционально)
    reranker = None
    try:
        from .Ranking import LocalReranker
        reranker = LocalReranker(device="cpu")
        print("   ✅ LocalReranker загружен")
    except Exception as e:
        print(f"   ⚠️ Реранкер не загружен: {e}")

    # 3. Обёртка над вашим пайплайном, которая возвращает список document_id
    def retrieval_fn(query_text: str, top_k: int) -> List[str]:
        # Используем instruction prefix по умолчанию
        from .pipelines import pipeline_search_by_text
        hits = pipeline_search_by_text(
            embedder, doc_store, vec_store, query_text,
            query_instruction=QUERY_INSTRUCTIONS["detailed_ru"],
            top_k=top_k,
            min_score=0.0,
            filter_toc=True,
            max_per_document=_adaptive_max_per_document(query_text),
            reranker=reranker,
            rerank_oversample=5,
        )
        return [h.chunk.document_id for h in hits]

    # 4. Оценка
    print("\n🧪 Прогон оценки retrieval...")
    agg, per_query = evaluate_retrieval(
        retrieval_fn=retrieval_fn,
        k=5,
        verbose=True,
    )
    print_summary(agg, k=5)