"""Оценка retrieval: какие документы попадают в контекст отвечающей LLM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    qid: str
    question: str
    retrieved_docs: list[str]
    relevant_docs: list[str]
    hit: float
    mrr: float
    recall: float
    found_at: Optional[int]


@dataclass
class RetrievalAggregate:
    hit_at_k: float
    mrr_at_k: float
    recall_at_k: float
    avg_retrieved: float
    n: int


def evaluate_retrieval(
    questions: dict[str, str],
    ground_truth: dict[str, list[str]],
    retrieval_fn: Callable[[str], list[str]],
) -> list[RetrievalResult]:
    results = []
    for qid, question in questions.items():
        relevant = ground_truth.get(qid, [])
        if not relevant:
            continue
        try:
            retrieved = retrieval_fn(question)
        except Exception as e:
            logger.exception("[%s] retrieval failed: %s", qid, e)
            retrieved = []

        relevant_set = set(relevant)
        retrieved_set = set(retrieved)

        hit = 1.0 if (retrieved_set & relevant_set) else 0.0
        recall = len(retrieved_set & relevant_set) / len(relevant)

        found_at = None
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                found_at = i
                break
        mrr = 1.0 / found_at if found_at else 0.0

        results.append(RetrievalResult(
            qid=qid, question=question,
            retrieved_docs=retrieved, relevant_docs=relevant,
            hit=hit, mrr=mrr, recall=recall, found_at=found_at,
        ))
    return results


def aggregate_retrieval(results: list[RetrievalResult]) -> RetrievalAggregate:
    n = len(results)
    if n == 0:
        return RetrievalAggregate(0, 0, 0, 0, 0)
    return RetrievalAggregate(
        hit_at_k=sum(r.hit for r in results) / n,
        mrr_at_k=sum(r.mrr for r in results) / n,
        recall_at_k=sum(r.recall for r in results) / n,
        avg_retrieved=sum(len(r.retrieved_docs) for r in results) / n,
        n=n,
    )