"""
Entrypoint локального грейдера.

Usage:
    # 1. Сгенерировать наши ответы
    python solution.py          # → submission.csv

    # 2. Оценить
    python -m evaluation.main

    # Опции
    python -m evaluation.main --judge-model GigaChat-3-ULTRA
    python -m evaluation.main --submission path/to/custom.csv
    python -m evaluation.main --skip-retrieval
    python -m evaluation.main --workers 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .config import load_judge_settings, load_paths
from .data_loader import load_answers_csv, load_questions_csv
from .judge import LLMJudge, build_judge_client
from .metrics import QueryResult, compute_metrics
from .report import print_report, save_json_report
from .retrieval_eval import aggregate_retrieval, evaluate_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# Judge evaluation
# ---------------------------------------------------------------------------

def run_judge_evaluation(
    questions: dict[str, str],
    ours: dict[str, str],
    baseline: dict[str, str],
    golden: dict[str, str] | None,
    judge: LLMJudge,
    workers: int = 2,
) -> list[QueryResult]:
    def _eval_one(qid: str, question: str) -> QueryResult:
        r = QueryResult(
            qid=qid, question=question,
            ours=ours.get(qid, ""),
            baseline=baseline.get(qid, ""),
            golden=golden.get(qid) if golden else None,
        )
        if not r.ours or not r.baseline:
            logger.warning("[%s] skip: ours=%d, baseline=%d chars",
                           qid, len(r.ours), len(r.baseline))
            r.judge_error = True
            return r
        try:
            r1, r2, avg = judge.evaluate_pair(
                question=question, ours=r.ours, baseline=r.baseline,
                golden=r.golden,
            )
            r.r1, r.r2, r.score = r1, r2, avg
            logger.info("[%s] r1=%.2f r2=%.2f → %.2f", qid, r1, r2, avg)
        except Exception as e:
            logger.exception("[%s] judge failed: %s", qid, e)
            r.judge_error = True
        return r

    results: list[QueryResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one, qid, q): qid
                   for qid, q in questions.items()}
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: int(r.qid) if r.qid.isdigit() else 0)
    return results


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------

def run_retrieval_evaluation(questions: dict[str, str]):
    """Проверяет, какие документы попадают в контекст отвечающей LLM."""
    try:
        from RAG.Embedding import EmbeddingWrapper
        from RAG.Ranking import LocalReranker
        from RAG.metrics import GROUND_TRUTH
        from RAG.pipelines import (
            QUERY_INSTRUCTIONS,
            pipeline_load,
            pipeline_search_by_text,
        )
    except Exception as e:
        logger.warning("RAG не загрузился, retrieval eval пропущен: %s", e)
        return None, None

    from .config import PROJECT_ROOT
    kb_dir = Path(os.getenv(
        "RAG_KB_DIR",
        PROJECT_ROOT / "RAG" / "Store" / "_test_" / "Data Knowledge",
    ))
    index_dir = Path(os.getenv(
        "RAG_INDEX_DIR",
        PROJECT_ROOT / "RAG" / "Store" / "Storage",
    ))

    try:
        embedder = EmbeddingWrapper()
        reranker = None
        try:
            reranker = LocalReranker(device="cpu")
        except Exception as e:
            logger.warning("Reranker not loaded: %s", e)
        doc_store, vec_store = pipeline_load(index_dir, kb_dir)
    except Exception as e:
        logger.warning("RAG pipeline load failed: %s", e)
        return None, None

    gt = {str(item["id"]): item["relevant_documents"] for item in GROUND_TRUTH}

    def retrieval_fn(query_text: str) -> list[str]:
        hits = pipeline_search_by_text(
            embedder, doc_store, vec_store, query_text,
            query_instruction=QUERY_INSTRUCTIONS["detailed_ru"],
            top_k=5, reranker=reranker, rerank_oversample=3,
            filter_toc=True,
        )
        return [h.chunk.document_id for h in hits]

    results = evaluate_retrieval(questions, gt, retrieval_fn)
    return results, aggregate_retrieval(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local grader for EnergyTool AI (MTS-methodology)",
    )
    parser.add_argument("--questions")
    parser.add_argument("--baseline")
    parser.add_argument("--submission")
    parser.add_argument("--golden")
    parser.add_argument("--output-dir")
    parser.add_argument("--judge-model", help="GigaChat-2-MAX | GigaChat-3-ULTRA | ...")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    # CLI overrides
    env_map = {
        "EVAL_QUESTIONS": args.questions,
        "EVAL_BASELINE": args.baseline,
        "EVAL_SUBMISSION": args.submission,
        "EVAL_GOLDEN": args.golden,
        "EVAL_OUTPUT_DIR": args.output_dir,
        "JUDGE_MODEL": args.judge_model,
    }
    for k, v in env_map.items():
        if v:
            os.environ[k] = v

    paths = load_paths()
    settings = load_judge_settings()

    if not settings.auth_key:
        logger.error("Нет ключа: GIGACHAT_AUTH_KEY / AI_AUTH_KEY")
        return 2

    logger.info("Questions : %s", paths.questions)
    logger.info("Baseline  : %s", paths.baseline)
    logger.info("Submission: %s", paths.submission)
    logger.info("Judge     : %s @ %s", settings.model, settings.base_url)

    questions = load_questions_csv(paths.questions)
    ours = load_answers_csv(paths.submission)
    baseline = load_answers_csv(paths.baseline)
    golden = (
        load_answers_csv(paths.golden)
        if paths.golden and paths.golden.exists()
        else None
    )

    if not ours:
        logger.error("submission.csv пустой/отсутствует. Сначала запусти solution.py")
        return 2

    # 1. Retrieval
    ret_results, ret_agg = (None, None)
    if not args.skip_retrieval:
        logger.info("Running retrieval evaluation...")
        ret_results, ret_agg = run_retrieval_evaluation(questions)

    # 2. Judge
    logger.info("Running judge evaluation (workers=%d)...", args.workers)
    llm = build_judge_client(settings)
    judge = LLMJudge(settings, llm)
    try:
        results = run_judge_evaluation(
            questions, ours, baseline, golden, judge,
            workers=args.workers,
        )
    finally:
        llm.close()

    # Attach retrieval info
    if ret_results:
        by_qid = {r.qid: r for r in ret_results}
        for r in results:
            if r.qid in by_qid:
                r.retrieved_docs = by_qid[r.qid].retrieved_docs

    # 3. Metrics + report
    agg = compute_metrics(results)
    meta = {
        "judge_model": settings.model,
        "questions": len(questions),
        "workers": args.workers,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    print_report(results, agg, ret_results, ret_agg, meta)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = paths.output_dir / f"eval_{ts}.json"
    save_json_report(out, results, agg, ret_results, ret_agg, meta)
    logger.info("Report saved: %s", out)

    return 0 if agg.score >= 0.5 else 1


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\evaluation\.env")
    
    sys.exit(main())