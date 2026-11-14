from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .metrics import AggregateMetrics, QueryResult, classify
from .retrieval_eval import RetrievalAggregate, RetrievalResult


def print_report(
    results: list[QueryResult],
    agg: AggregateMetrics,
    retrieval_results: list[RetrievalResult] | None = None,
    retrieval_agg: RetrievalAggregate | None = None,
    meta: dict | None = None,
) -> None:
    print("\n" + "=" * 72)
    print("  EVALUATION REPORT")
    if meta:
        for k, v in meta.items():
            print(f"  {k}: {v}")
    print("=" * 72)

    print("\n[ ANSWER QUALITY — LLM Judge (MTS methodology) ]")
    print(f"  MatchRate : {agg.match_rate:.4f}")
    print(f"  WinRate   : {agg.win_rate:.4f}  "
          f"(W={agg.n_wins}, L={agg.n_losses}, T={agg.n_ties})")
    print(f"  ─────────────────────────────────")
    print(f"  Score     : {agg.score:.4f}   ← прогноз MTS grader")
    if agg.n_errors:
        print(f"  ⚠️  Judge errors: {agg.n_errors}/{agg.n_total}")

    print("\n[ PER-QUESTION ]")
    for r in results:
        if r.score is None:
            print(f"  [{r.qid:>3}] ERROR (judge failed)")
            continue
        print(f"  [{r.qid:>3}] {r.score:.2f}  "
              f"(r1={r.r1:.2f} r2={r.r2:.2f})  {classify(r.score)}")

    if retrieval_agg:
        print("\n[ RETRIEVAL QUALITY ]")
        print(f"  Hit@k    : {retrieval_agg.hit_at_k:.4f}  "
              f"({retrieval_agg.hit_at_k * retrieval_agg.n:.0f}/{retrieval_agg.n})")
        print(f"  MRR@k    : {retrieval_agg.mrr_at_k:.4f}")
        print(f"  Recall@k : {retrieval_agg.recall_at_k:.4f}")
        print(f"  Avg docs : {retrieval_agg.avg_retrieved:.1f}")

    failures = [r for r in results if r.score is not None and r.score < 0.5]
    if failures:
        print(f"\n[ TOP FAILURES ({len(failures)}) — fix priority ]")
        for r in sorted(failures, key=lambda x: x.score)[:5]:
            print(f"\n  ╭─ [{r.qid}] score={r.score:.2f}")
            print(f"  │ Q: {r.question[:150]}")
            print(f"  │ OURS:     {r.ours[:200]}")
            print(f"  │ BASELINE: {r.baseline[:200]}")
            if r.retrieved_docs:
                print(f"  ╰─ Docs: {r.retrieved_docs}")

    wins = [r for r in results if r.score is not None and r.score > 0.5]
    if wins:
        print(f"\n[ TOP WINS ({len(wins)}) ]")
        for r in sorted(wins, key=lambda x: -x.score)[:3]:
            print(f"  [{r.qid}] {r.score:.2f}: {r.ours[:120]}")

    print("\n[ STYLE DIAGNOSTICS ]")
    _style_stats(results)
    print("=" * 72)


def _style_stats(results: list[QueryResult]) -> None:
    valid = [r for r in results if r.ours and r.baseline]
    if not valid:
        return
    our_lens = [len(r.ours.split()) for r in valid]
    base_lens = [len(r.baseline.split()) for r in valid]
    our_q = sum(1 for r in valid if "?" in r.ours)
    base_q = sum(1 for r in valid if "?" in r.baseline)
    our_md = sum(1 for r in valid if any(m in r.ours for m in ("**", "##", "```")))
    base_md = sum(1 for r in valid if any(m in r.baseline for m in ("**", "##", "```")))
    n = len(valid)
    print(f"  Avg length: ours={statistics.mean(our_lens):.0f}w  "
          f"baseline={statistics.mean(base_lens):.0f}w")
    print(f"  With '?':   ours={our_q}/{n}  baseline={base_q}/{n}")
    print(f"  Markdown:   ours={our_md}/{n}  baseline={base_md}/{n}")


def save_json_report(
    path: Path,
    results: list[QueryResult],
    agg: AggregateMetrics,
    retrieval_results: list[RetrievalResult] | None,
    retrieval_agg: RetrievalAggregate | None,
    meta: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "answer_quality": {
            "match_rate": round(agg.match_rate, 4),
            "win_rate": round(agg.win_rate, 4),
            "score": round(agg.score, 4),
            "n_wins": agg.n_wins, "n_losses": agg.n_losses,
            "n_ties": agg.n_ties, "n_errors": agg.n_errors,
            "n_total": agg.n_total,
        },
        "per_question": [
            {
                "qid": r.qid, "score": r.score, "r1": r.r1, "r2": r.r2,
                "question": r.question,
                "ours": r.ours, "baseline": r.baseline,
                "retrieved_docs": r.retrieved_docs,
            }
            for r in results
        ],
    }
    if retrieval_agg:
        payload["retrieval_quality"] = {
            "hit_at_k": round(retrieval_agg.hit_at_k, 4),
            "mrr_at_k": round(retrieval_agg.mrr_at_k, 4),
            "recall_at_k": round(retrieval_agg.recall_at_k, 4),
            "avg_retrieved": round(retrieval_agg.avg_retrieved, 1),
        }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")