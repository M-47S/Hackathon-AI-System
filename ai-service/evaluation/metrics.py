from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryResult:
    qid: str
    question: str
    ours: str
    baseline: str
    golden: Optional[str] = None
    r1: Optional[float] = None
    r2: Optional[float] = None
    score: Optional[float] = None
    judge_error: bool = False
    retrieved_docs: list[str] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    match_rate: float
    win_rate: float
    score: float
    n_wins: int
    n_losses: int
    n_ties: int
    n_errors: int
    n_total: int


def compute_metrics(results: list[QueryResult]) -> AggregateMetrics:
    """Score = 0.7 * MatchRate + 0.3 * WinRate — формула МТС."""
    valid = [r for r in results if r.score is not None and not r.judge_error]
    n = len(valid)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0, 0, len(results), len(results))

    match_rate = sum(r.score for r in valid) / n

    wins = sum(1 for r in valid if r.score > 0.5)
    losses = sum(1 for r in valid if r.score < 0.5)
    ties = sum(1 for r in valid if r.score == 0.5)

    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
    score = 0.7 * match_rate + 0.3 * win_rate

    return AggregateMetrics(
        match_rate=match_rate, win_rate=win_rate, score=score,
        n_wins=wins, n_losses=losses, n_ties=ties,
        n_errors=len(results) - n, n_total=len(results),
    )


def classify(score: float) -> str:
    if score >= 0.875: return "STRONG WIN ▲▲"
    if score > 0.5:    return "WIN ▲"
    if score == 0.5:   return "TIE ─"
    if score > 0:      return "LOSS ▼"
    return "STRONG LOSS ▼▼"