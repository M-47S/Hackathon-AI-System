from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_questions_csv(path: Path) -> dict[str, str]:
    """id → question."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = {}
        for r in csv.DictReader(f):
            qid = str(r.get("id", "")).strip()
            q = str(r.get("question", "")).strip()
            if qid and q:
                rows[qid] = q
    logger.info("Loaded %d questions from %s", len(rows), path)
    return rows


def load_answers_csv(path: Path) -> dict[str, str]:
    """id → answer."""
    if not path.exists():
        logger.warning("Answers file not found: %s", path)
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = {}
        for r in csv.DictReader(f):
            qid = str(r.get("id", "")).strip()
            a = str(r.get("answer", "")).strip()
            if qid:
                rows[qid] = a
    logger.info("Loaded %d answers from %s", len(rows), path)
    return rows