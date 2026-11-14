from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class JudgeSettings:
    model: str
    base_url: str
    auth_key: str
    scope: str
    temperature: float = 0.0
    max_tokens: int = 10


def load_judge_settings() -> JudgeSettings:
    return JudgeSettings(
        model=os.getenv("JUDGE_MODEL", "GigaChat-2-MAX"),
        base_url=os.getenv("JUDGE_BASE_URL", "https://api.giga.chat/v1"),
        auth_key=(os.getenv("GIGACHAT_AUTH_KEY") or os.getenv("AI_AUTH_KEY") or ""),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
    )


@dataclass
class EvalPaths:
    questions: Path
    baseline: Path
    submission: Path
    golden: Path | None
    output_dir: Path


def load_paths() -> EvalPaths:
    def _p(env: str, default: Path) -> Path:
        v = os.getenv(env)
        return Path(v) if v else default

    golden_env = os.getenv("EVAL_GOLDEN")
    return EvalPaths(
        questions=_p("EVAL_QUESTIONS", PROJECT_ROOT / "demo_questions.csv"),
        baseline=_p("EVAL_BASELINE", PROJECT_ROOT / "demo_output.csv"),
        submission=_p("EVAL_SUBMISSION", PROJECT_ROOT / "submission.csv"),
        golden=Path(golden_env) if golden_env else None,
        output_dir=_p("EVAL_OUTPUT_DIR", PROJECT_ROOT / "evaluation" / "results"),
    )