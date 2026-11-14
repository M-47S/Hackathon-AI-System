"""
LLM-судья, эквивалентный судье MTS-грейдера.

Методика:
    Round 1: A=baseline, B=ours  → r1 (насколько ours лучше baseline)
    Round 2: A=ours,     B=baseline → r2_raw → r2 = 1 - r2_raw (инверсия)
    score = (r1 + r2) / 2

Шкала {0.00, 0.25, 0.50, 0.75, 1.00} — как у МТС.
Golden (если есть) передаётся судье как эталон для справки.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from LLM import ChatRequest, LLMClient, LLMMessage, Role
from LLM.auth import GigaChatTokenProvider
from LLM.exceptions import LLMError

from .config import JudgeSettings

logger = logging.getLogger(__name__)

SCORE_VALUES = {0.0, 0.25, 0.5, 0.75, 1.0}


JUDGE_SYSTEM_PROMPT = """\
Ты — опытный эксперт-оценщик качества ответов ИИ-консультанта по продуктам \
МТС EnergyTool (решения для учёта и экономии электроэнергии).

Твоя задача — сравнить два ответа на один вопрос клиента и определить, \
насколько ответ B лучше или хуже ответа A. Оба ответа получены от \
ИИ-консультанта, работающего по одной и той же базе знаний.

Шкала:
- 1.00 — ответ B значительно лучше ответа A
- 0.75 — ответ B немного лучше ответа A
- 0.50 — ответы сопоставимы по качеству
- 0.25 — ответ B немного хуже ответа A
- 0.00 — ответ B значительно хуже ответа A

Критерии (по важности):
1. ФАКТИЧЕСКАЯ ТОЧНОСТЬ — нет выдумок, цен, гарантий, характеристик, которых \
не существует. Упоминаются только реальные продукты EnergyTool.
2. РЕЛЕВАНТНОСТЬ — ответ прямо отвечает на последний вопрос клиента, \
учитывает контекст диалога (если он есть).
3. СТРУКТУРА — сначала прямой ответ (Да/Нет/Зависит/конкретный продукт), \
затем краткое обоснование, в конце — уместный уточняющий вопрос.
4. ПОЛЕЗНОСТЬ — ответ помогает клиенту продвинуться, есть следующий шаг.
5. ЯЗЫК — естественный, экспертный, без воды и markdown, до 150 слов.

# Формат ответа

Верни ТОЛЬКО одно число из набора {1.00, 0.75, 0.50, 0.25, 0.00}.
Без пояснений. Без markdown. Только число.

Примеры корректного ответа: `0.75` или `0.50` или `1.00`.
"""


JUDGE_USER_TEMPLATE = """\
Вопрос клиента:
{question}

Ответ A:
{answer_a}

Ответ B:
{answer_b}
{golden_block}

Оцени ответ B относительно ответа A по шкале. Верни только число.
"""


@dataclass
class JudgeVerdict:
    score: float
    raw_text: str
    valid: bool


class LLMJudge:
    """LLM-судья по методике МТС."""

    def __init__(self, settings: JudgeSettings, llm: LLMClient):
        self._settings = settings
        self._llm = llm

    def _compare_once(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        golden: Optional[str] = None,
    ) -> JudgeVerdict:
        golden_block = (
            f"\nЭталонный (golden) ответ для справки:\n{golden}\n"
            if golden else ""
        )
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b,
            golden_block=golden_block,
        )

        try:
            resp = self._llm.chat(ChatRequest(
                messages=[
                    LLMMessage(role=Role.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
                    LLMMessage(role=Role.USER, content=user_msg),
                ],
                temperature=0.0,
                max_output_tokens=10,
            ))
            raw = (resp.text or "").strip()
            score = self._parse_score(raw)
            return JudgeVerdict(score=score if score is not None else 0.5,
                                raw_text=raw, valid=score is not None)
        except LLMError as e:
            logger.error("Judge LLM error: %s", e)
            return JudgeVerdict(score=0.5, raw_text=str(e), valid=False)

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        # Ищем любое из значений шкалы. Порядок важен: 1.00 раньше 1.0
        match = re.search(r"\b(1\.00|0\.75|0\.50|0\.25|0\.00|1\.0|0\.5|0\.0)\b", text)
        if not match:
            return None
        try:
            v = float(match.group(1))
        except ValueError:
            return None
        if v not in SCORE_VALUES:
            return None
        return v

    def evaluate_pair(
        self,
        question: str,
        ours: str,
        baseline: str,
        golden: Optional[str] = None,
    ) -> tuple[float, float, float]:
        """Возвращает (r1, r2, average).

        Round 1: A=baseline, B=ours → r1 (насколько ours лучше baseline)
        Round 2: A=ours, B=baseline → r2_raw (насколько baseline лучше ours)
                 → r2 = 1 - r2_raw (инвертируем: насколько ours лучше baseline)
        """
        v1 = self._compare_once(question, baseline, ours, golden)
        r1 = v1.score

        v2 = self._compare_once(question, ours, baseline, golden)
        r2 = 1.0 - v2.score

        return r1, r2, (r1 + r2) / 2


def build_judge_client(settings: JudgeSettings) -> LLMClient:
    """Создаёт LLMClient для GigaChat или любого OpenAI-совместимого API."""
    if "giga.chat" in settings.base_url:
        provider = GigaChatTokenProvider(
            auth_key=settings.auth_key,
            scope=settings.scope,
            verify_ssl=False,
        )
        return LLMClient(
            base_url=settings.base_url,
            token_provider=provider,
            default_model=settings.model,
            verify_ssl=False,
        )
    return LLMClient(
        base_url=settings.base_url,
        api_key=settings.auth_key,
        default_model=settings.model,
        verify_ssl=True,
    )