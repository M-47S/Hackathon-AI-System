# ai-service/local_solution.py
"""
Локальный прогон solution на demo_questions.csv.

Использует прод-ресурсы (EmbeddingWrapper, LocalReranker, LLMClient),
читает RAG-индекс с диска, пишет submission.csv рядом с собой.

Запуск:
    python local_solution.py                    # только генерация
    python local_solution.py --evaluate         # генерация + оценка
    python local_solution.py --workers 1        # меньше параллелизма (меньше 429)
    python local_solution.py --skip-rerank      # без реранкера (быстрее)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (жёстко заданы — для локального прогона)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

RAG_KB_DIR    = Path(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Store\_test_\Data Knowledge")
RAG_INDEX_DIR = Path(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Store\Storage")
QUESTIONS_PATH = Path(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Store\_test_\demo_questions.csv")
BASELINE_PATH  = Path(r"C:\Users\0\OneDrive\Desktop\HackAI\ai-service\RAG\Store\_test_\demo_output.csv")

# Output — рядом с local_solution.py
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"

# ---------------------------------------------------------------------------
# Env / Settings
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    for _p in (PROJECT_ROOT / "evaluation" / ".env", PROJECT_ROOT / "LLM" / ".env", PROJECT_ROOT / ".env"):
        if _p.exists():
            load_dotenv(_p, override=False)
            break
except ImportError:
    pass


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


LLM_BASE_URL = _env("AI_BASE_URL", "GIGACHAT_BASE_URL", "LLM_BASE_URL",
                    default="https://api.giga.chat/v1")
LLM_AUTH_KEY = _env("AI_AUTH_KEY", "GIGACHAT_AUTH_KEY", "LLM_API_KEY")
LLM_SCOPE    = _env("AI_SCOPE", "GIGACHAT_SCOPE", default="GIGACHAT_API_PERS")
LLM_MODEL    = _env("AI_MODEL", "GIGACHAT_MODEL", "LLM_MODEL", default="GigaChat-2-MAX")

TOP_K         = int(_env("TOP_K", default="4") or "4")
MAX_WORKERS   = int(_env("MAX_WORKERS", default="2") or "2")
TEMPERATURE   = float(_env("TEMPERATURE", default="0.1") or "0.1")
MAX_TOKENS = int(_env("MAX_TOKENS", default="300") or "300")  # было 600
RERANK_ENABLED = (_env("RAG_RERANK", default="true") or "true").lower() in ("1", "true", "yes")

# Retry-политика для LLM: 429 не должен убивать вопрос
LLM_MAX_ATTEMPTS = 4              # 1 попытка + 3 retry
LLM_BACKOFF_BASE = 3.0            # 3s, 6s, 12s
LLM_CLIENT_MAX_RETRIES = 5        # внутренние retry openai SDK

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("local_solution")

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Ты — ИИ-консультант МТС EnergyTool.

# Формат — ОБЯЗАТЕЛЬНО (как в примерах ниже)
1. Прямой ответ на последнюю реплику (Да / Нет / Зависит / продукт).
2. Конкретный факт из контекста (1-2 предложения, БЕЗ общих слов).
3. Уточняющий вопрос про МАСШТАБ или ТЕКУЩЕЕ СОСТОЯНИЕ клиента.

# ЖЁСТКИЕ ЗАПРЕТЫ
- НЕ выдумывай функции, интерфейсы, каналы (Telegram, WhatsApp, мобильное
  приложение), цифры точности, проценты экономии — если их НЕТ в контексте.
- НЕ пиши общие фразы: «это позволит эффективно», «это удобно для
  централизованного мониторинга», «важно учитывать» — БЕЗ конкретики.
- НЕ пиши длиннее 80 слов. Идеал — 40–60 слов, как в примерах.
- НЕ задавай вопросы про предпочтения клиента (хочет/планирует). Спрашивай
  про ФАКТЫ: количество объектов, текущую систему, мощность, регион.

# ПРИМЕРЫ ХОРОШИХ ОТВЕТОВ (учись стилю)

Вопрос: «У нас несколько магазинов. Можно смотреть их расход в одном кабинете?»
Ответ: «Да, платформа подходит сетевому ретейлу: данные по разным точкам
можно анализировать в одном интерфейсе, сравнивать магазины между собой
и замечать аномальный расход. Сколько у вас магазинов и как сейчас
передаются показания?»

Вопрос: «Можно сначала поставить станцию на 60 кВт, а потом расширить?»
Ответ: «Да, станцию можно вводить поэтапно: начать с 60 кВт, а затем
нарастить мощность. Возможность и параметры расширения лучше учесть при
проектировании. У вас есть данные о дневном потреблении объекта?»

Вопрос: «У нас старая шиферная крыша. Стоит ли ставить панели?»
Ответ: «Старую шиферную кровлю не рекомендуем для солнечной станции без
дополнительной оценки. Срок службы панелей — около 25 лет, а кровлю может
понадобиться менять раньше, что потребует демонтажа оборудования.
Сколько лет вашей кровле?»
"""

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

_DIALOG_LABEL_RE = re.compile(
    r"(?im)^\s*(?:пользователь|ассистент|user|assistant|клиент|бот|consultant)\s*[:：]\s*"
)


def strip_dialogue_labels(text: str) -> str:
    return " ".join(_DIALOG_LABEL_RE.sub("", text).split())


def extract_last_user_message(question: str) -> str:
    """Достаёт последнюю реплику клиента из диалога для retrieval."""
    q_low = question.lower()
    if "пользователь:" not in q_low and "user:" not in q_low:
        return question.strip()
    parts = re.split(r"(?im)^\s*(?:пользователь|user|клиент)\s*[:：]", question)
    return " ".join(parts[-1].split()) if len(parts) >= 2 else question.strip()


def smart_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in (". ", ".\n", "! ", "? ", "\n\n"):
        pos = cut.rfind(sep)
        if pos > max_chars * 0.6:
            return cut[:pos + len(sep)].rstrip() + " […]"
    return cut.rstrip() + " […]"


def build_context_block(contexts) -> str:
    if not contexts:
        return "Контекст из базы знаний: пусто."
    parts = ["Контекст из базы знаний:"]
    total = 0
    for i, ctx in enumerate(contexts, 1):
        text = smart_truncate(ctx.context_text or ctx.chunk.text or "", 3000)
        block = f"\n[{i}] {ctx.chunk.document_id} / {ctx.chunk.section_title or '—'}\n{text}"
        if total + len(block) > 12000:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def fallback_answer() -> str:
    return (
        "В моей базе знаний нет точной информации по этому вопросу. "
        "Рекомендую уточнить у специалиста МТС EnergyTool. "
        "Подскажите, какая задача для вас сейчас в приоритете?"
    )


def postprocess(answer: str) -> str:
    import re
    # 1. Убрать markdown
    answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    answer = re.sub(r"^#{1,6}\s+", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"```[\s\S]*?```", "", answer)
    # 2. Убрать запрещённые «выдумки»
    forbidden = [
        "telegram", "телеграм", "whatsapp", "мобильное приложение",
        "точность более 95", "промышленное компьютерное зрение",
        "гарантированная экономия",
    ]
    for word in forbidden:
        if word in answer.lower():
            # Удалить предложение с запрещённым словом
            sentences = re.split(r"(?<=[.!?])\s+", answer)
            sentences = [s for s in sentences if word not in s.lower()]
            answer = " ".join(sentences)
    # 3. Обрезать до 80 слов
    words = answer.split()
    if len(words) > 80:
        cut = " ".join(words[:78]).rstrip(",.;:") + "…"
        answer = cut
    return " ".join(answer.split())

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            qid = str(r.get("id", "")).strip()
            q = str(r.get("question", "")).strip()
            if qid and q:
                rows.append({"id": qid, "question": q})
    logger.info("Загружено вопросов: %d", len(rows))
    return rows


def write_submission(path: Path, answers: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer"])
        w.writeheader()
        for row in answers:
            w.writerow(row)
    logger.info("Записан %s (%d строк)", path, len(answers))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_llm_client():
    from LLM import GigaChatTokenProvider, LLMClient
    if not LLM_AUTH_KEY:
        raise RuntimeError("Нет ключа: AI_AUTH_KEY / GIGACHAT_AUTH_KEY")

    if "giga.chat" in LLM_BASE_URL:
        provider = GigaChatTokenProvider(
            auth_key=LLM_AUTH_KEY, scope=LLM_SCOPE, verify_ssl=False,
        )
        return LLMClient(
            base_url=LLM_BASE_URL, token_provider=provider,
            default_model=LLM_MODEL, verify_ssl=False,
            max_retries=LLM_CLIENT_MAX_RETRIES,
        )
    return LLMClient(
        base_url=LLM_BASE_URL, api_key=LLM_AUTH_KEY,
        default_model=LLM_MODEL, verify_ssl=True,
        max_retries=LLM_CLIENT_MAX_RETRIES,
    )


def load_or_build_rag(embedder):
    from RAG.Chunking import MarkdownChunker
    from RAG.pipelines import pipeline_build_from_directory, pipeline_load

    try:
        doc_store, vec_store = pipeline_load(RAG_INDEX_DIR, RAG_KB_DIR)
        logger.info("RAG index loaded: %d chunks, %d leaves",
                    len(doc_store), len(doc_store.leaf_chunks))
        return doc_store, vec_store
    except FileNotFoundError:
        logger.warning("RAG index not found at %s — building...", RAG_INDEX_DIR)

    if not RAG_KB_DIR.exists():
        raise RuntimeError(f"KB not found: {RAG_KB_DIR}")

    t0 = time.time()
    stats = pipeline_build_from_directory(
        raw_dir=RAG_KB_DIR, index_dir=RAG_INDEX_DIR, embedder=embedder,
        chunker=MarkdownChunker(chunk_size=640, chunk_overlap=64),
        pattern="**/*.md", save=True,
    )
    logger.info("Index built in %.1fs: %s", time.time() - t0, stats.summary())
    return pipeline_load(RAG_INDEX_DIR, RAG_KB_DIR)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help="После генерации запустить evaluation/main.py")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("LOCAL SOLUTION — EnergyTool")
    logger.info("LLM       : %s @ %s", LLM_MODEL, LLM_BASE_URL)
    logger.info("KB        : %s", RAG_KB_DIR)
    logger.info("INDEX     : %s", RAG_INDEX_DIR)
    logger.info("QUESTIONS : %s", QUESTIONS_PATH)
    logger.info("OUTPUT    : %s", SUBMISSION_PATH)
    logger.info("WORKERS   : %d, RERANK: %s", args.workers, not args.skip_rerank)
    logger.info("=" * 60)

    if not QUESTIONS_PATH.exists():
        logger.error("Вопросы не найдены: %s", QUESTIONS_PATH)
        return 2
    if not RAG_KB_DIR.exists():
        logger.error("KB не найдена: %s", RAG_KB_DIR)
        return 2

    # 1. Questions
    questions = read_questions(QUESTIONS_PATH)

    # 2. Embedder
    logger.info("Загрузка EmbeddingWrapper...")
    t0 = time.time()
    from RAG.Embedding import EmbeddingWrapper
    embedder = EmbeddingWrapper()
    logger.info("Embedder готов за %.1fs", time.time() - t0)

    # 3. Reranker
    reranker = None
    if RERANK_ENABLED and not args.skip_rerank:
        try:
            from RAG.Ranking import LocalReranker
            t0 = time.time()
            reranker = LocalReranker(device="cpu")
            logger.info("Reranker готов за %.1fs", time.time() - t0)
        except Exception as e:
            logger.warning("Reranker не загружен: %s", e)

    # 4. RAG index
    t0 = time.time()
    doc_store, vec_store = load_or_build_rag(embedder)
    logger.info("RAG ready за %.1fs", time.time() - t0)

    # 5. LLM
    llm = build_llm_client()

    # 6. Processing
    from LLM import ChatRequest, LLMMessage, Role
    from LLM.exceptions import LLMRateLimitError
    from RAG.pipelines import QUERY_INSTRUCTIONS, pipeline_search_with_context_by_text
    from RAG.structs import ContextMode

    instruction = QUERY_INSTRUCTIONS["detailed_ru"]

    def process_one(idx: int, row: dict) -> tuple[int, str]:
        qid = row["id"]
        raw_q = row["question"]
        retrieval_q = extract_last_user_message(raw_q)

        last_err: Exception | None = None
        for attempt in range(LLM_MAX_ATTEMPTS):
            try:
                # --- Retrieval ---
                contexts = pipeline_search_with_context_by_text(
                    embedder=embedder,
                    doc_store=doc_store,
                    vec_store=vec_store,
                    query_text=retrieval_q,
                    query_instruction=instruction,
                    rerank_query_text=retrieval_q,
                    mode=ContextMode.PARENT,
                    top_k=TOP_K,
                    reranker=reranker,
                    rerank_oversample=3,
                    filter_toc=True,
                )

                # --- Prompt ---
                user_msg = (
                    f"{build_context_block(contexts)}\n\n"
                    f"Диалог с клиентом:\n{raw_q}\n\n"
                    "Ответь по правилам. Не повторяй вопрос клиента. "
                    "Обязательно закончи одним уточняющим вопросом."
                )

                # --- LLM ---
                resp = llm.chat(ChatRequest(
                    messages=[
                        LLMMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                        LLMMessage(role=Role.USER, content=user_msg),
                    ],
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                ))

                # --- Post-processing ---
                raw_answer = resp.text or ""
                answer = postprocess(raw_answer)
                if len(answer) > 10000:
                    answer = answer[:9950] + "…"

                logger.info(
                    "[%s] ok ctx=%d raw=%d clean=%d (attempt %d)",
                    qid, len(contexts), len(raw_answer), len(answer), attempt + 1,
                )
                return idx, answer

            except LLMRateLimitError as e:
                last_err = e
                if attempt < LLM_MAX_ATTEMPTS - 1:
                    wait = LLM_BACKOFF_BASE * (2 ** attempt)   # 3s, 6s, 12s
                    logger.warning(
                        "[%s] 429 Too Many Requests — retry %d/%d через %.1fs",
                        qid, attempt + 1, LLM_MAX_ATTEMPTS - 1, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error("[%s] 429 после %d попыток — fallback",
                             qid, LLM_MAX_ATTEMPTS)
                return idx, fallback_answer()

            except Exception as e:
                last_err = e
                logger.exception("[%s] ERROR: %s", qid, e)
                return idx, fallback_answer()

        # Теоретически недостижимо, но на всякий случай
        logger.error("[%s] exhausted retries: %s", qid, last_err)
        return idx, fallback_answer()

    results: list[str | None] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, i, r): i for i, r in enumerate(questions)}
        for fut in as_completed(futures):
            i, ans = fut.result()
            results[i] = ans

    answers = [
        {"id": q["id"], "answer": results[i] or fallback_answer()}
        for i, q in enumerate(questions)
    ]
    write_submission(SUBMISSION_PATH, answers)

    # Cleanup
    try:
        llm.close()
    except Exception:
        pass
    if reranker:
        try:
            reranker.close()
        except Exception:
            pass

    logger.info("DONE за %.1fs", time.time() - t_start)

    # 7. Evaluate (если флаг)
    if args.evaluate:
        logger.info("=" * 60)
        logger.info("Запуск evaluation...")
        logger.info("=" * 60)
        os.environ["EVAL_QUESTIONS"]  = str(QUESTIONS_PATH)
        os.environ["EVAL_BASELINE"]   = str(BASELINE_PATH)
        os.environ["EVAL_SUBMISSION"] = str(SUBMISSION_PATH)

        # Очищаем sys.argv, чтобы evaluation.main.main()
        # не пытался распарсить наши флаги (--evaluate, --workers, ...)
        _saved_argv = sys.argv
        sys.argv = [_saved_argv[0]]
        try:
            from evaluation.main import main as eval_main
            return eval_main()
        finally:
            sys.argv = _saved_argv

    logger.info("-" * 60)
    logger.info("Чтобы оценить: python local_solution.py --evaluate")
    logger.info("Или вручную:   python -m evaluation.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())