# ai-service/local_solution.py
"""
Локальный прогон solution на demo_questions.csv.

Использует прод-ресурсы (EmbeddingWrapper, LocalReranker, LLMClient),
читает RAG-индекс с диска, пишет submission.csv рядом с собой.

Это версия Round 6 — лучший локальный Score (0.4688).
Не менять промпт/постобработку без нового eval-прогона.

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
    for _p in (PROJECT_ROOT / "evaluation" / ".env",
               PROJECT_ROOT / "LLM" / ".env",
               PROJECT_ROOT / ".env"):
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

TOP_K         = int(_env("TOP_K", default="3") or "3")
MAX_WORKERS   = int(_env("MAX_WORKERS", default="2") or "2")
TEMPERATURE   = float(_env("TEMPERATURE", default="0.1") or "0.1")
MAX_TOKENS    = int(_env("MAX_TOKENS", default="600") or "600")
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
# System Prompt (Round 6 — лучший вариант)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Ты — ИИ-консультант МТС EnergyTool.

# КРИТИЧЕСКОЕ ПРАВИЛО ПРО ЧИСЛА
- Любое число (5-10 лет, 30 лет, 95%, 750 кВт, 25 лет) ДОЛЖНО быть
  дословно в блоке «Факты из контекста».
- Если числа нет в фактах — НЕ ПИШИ ЕГО. Лучше без числа, чем выдумка.
- НЕ добавляй уточнений, которых нет в контексте:
  • «старше 5-10 лет» — если в контексте только «старая кровля»
  • «стоимость ориентировочно» — если про стоимость ничего нет
  • «коэффициент мощности cos φ» — если этого термина нет в контексте

# Правила ответа
1. Прямой ответ: Да / Нет / Зависит / название продукта.
2. МИНИМУМ 2 КОНКРЕТНЫХ ФАКТА из блока «Факты» — цифры, названия,
   условия, функции продуктов.
3. Уточняющий вопрос про ФАКТЫ клиента (масштаб, модели, регион).

# Open-ended вопросы («С чего начать?», «Что посоветуете?»)
Перечисли КОНКРЕТНЫЕ продукты из контекста и их функции.
Пример: «Начните с перечня счётчиков по арендаторам. Приборы, которые
можно подключить, будут передавать показания в облачную платформу.
Для неподключённых — распознавание по фотографии. Сколько счётчиков
обслуживает техник?»

# Запрещено
- «Это позволит», «Это удобно», «Важно учитывать» — пустые фразы
- Telegram, WhatsApp, мобильное приложение, «промышленное компьютерное
  зрение», проценты точности 95%/90% — если их нет в контексте
- markdown

# Длина
40–120 слов.
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


def extract_dialogue_query(question: str) -> str:
    """Для retrieval: последние 2 реплики пользователя (не только последнюю)."""
    q_low = question.lower()
    if "пользователь:" not in q_low and "user:" not in q_low:
        return question.strip()
    user_msgs = re.findall(
        r"(?im)^\s*(?:пользователь|user|клиент)\s*[:：]\s*(.+)$", question
    )
    if not user_msgs:
        return question.strip()
    return " ".join(user_msgs[-2:]).strip()


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
    """Round 6: markdown cleanup + точные галлюцинации. Без проверки чисел."""
    import re
    if not answer:
        return fallback_answer()
    original = answer

    # Markdown
    answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    answer = re.sub(r"^#{1,6}\s+", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"```[\s\S]*?```", "", answer)

    # Точные галлюцинации (без стемов — стемы режут лишнее)
    exact_bad = (
        "телеграм", "telegram", "whatsapp", "мобильное приложение",
        "профи", "лайт", "опросный лист", "опросного листа",
    )
    if any(w in answer.lower() for w in exact_bad):
        sents = re.split(r"(?<=[.!?])\s+", answer)
        cleaned = [
            s for s in sents
            if not any(w in s.lower() for w in exact_bad)
        ]
        answer = " ".join(cleaned) if cleaned else original

    return " ".join(answer.split()) or fallback_answer()


def extract_context_facts(contexts) -> list[str]:
    """Извлекает числа, проценты, сроки, модели из retrieved-контекста."""
    import re
    text = " ".join((c.context_text or c.chunk.text or "") for c in contexts)
    facts = set()
    for m in re.finditer(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|кВт|МВт|лет|года|год|мес|руб|₽|тыс)", text
    ):
        facts.add(m.group(0).strip())
    for m in re.finditer(r"\b[A-ZА-Я][a-zа-я]+\s+\d+[A-ZА-Я0-9-]*\b", text):
        facts.add(m.group(0).strip())
    for m in re.finditer(r"«([^»]{2,40})»", text):
        facts.add(m.group(1).strip())
    return sorted(facts)[:25]

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
    logger.info("LOCAL SOLUTION — EnergyTool (Round 6)")
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
        retrieval_q = extract_dialogue_query(raw_q)

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

                # --- Facts block (Round 6) ---
                facts = extract_context_facts(contexts)
                facts_block = (
                    "Факты из контекста (используй ТОЛЬКО эти; если числа нет здесь — НЕ ПИШИ ЕГО):\n"
                    + "\n".join(f"- {f}" for f in facts)
                    if facts else
                    "В контексте нет явных чисел/сроков/моделей. Не выдумывай их."
                )

                # --- Prompt ---
                user_msg = (
                    f"{build_context_block(contexts)}\n\n"
                    f"{facts_block}\n\n"
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

                # --- Post-processing (Round 6: без context_text) ---
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