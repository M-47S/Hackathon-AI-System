# GRADER/solution.py
from __future__ import annotations

import csv
import logging
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ---------------- Env ----------------
def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

LLM_BASE_URL = _env("LLM_BASE_URL", "OPENAI_BASE_URL")
LLM_API_KEY = _env("LLM_API_KEY", "OPENAI_API_KEY")
LLM_MODEL = _env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", default="qwen/qwen3.8-27b")
EMB_MODEL = _env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = _env("RERANKER_MODEL", default="qwen/qwen3-reranker-8b")

# ── САБМИТ #1: больше контекста и места для ответа (vs Round 1 = 0.5798) ──
TOP_K = int(_env("TOP_K", default="5") or "5")            # 4 → 5
MAX_TOKENS = int(_env("MAX_TOKENS", default="800") or "800")  # 600 → 800

TEMPERATURE = float(_env("TEMPERATURE", default="0.1") or "0.1")

try:
    _w = int(_env("MAX_WORKERS", default="5") or "5")
except ValueError:
    _w = 5
MAX_WORKERS = min(8, max(4, _w))

DATA_DIR = Path(_env("DATA_DIR", default="data") or "data")

# ── САБМИТ #1: больше контекста на чанк и всего ──
MAX_CTX_CHARS_PER_CHUNK = 4000    # 3500 → 4000
MAX_CTX_TOTAL_CHARS = 20000       # 14000 → 20000

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("solution")

# ---------------- Imports ----------------
from llm_client import LLMClient
from structs import ContextMode, ChatRequest, LLMMessage, Role, LLMError
from document_store import DocumentStore
from vector_store import VectorStore
from chunker import MarkdownChunker
from rag_embed import RemoteEmbedder
from rag_rank import QwenReranker
from pipelines import (
    pipeline_fill_store,
    pipeline_search_with_context_by_text,
    QUERY_INSTRUCTIONS,
    DEFAULT_QUERY_INSTRUCTION,
)

# ---------------- System prompt (Round 1 = 0.5798, не трогаем) ----------------
SYSTEM_PROMPT = """\
Ты — профессиональный ИИ-консультант МТС EnergyTool. Ты помогаешь бизнесу \
с учётом, мониторингом и оптимизацией электроэнергии.

# Как отвечать

1. **Прямой ответ в начале.** Первая фраза — это прямой ответ на последний \
вопрос клиента (Да / Нет / Зависит / Конкретный продукт).
2. **Краткое обоснование.** 1–2 факта из контекста базы знаний. \
Не пересказывай контекст целиком, выдели главное.
3. **Один уточняющий вопрос в конце.** Всегда. Он должен продвигать \
консультацию: уточнить масштаб, текущую систему учёта, цель клиента или регион.
4. **Учитывай историю диалога.** Если клиент уже назвал важные детали \
(масштаб, регион, тип объекта, цель) — не спрашивай повторно.
5. **Только факты из контекста.** Если в контексте нет ответа — честно скажи: \
"В моей базе знаний нет точной информации об этом. Рекомендую уточнить у \
специалиста" и задай уточняющий вопрос.
6. **Никаких выдумок.** Не придумывай цены, характеристики, сроки, названия \
компаний и партнёров.
7. **Длина:** обычно до 150 слов. Живой экспертный язык, без «воды».

# Пример стиля

Вопрос: «У нас старая шиферная крыша. Стоит ли ставить панели?»
Ответ: «Старую шиферную кровлю не рекомендуем использовать для солнечной \
станции без дополнительной оценки. Срок службы панелей — около 25 лет, а \
кровлю может понадобиться менять раньше, что приведёт к демонтажу \
оборудования. Подскажите, сколько лет вашей кровле и планируете ли ремонт \
в ближайшие годы?»
"""

# ---------------- Utils ----------------
_DIALOGUE_LABEL_RE = re.compile(
    r"(?im)^\s*(?:пользователь|ассистент|user|assistant|клиент|бот|consultant)\s*[:：]\s*"
)

def strip_dialogue_labels(text: str) -> str:
    """Убирает ярлыки 'Пользователь:', 'Ассистент:' — они шумят в эмбеддингах."""
    cleaned = _DIALOGUE_LABEL_RE.sub("", text)
    return " ".join(cleaned.split())


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
        title = ctx.chunk.section_title or "—"
        doc = ctx.chunk.document_id
        text = (ctx.context_text or ctx.chunk.text or "").strip()
        text = smart_truncate(text, MAX_CTX_CHARS_PER_CHUNK)
        block = f"\n[{i}] {doc} / {title}\n{text}"
        if total + len(block) > MAX_CTX_TOTAL_CHARS:
            log.warning("Контекст превысил %d символов — обрезан", MAX_CTX_TOTAL_CHARS)
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


# ---------------- IO ----------------
def prepare_knowledge_base(dst_dir: Path) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "knowledge-base.zip"
    if zip_path.exists():
        log.info("Распаковка %s", zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dst_dir)
    else:
        log.warning("knowledge-base.zip не найден, ищу .md прямо в DATA_DIR")
        for f in DATA_DIR.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".md", ".txt"):
                target = dst_dir / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
    md_files = sorted(dst_dir.rglob("*.md"))
    log.info("Найдено .md файлов: %d", len(md_files))
    return md_files


def read_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
        rows = []
        for r in csv.DictReader(f, dialect=dialect):
            norm = {(k or "").strip().lstrip("\ufeff").lower(): (v or "").strip()
                    for k, v in r.items()}
            qid = norm.get("id", "")
            q = norm.get("question", "")
            if qid and q:
                rows.append({"id": qid, "question": q})
    log.info("Загружено вопросов: %d", len(rows))
    return rows


def write_submission(path: Path, answers: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for row in answers:
            writer.writerow(row)
    log.info("Записан %s, строк: %d", path, len(answers))


# ---------------- Main ----------------
def main() -> None:
    t_start = time.time()

    log.info("=" * 60)
    log.info("LLM=%s  EMB=%s  RERANK=%s", LLM_MODEL, EMB_MODEL, RERANKER_MODEL)
    log.info("TOP_K=%d  WORKERS=%d  TEMP=%.2f  MAX_TOKENS=%d",
             TOP_K, MAX_WORKERS, TEMPERATURE, MAX_TOKENS)
    log.info("CTX per chunk=%d  total=%d",
             MAX_CTX_CHARS_PER_CHUNK, MAX_CTX_TOTAL_CHARS)
    log.info("=" * 60)

    if not LLM_BASE_URL or not LLM_API_KEY:
        log.error("Нет LLM_BASE_URL / LLM_API_KEY")
        sys.exit(1)

    questions_path = DATA_DIR / "questions.csv"
    if not questions_path.exists():
        log.error("Нет %s", questions_path)
        sys.exit(1)
    questions = read_questions(questions_path)

    kb_dir = Path("_kb_extracted")
    md_files = prepare_knowledge_base(kb_dir)

    # --- Клиенты (создаются ОДИН раз) ---
    llm = LLMClient(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, default_model=LLM_MODEL)
    embedder = RemoteEmbedder(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=EMB_MODEL)
    probe_dim = embedder._get_text_embedding("probe").size
    log.info("Размерность эмбеддингов: %d", probe_dim)

    reranker: Optional[QwenReranker] = None
    try:
        reranker = QwenReranker(
            base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=RERANKER_MODEL,
        )
        log.info("Реранкер инициализирован")
    except Exception as e:
        log.warning("Реранкер не создан: %s", e)

    # --- Индекс (строится ОДИН раз) ---
    doc_store = DocumentStore(documents_root=kb_dir)
    vec_store = VectorStore(dim=probe_dim)
    chunker = MarkdownChunker(chunk_size=640, chunk_overlap=64)

    log.info("Индексирование...")
    t0 = time.time()
    stats = pipeline_fill_store(
        doc_store=doc_store, vec_store=vec_store, embedder=embedder,
        paths=md_files, chunker=chunker,
    )
    log.info("Индекс за %.1fs: %s", time.time() - t0, stats.summary())

    instruction = QUERY_INSTRUCTIONS.get("detailed_ru", DEFAULT_QUERY_INSTRUCTION)

    # --- Обработка ---
    def process_one(i: int) -> tuple[int, str]:
        q = questions[i]
        raw_question = q["question"]
        # Чистим ярлыки для retrieval, но в LLM уходит полный текст
        retrieval_query = strip_dialogue_labels(raw_question) or raw_question

        try:
            contexts = pipeline_search_with_context_by_text(
                embedder=embedder,
                doc_store=doc_store,
                vec_store=vec_store,
                query_text=retrieval_query,
                query_instruction=instruction,
                rerank_query_text=retrieval_query,
                mode=ContextMode.PARENT,
                top_k=TOP_K,
                reranker=reranker,
                rerank_oversample=3,
                filter_toc=True,
            )
            ctx_block = build_context_block(contexts)

            user_msg = (
                f"{ctx_block}\n\n"
                f"Диалог с клиентом:\n{raw_question}\n\n"
                f"Ответь по правилам. Не повторяй вопрос клиента. "
                f"Обязательно закончи одним уточняющим вопросом."
            )
            response = llm.chat(ChatRequest(
                messages=[
                    LLMMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                    LLMMessage(role=Role.USER, content=user_msg),
                ],
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ))
            answer = (response.text or "").strip() or fallback_answer()
            answer = " ".join(answer.split())   # одна строка для CSV
            if len(answer) > 10_000:
                answer = answer[:9_950] + "…"
            log.info("[%s] ok ctx=%d chars=%d", q["id"], len(contexts), len(answer))
            return i, answer
        except LLMError as e:
            log.error("[%s] LLM error: %s", q["id"], e)
            return i, fallback_answer()
        except Exception as e:
            log.exception("[%s] ERROR: %s", q["id"], e)
            return i, fallback_answer()

    results: list[Optional[str]] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one, i): i for i in range(len(questions))}
        for fut in as_completed(futures):
            i, ans = fut.result()
            results[i] = ans

    answers = [
        {"id": q["id"], "answer": results[i] or fallback_answer()}
        for i, q in enumerate(questions)
    ]
    write_submission(Path("submission.csv"), answers)

    log.info("DONE за %.1f сек", time.time() - t_start)
    if reranker is not None:
        reranker.close()
    llm.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("FATAL")
        try:
            if not Path("submission.csv").exists():
                qp = DATA_DIR / "questions.csv"
                if qp.exists():
                    qs = read_questions(qp)
                    write_submission(
                        Path("submission.csv"),
                        [{"id": q["id"], "answer": fallback_answer()} for q in qs],
                    )
        except Exception:
            pass
        sys.exit(1)