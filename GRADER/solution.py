# GRADER/solution.py
"""
solution.py — точка входа для автоматической проверки хакатона Hack&Change 2026.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
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

LLM_BASE_URL  = _env("LLM_BASE_URL", "OPENAI_BASE_URL")
LLM_API_KEY   = _env("LLM_API_KEY", "OPENAI_API_KEY")
LLM_MODEL     = _env("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL", default="Qwen3.8-27B")
EMB_MODEL     = _env("EMB_MODEL", default="BAAI/bge-m3")
RERANKER_MODEL = _env("RERANKER_MODEL", default="Qwen3-Reranker-8B")
TOP_K          = int(_env("TOP_K", default="5") or "5")
MAX_WORKERS    = int(_env("MAX_WORKERS", default="4") or "4")
TEMPERATURE    = float(_env("TEMPERATURE", default="0.3") or "0.3")
DATA_DIR       = Path(_env("DATA_DIR", default="data"))

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("solution")

# ---------------- Imports из модулей (в корне ZIP) ----------------
from llm_client import LLMClient
from structs import Embedding, ContextMode, ChatRequest, LLMMessage, Role
from document_store import DocumentStore
from vector_store import VectorStore
from chunker import MarkdownChunker
from rag_embed import RemoteEmbedder
from rag_rank import QwenReranker
from pipelines import pipeline_fill_store, pipeline_search_with_context


# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """\
Ты — ИИ-консультант по продуктам и услугам МТС EnergyTool.

# Твоя роль
Ты помогаешь бизнес-клиенту разобраться в решениях по учёту и экономии 
энергоресурсов и подобрать подходящий продукт. Ты ведёшь консультацию 
как эксперт, а не поисковик: помогаешь сформулировать задачу, 
рекомендуешь решение и предлагаешь связаться с менеджером, когда 
клиент готов оставить заявку.

# Продукты, о которых ты знаешь
1. Облачная платформа учёта электроэнергии (АСКУЭ) — облачный учёт 
   по объектам, арендаторам, цехам; работа с 300+ типами приборов.
2. Солнечные электростанции — сетевые (снижение платежа) и гибридные 
   (с накопителями, для автономии). Мощность 15–500 кВт.
3. Выход на оптовый рынок электроэнергии (ОРЕМ) — для предприятий 
   с присоединённой мощностью от 750 кВт.
4. Поиск коммерческих потерь — для электросетевых компаний, ML-сервис 
   приоритизации выездных проверок.
5. Распознавание показаний по фото — для приборов, не подключённых 
   к автоматике; электроэнергия и вода.
6. Расчёт экономического эффекта, аудит учёта — как сопутствующие услуги.

# Правила ответа
1. Длина ответа: как правило до 150 слов. Если вопрос требует 
   подробного объяснения — можно больше, но без «воды».
2. Отвечай прямо на последнюю реплику клиента, учитывая контекст 
   диалога, если он есть.
3. Используй ТОЛЬКО факты из предоставленного контекста. Если 
   информации в контексте нет — прямо скажи об этом и предложи 
   связаться со специалистом.
4. Не придумывай цены, гарантии, характеристики, сроки, названия 
   компаний и партнёров. Если в контексте нет цены — не называй её.
5. Если для подбора продукта не хватает данных — задай ОДИН 
   уточняющий вопрос в конце ответа.
6. Если клиент уже описал задачу и продукт подходит — назови продукт 
   и кратко объясни почему.

# Чего делать нельзя
- Не обещай конкретную экономию, окупаемость или сроки до расчёта.
- Не называй компании-клиенты, поставщиков, подрядчиков.
- Не давай юридических консультаций по схемам, договорам, персональным данным.
- Не уходи в темы вне продуктов EnergyTool и связанных задач клиента.
- Не выдумывай новые продукты, которых нет в списке выше.

# Пример хорошего ответа
Вопрос: «У нас сеть магазинов, как смотреть расход в одном кабинете?»
Ответ: «Для сетевого ретейла подходит Облачная платформа учёта — 
данные по всем точкам видны в одном интерфейсе, можно сравнивать 
магазины и отслеживать аномальный расход. Сколько у вас магазинов 
и как сейчас передаются показания?»

Контекст из базы знаний будет предоставлен в следующем сообщении. 
Опирайся на него как на единственный источник фактов.
"""

EXPANSION_PROMPT = """Ты — эксперт по продуктам МТС EnergyTool (учёт энергоресурсов, АСКУЭ, солнечные электростанции).
Перепиши запрос клиента, заменив аббревиатуры и жаргонизмы на развёрнутые термины из области энергетики.
Пример: "АСКУЭ" -> "автоматизированная система контроля и учёта электроэнергии, облачная платформа учёта".
Верни ТОЛЬКО переписанный запрос без кавычек и пояснений.
Запрос: {query}
Переписанный запрос:"""


# ---------------- Utils ----------------
def extract_last_user_message(question: str) -> str:
    if "пользователь:" not in question.lower() and "user:" not in question.lower():
        return question

    last_user: Optional[str] = None
    current_user: Optional[str] = None

    for line in question.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("пользователь:") or low.startswith("user:"):
            if current_user:
                last_user = current_user
            current_user = s.split(":", 1)[1].strip()
        elif low.startswith("ассистент:") or low.startswith("assistant:"):
            if current_user:
                last_user = current_user
            current_user = None
        elif current_user is not None:
            current_user += " " + s

    if current_user:
        last_user = current_user
    return last_user or question

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
            if f.is_file():
                target = dst_dir / f.name
                if not target.exists():
                    shutil.copy(f, target)
    md_files = sorted(dst_dir.rglob("*.md"))
    log.info("Найдено .md файлов: %d", len(md_files))
    return md_files

def read_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{"id": r["id"], "question": r["question"]} for r in reader]
    log.info("Загружено вопросов: %d", len(rows))
    return rows

def write_submission(path: Path, answers: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for row in answers:
            writer.writerow({"id": row["id"], "answer": row["answer"]})
    log.info("Записан %s, строк: %d", path, len(answers))

def build_context_block(contexts) -> str:
    if not contexts:
        return "Контекст из базы знаний: пусто."
    parts = ["Контекст из базы знаний:"]
    for i, ctx in enumerate(contexts, 1):
        title = ctx.chunk.section_title or "—"
        doc = ctx.chunk.document_id
        text = ctx.context_text.strip()
        parts.append(f"\n[{i}] {doc} / {title}\n{text}")
    return "\n".join(parts)

def fallback_answer() -> str:
    return (
        "Не удалось подготовить ответ по этому вопросу. "
        "Рекомендую связаться со специалистом МТС EnergyTool для консультации."
    )

def expand_query(llm: LLMClient, query: str) -> str:
    try:
        resp = llm.chat(ChatRequest(
            messages=[LLMMessage(role=Role.USER, content=EXPANSION_PROMPT.format(query=query))],
            temperature=0.1,
            max_output_tokens=150,
        ))
        expanded = (resp.text or "").strip()
        return expanded if expanded else query
    except Exception as e:
        log.warning("Query expansion failed for %r: %s", query, e)
        return query

# ---------------- Main pipeline ----------------
def main() -> None:
    t_start = time.time()
    log.info("=" * 60)
    log.info("START. LLM_MODEL=%s, EMB_MODEL=%s, TOP_K=%d, WORKERS=%d",
             LLM_MODEL, EMB_MODEL, TOP_K, MAX_WORKERS)
    log.info("=" * 60)

    if not LLM_BASE_URL or not LLM_API_KEY:
        log.error("Не переданы LLM_BASE_URL / LLM_API_KEY, дальше работать нельзя.")
        sys.exit(1)

    questions_path = DATA_DIR / "questions.csv"
    if not questions_path.exists():
        log.error("Не найден %s", questions_path)
        sys.exit(1)
    questions = read_questions(questions_path)

    kb_dir = Path("_kb_extracted")
    md_files = prepare_knowledge_base(kb_dir)

    # --- Клиенты ---
    llm = LLMClient(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, default_model=LLM_MODEL)

    embedder = RemoteEmbedder(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=EMB_MODEL)
    probe_dim = embedder._get_text_embedding("probe").size
    log.info("Размерность эмбеддингов: %d", probe_dim)

    reranker = None
    try:
        reranker = QwenReranker(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=RERANKER_MODEL)
        log.info("QwenReranker инициализирован (модель %s)", RERANKER_MODEL)
    except Exception as e:
        log.warning("Reranker не инициализирован (%s), работаем без него", e)

    # --- Индекс ---
    doc_store = DocumentStore(documents_root=kb_dir)
    vec_store = VectorStore(dim=probe_dim)
    chunker = MarkdownChunker(chunk_size=384, chunk_overlap=30)

    log.info("Индексирование базы знаний...")
    t0 = time.time()
    stats = pipeline_fill_store(
        doc_store=doc_store, vec_store=vec_store, embedder=embedder,
        paths=md_files, chunker=chunker,
    )
    log.info("Индекс построен за %.1fs: %s", time.time() - t0, stats.summary())

    # --- Query Expansion ---
    log.info("Расширение запросов (Query Expansion)...")
    queries_text = [extract_last_user_message(q["question"]) for q in questions]
    
    expanded_texts: list[str] = [""] * len(queries_text)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(expand_query, llm, q): i for i, q in enumerate(queries_text)}
        for fut in as_completed(futures):
            i = futures[fut]
            expanded_texts[i] = fut.result()
    
    # --- Эмбеддинги вопросов ---
    log.info("Эмбеддинг расширенных запросов...")
    q_embeddings = embedder._get_text_embedding_batch(expanded_texts)

    # --- Генерация ответов ---
    log.info("Генерация ответов (workers=%d)...", MAX_WORKERS)

    def process_one(i: int) -> tuple[int, str]:
        q = questions[i]
        q_emb = q_embeddings[i]
        query_text = queries_text[i]  # оригинальный запрос (для реранкера)
        
        try:
            # Retrieval + reranking
            contexts = pipeline_search_with_context(
                doc_store=doc_store, vec_store=vec_store, query=q_emb,
                mode=ContextMode.PARENT, top_k=TOP_K, min_score=None,
                reranker=reranker, query_text=query_text, rerank_oversample=4,
            )

            # Промпт
            context_block = build_context_block(contexts)
            user_msg = (
                f"{context_block}\n\n"
                f"Вопрос клиента:\n{q['question']}\n\n"
                f"Дай ответ строго по контексту. Если в контексте нет нужной "
                f"информации — скажи об этом и предложи связаться со специалистом. "
                f"Если нужны уточнения — задай один короткий вопрос в конце."
            )

            # LLM
            response = llm.chat(ChatRequest(
                messages=[
                    LLMMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                    LLMMessage(role=Role.USER, content=user_msg),
                ],
                temperature=TEMPERATURE, max_output_tokens=600,
            ))
            answer = (response.text or "").strip() or fallback_answer()
            if len(answer) > 10_000:
                answer = answer[:9_950] + "…"

            log.info("[%s] ok (%d chars, ctx=%d)", q["id"], len(answer), len(contexts))
            return i, answer
        except Exception as e:
            log.exception("[%s] ERROR: %s", q["id"], e)
            return i, fallback_answer()

    results: list[Optional[str]] = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one, i): i for i in range(len(questions))}
        for fut in as_completed(futures):
            i, answer = fut.result()
            results[i] = answer

    # Запись
    answers = [{"id": q["id"], "answer": results[i] or fallback_answer()} for i, q in enumerate(questions)]
    write_submission(Path("submission.csv"), answers)
    log.info("=" * 60)
    log.info("DONE за %.1f секунд", time.time() - t_start)
    log.info("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("FATAL: необработанная ошибка")
        try:
            if not Path("submission.csv").exists():
                questions_path = DATA_DIR / "questions.csv"
                if questions_path.exists():
                    qs = read_questions(questions_path)
                    write_submission(Path("submission.csv"), [{"id": q["id"], "answer": fallback_answer()} for q in qs])
        except Exception:
            pass
        sys.exit(1)