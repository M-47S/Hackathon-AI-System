"""Основная логика диалога: retrieval → LLM → пост-обработка."""
from __future__ import annotations

import logging

from LLM import ChatRequest, LLMError, LLMMessage, Role
from RAG.pipelines import QUERY_INSTRUCTIONS, pipeline_search_with_context_by_text, BM25Index
from RAG.structs import ContextMode

from .products import PRODUCT_CATALOG
from .prompts import SYSTEM_PROMPT
from .schemas import (
    Action, AIResponseForm, CustomerProfileForm, HistoryMessage,
    LLMCustomerProfile, LLMStructuredResponse, RequestToAI,
)

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(
        self,
        *,
        llm_client,
        embedder,
        doc_store,
        vec_store,
        reranker=None,
        bm25_index: BM25Index | None = None,
        use_hybrid: bool = False,
        top_k: int = 6,
        exclude_docs: set[str] | None = None,
    ):
        self._llm = llm_client
        self._embedder = embedder
        self._doc_store = doc_store
        self._vec_store = vec_store
        self._reranker = reranker
        self._bm25_index = bm25_index
        self._use_hybrid = use_hybrid
        self._top_k = top_k
        self._exclude_docs = exclude_docs or {"20-portfolio", "26-general", "00-product-map"}

    def handle(self, req: RequestToAI) -> AIResponseForm:
        log_prefix = f"[session={req.session_id} req={req.request_id}]"
        logger.info("%s handle: message=%r, history=%d", log_prefix, req.message[:120], len(req.history))

        contexts = []
        try:
            contexts = self._retrieve(req)
        except Exception as e:
            logger.exception("%s retrieval failed: %s", log_prefix, e)

        try:
            parsed = self._call_llm(req, self._build_context_block(contexts))
        except LLMError as e:
            logger.exception("%s LLM failed: %s", log_prefix, e)
            return self._fallback_response(req)
        except Exception as e:
            logger.exception("%s unexpected error: %s", log_prefix, e)
            return self._fallback_response(req)

        return self._postprocess(req, parsed, contexts, log_prefix)

    def _retrieve(self, req: RequestToAI):
        query_parts = [req.message]
        for msg in reversed(req.history):
            if msg.role == "user":
                query_parts.append(msg.content)
                break
        query = " ".join(query_parts)

        contexts = pipeline_search_with_context_by_text(
            embedder=self._embedder,
            doc_store=self._doc_store,
            vec_store=self._vec_store,
            query_text=query,
            query_instruction=QUERY_INSTRUCTIONS["detailed_ru"],
            rerank_query_text=req.message,
            mode=ContextMode.PARENT,
            top_k=self._top_k,
            reranker=self._reranker,
            rerank_oversample=3,
            filter_toc=True,
            exclude_documents=self._exclude_docs,
            bm25_index=self._bm25_index,
            use_hybrid=self._use_hybrid,
        )
        logger.debug("Retrieved %d contexts: %s", len(contexts), [c.chunk.document_id for c in contexts])
        return contexts

    @staticmethod
    def _build_context_block(contexts) -> str:
        if not contexts:
            return "Контекст базы знаний пуст. Верни action=\"out_of_scope\"."
        parts = ["Контекст из базы знаний EnergyTool:"]
        total = 0
        for i, ctx in enumerate(contexts, 1):
            text = (ctx.context_text or ctx.chunk.text or "")[:4000]
            block = f"\n[{i}] {ctx.chunk.document_id} / {ctx.chunk.section_title or '—'}\n{text}"
            if total + len(block) > 24_000: break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)

    def _call_llm(self, req: RequestToAI, context_block: str) -> LLMStructuredResponse:
        dialog_text = _format_dialog(req.history, req.message)
        profile_dict = req.customer_profile.model_dump(exclude_none=True)
        profile_block = f"Текущий профиль клиента (JSON):\n{profile_dict}\n" if profile_dict else "Профиль пока пуст.\n"
        summary_block = f"\nРезюме предыдущей консультации:\n{req.conversation_summary}\n" if req.conversation_summary else ""

        user_msg = (
            f"{context_block}\n\n{profile_block}{summary_block}\n"
            f"Диалог:\n{dialog_text}\n\n"
            "Сформируй ответ по правилам. Верни JSON с полями: action, answer, recommended_product_ids, customer_profile, conversation_summary, sequence."
        )

        resp = self._llm.chat(ChatRequest(
            messages=[
                LLMMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                LLMMessage(role=Role.USER, content=user_msg),
            ],
            temperature=0.1,
            max_output_tokens=800,
            response_format=LLMStructuredResponse,
        ))

        if resp.parsed is None:
            logger.error("LLM returned no parsed response: %r", resp.text[:400])
            return LLMStructuredResponse(
                action=Action.OUT_OF_SCOPE,
                answer="Не удалось обработать запрос.",
                customer_profile=LLMCustomerProfile(),
                sequence="Ошибка парсинга JSON от LLM"
            )
        return resp.parsed

    def _postprocess(self, req: RequestToAI, parsed: LLMStructuredResponse, contexts, log_prefix: str) -> AIResponseForm:
        answer = parsed.answer.strip() or "Уточните, пожалуйста, ваш вопрос."
        action = parsed.action

        valid_pids = [pid for pid in parsed.recommended_product_ids if pid in PRODUCT_CATALOG]

        if action == Action.RECOMMEND and not valid_pids:
            logger.warning("%s recommend without valid product_ids %r → ask_question", log_prefix, parsed.recommended_product_ids)
            action = Action.ASK_QUESTION
            if not answer.rstrip().endswith(("?", "？")):
                answer = answer.rstrip() + " Уточните, пожалуйста, детали задачи."

        if action != Action.RECOMMEND:
            valid_pids = []

        merged_profile = _merge_profile(req.customer_profile, parsed.customer_profile)
        summary = parsed.conversation_summary or req.conversation_summary
        if summary: summary = summary.strip() or None

        # ── NEW: собираем документы и sequence ──
        used_docs = list(dict.fromkeys(c.chunk.document_id for c in contexts))
        
        # Если LLM не вернула sequence, генерируем техническую заглушку
        seq = parsed.sequence.strip() or f"RAG retrieval: {len(contexts)} chunks. Action: {action.value}."

        return AIResponseForm(
            session_id=req.session_id,
            action=action,
            answer=answer,
            recommended_product_ids=valid_pids,
            customer_profile=merged_profile,
            conversation_summary=summary,
            sequence=seq,
            used_documents=used_docs,
        )

    @staticmethod
    def _fallback_response(req: RequestToAI) -> AIResponseForm:
        return AIResponseForm(
            session_id=req.session_id,
            action=Action.OUT_OF_SCOPE,
            answer="Извините, не удалось обработать запрос. Пожалуйста, повторите попытку.",
            recommended_product_ids=[],
            customer_profile=req.customer_profile,
            conversation_summary=req.conversation_summary,
            sequence="Fallback: LLM error",
            used_documents=[],
        )

def _format_dialog(history: list[HistoryMessage], current: str) -> str:
    lines = []
    for msg in history:
        prefix = "Клиент" if msg.role == "user" else "Консультант"
        lines.append(f"{prefix}: {msg.content}")
    lines.append(f"Клиент: {current}")
    return "\n".join(lines)

def _merge_profile(old: CustomerProfileForm, new: LLMCustomerProfile) -> CustomerProfileForm:
    return CustomerProfileForm(
        business_type=new.business_type or old.business_type,
        object_count=(new.object_count if new.object_count is not None else old.object_count),
        problem=new.problem or old.problem,
        metering_method=new.metering_method or old.metering_method,
    )