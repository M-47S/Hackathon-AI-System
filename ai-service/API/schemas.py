"""
Pydantic-схемы AI Service.

⚠️  Названия полей синхронизированы с Go-структурами напарника, а не с
   api-contract.md (есть расхождения: object_count vs objects_count,
   recommended_product_ids vs product_ids). Источник истины — Go-код.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(str, Enum):
    ANSWER_QUESTION = "answer_question"
    ASK_QUESTION = "ask_question"
    RECOMMEND = "recommend"
    OFFER_HANDOFF = "offer_handoff"
    OUT_OF_SCOPE = "out_of_scope"


# ---------------------------------------------------------------------------
# CustomerProfile
# ---------------------------------------------------------------------------

class CustomerProfileForm(BaseModel):
    """
    Соответствует Go `CustomerProfileForm`:
        BusinessType  string `json:"business_type"`
        ObjectCount   int    `json:"object_count"`
        Problem       string `json:"problem"`
        MeteringMethod string `json:"metering_method"`

    Go использует non-pointer int/string → `null` приезжает как 0/"".
    Внутри сервиса трактуем 0 и "" как "неизвестно" (None).
    """
    business_type: str | None = None
    object_count: int | None = None       # ← object_count (Go tag)
    problem: str | None = None
    metering_method: str | None = None

    @field_validator("business_type", "problem", "metering_method", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v):
        if v is None or v == "":
            return None
        return v

    @field_validator("object_count", mode="before")
    @classmethod
    def _zero_to_none(cls, v):
        if v is None or v == 0:
            return None
        return v


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Inbound: Backend → AI
# ---------------------------------------------------------------------------

class RequestToAI(BaseModel):
    """Соответствует Go `RequestToAI`."""
    session_id: str
    request_id: str
    message: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list)
    customer_profile: CustomerProfileForm = Field(default_factory=CustomerProfileForm)
    conversation_summary: str | None = None

    @field_validator("conversation_summary", mode="before")
    @classmethod
    def _empty_summary_to_none(cls, v):
        if v is None or v == "":
            return None
        return v


# ---------------------------------------------------------------------------
# Outbound: AI → Backend
# ---------------------------------------------------------------------------

class AIResponseForm(BaseModel):
    """Соответствует Go `AIResponseForm`."""
    session_id: str
    action: Action
    answer: str = Field(min_length=1)
    recommended_product_ids: list[str] = Field(default_factory=list)   # ← Go tag
    customer_profile: CustomerProfileForm
    conversation_summary: str | None = None


# ---------------------------------------------------------------------------
# LLM structured output (внутренняя модель — не выходит за пределы сервиса)
# ---------------------------------------------------------------------------

class LLMCustomerProfile(BaseModel):
    """То, что LLM заполняет в structured output (без валидаторов)."""
    business_type: str | None = None
    object_count: int | None = None
    problem: str | None = None
    metering_method: str | None = None


class LLMStructuredResponse(BaseModel):
    """Схема, которую LLMClient передаёт в response_format.

    Сгенерированный JSON попадёт в AIChatResponse после пост-обработки.
    """
    action: Action
    answer: str
    recommended_product_ids: list[str] = Field(default_factory=list)
    customer_profile: LLMCustomerProfile = Field(default_factory=LLMCustomerProfile)
    conversation_summary: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"