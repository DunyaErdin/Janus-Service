from __future__ import annotations

from app.domain.models.ai_response_plan import AIResponsePlan


class LlmStructuredResponse(AIResponsePlan):
    """Canonical schema the LLM must satisfy before any response is dispatched."""

