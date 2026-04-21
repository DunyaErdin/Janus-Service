from __future__ import annotations

from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.ports.llm_port import LlmPort, LlmPromptInput
from app.domain.ports.provider_errors import ProviderUnavailableError


class GeminiLlmAdapter(LlmPort):
    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None,
        model_id: str,
        request_timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._request_timeout_seconds = request_timeout_seconds

    async def generate_response(self, prompt: LlmPromptInput) -> AIResponsePlan:
        if not self._api_key:
            raise ProviderUnavailableError(
                "Gemini API key is not configured. Set EDGE_AI_GEMINI_API_KEY to enable a real adapter."
            )

        raise ProviderUnavailableError(
            "Gemini adapter scaffolding is present but the provider-specific HTTP request and strict structured response parsing are still intentionally left as a placeholder. "
            "The rest of the service already depends only on LlmPort so this adapter can be completed or swapped later without changing orchestration code. "
            f"Configured model_id='{self._model_id}', timeout_seconds={self._request_timeout_seconds}."
        )

