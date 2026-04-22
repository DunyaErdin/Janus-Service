from __future__ import annotations

from typing import Any

import httpx

from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.ports.llm_port import LlmPort, LlmPromptInput
from app.domain.ports.provider_errors import (
    ProviderInvocationError,
    ProviderUnavailableError,
)
from app.schemas.llm_response_schema import (
    StructuredResponseSchemaError,
    parse_llm_structured_response,
)


class GeminiLlmAdapter(LlmPort):
    provider_name = "gemini"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

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

        request_payload = self._build_request_payload(prompt)
        request_url = f"{self._BASE_URL}/models/{self._model_id}:generateContent"

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
                response = await client.post(
                    request_url,
                    headers={
                        "x-goog-api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderInvocationError("Gemini request timed out before a valid response was received.") from exc
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text[:500]
            raise ProviderInvocationError(
                f"Gemini request failed with status {exc.response.status_code}: {response_text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderInvocationError("Gemini request failed before a valid response was received.") from exc

        return self._parse_generate_content_response(response.json())

    def _build_request_payload(self, prompt: LlmPromptInput) -> dict[str, Any]:
        return {
            "system_instruction": {
                "parts": [
                    {
                        "text": prompt.render_system_instruction(),
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt.render_user_prompt(),
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "candidateCount": 1,
                "responseMimeType": "application/json",
                "responseJsonSchema": prompt.response_schema,
            },
        }

    def _parse_generate_content_response(self, payload: dict[str, Any]) -> AIResponsePlan:
        candidates = payload.get("candidates") or []
        if not candidates:
            prompt_feedback = payload.get("promptFeedback")
            raise ProviderInvocationError(
                f"Gemini returned no candidates. promptFeedback={prompt_feedback!r}"
            )

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason")
        if finish_reason not in {None, "STOP"}:
            raise ProviderInvocationError(
                f"Gemini did not finish cleanly. finishReason={finish_reason!r}"
            )

        raw_text = self._extract_candidate_text(first_candidate)
        if not raw_text:
            raise ProviderInvocationError("Gemini returned an empty candidate body.")

        try:
            return parse_llm_structured_response(raw_text)
        except StructuredResponseSchemaError as exc:
            raise ProviderInvocationError(
                "Gemini returned text that could not be validated as RobotStructuredResponsePlan JSON."
            ) from exc

    def _extract_candidate_text(self, candidate: dict[str, Any]) -> str:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_parts = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts).strip()
