from __future__ import annotations

import asyncio
import json
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


class ClaudeLlmAdapter(LlmPort):
    provider_name = "claude"
    _MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    _ANTHROPIC_VERSION = "2023-06-01"
    _MAX_ATTEMPTS = 3
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_key: str | None,
        model_id: str,
        max_tokens: int,
        request_timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._request_timeout_seconds = request_timeout_seconds

    async def generate_response(self, prompt: LlmPromptInput) -> AIResponsePlan:
        if not self._api_key:
            raise ProviderUnavailableError(
                "Anthropic API key is not configured. Set EDGE_AI_ANTHROPIC_API_KEY to enable Claude."
            )

        request_payload = self._build_request_payload(prompt)

        async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
            for attempt in range(1, self._MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        self._MESSAGES_URL,
                        headers={
                            "x-api-key": self._api_key,
                            "anthropic-version": self._ANTHROPIC_VERSION,
                            "content-type": "application/json",
                        },
                        json=request_payload,
                    )
                    response.raise_for_status()
                    break
                except httpx.TimeoutException as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        raise ProviderInvocationError(
                            "Claude request timed out before a valid response was received."
                        ) from exc
                except httpx.HTTPStatusError as exc:
                    if (
                        attempt >= self._MAX_ATTEMPTS
                        or exc.response.status_code not in self._RETRYABLE_STATUS_CODES
                    ):
                        response_text = exc.response.text[:500]
                        raise ProviderInvocationError(
                            f"Claude request failed with status {exc.response.status_code}: {response_text}"
                        ) from exc
                except httpx.HTTPError as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        raise ProviderInvocationError(
                            "Claude request failed before a valid response was received."
                        ) from exc

                await asyncio.sleep(0.4 * attempt)

        return self._parse_messages_response(response.json())

    def _build_request_payload(self, prompt: LlmPromptInput) -> dict[str, Any]:
        user_prompt = "\n\n".join(
            [
                prompt.render_user_prompt(),
                "Return exactly one valid JSON object matching this schema. Do not wrap it in markdown.",
                json.dumps(prompt.response_schema, ensure_ascii=False),
            ]
        )
        return {
            "model": self._model_id,
            "max_tokens": self._max_tokens,
            "temperature": 0.2,
            "system": prompt.render_system_instruction(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        }
                    ],
                }
            ],
        }

    def _parse_messages_response(self, payload: dict[str, Any]) -> AIResponsePlan:
        stop_reason = payload.get("stop_reason")
        if stop_reason not in {None, "end_turn", "stop_sequence"}:
            raise ProviderInvocationError(
                f"Claude did not finish cleanly. stop_reason={stop_reason!r}"
            )

        raw_text = self._extract_text(payload)
        if not raw_text:
            raise ProviderInvocationError("Claude returned an empty message body.")

        try:
            return parse_llm_structured_response(self._strip_markdown_fence(raw_text))
        except StructuredResponseSchemaError as exc:
            raise ProviderInvocationError(
                "Claude returned text that could not be validated as RobotStructuredResponsePlan JSON."
            ) from exc

    def _extract_text(self, payload: dict[str, Any]) -> str:
        content_blocks = payload.get("content") or []
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts).strip()

    def _strip_markdown_fence(self, raw_text: str) -> str:
        stripped = raw_text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            return stripped
        return "\n".join(lines[1:-1]).strip()
