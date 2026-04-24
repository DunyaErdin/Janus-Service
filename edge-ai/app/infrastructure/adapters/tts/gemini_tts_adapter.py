from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.provider_errors import (
    ProviderInvocationError,
    ProviderUnavailableError,
)
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest

_STYLE_DIRECTIVES: dict[VoiceStyle, str] = {
    VoiceStyle.CALM: "Speak calmly and clearly.",
    VoiceStyle.WARM: "Speak warmly and kindly.",
    VoiceStyle.SOFT: "Speak softly and gently.",
    VoiceStyle.ENERGETIC: "Speak with bright energy and momentum.",
    VoiceStyle.SERIOUS: "Speak in a serious, precise, composed tone.",
    VoiceStyle.PLAYFUL: "Speak playfully and lightly.",
    VoiceStyle.SLEEPY: "Speak softly with a drowsy tone.",
    VoiceStyle.CHEERFUL: "Speak cheerfully and optimistically.",
}


class GeminiTtsAdapter(TtsPort):
    provider_name = "gemini_tts"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    _MAX_ATTEMPTS = 3
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_key: str | None,
        model_id: str,
        voice_name: str,
        request_timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._voice_name = voice_name
        self._request_timeout_seconds = request_timeout_seconds

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        if not self._api_key:
            raise ProviderUnavailableError(
                "Gemini API key is not configured. Set EDGE_AI_GEMINI_API_KEY to enable TTS."
            )

        request_url = f"{self._BASE_URL}/models/{self._model_id}:generateContent"
        request_payload = self._build_request_payload(request)

        async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
            for attempt in range(1, self._MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        request_url,
                        headers={
                            "x-goog-api-key": self._api_key,
                            "Content-Type": "application/json",
                        },
                        json=request_payload,
                    )
                    response.raise_for_status()
                    break
                except httpx.TimeoutException as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        raise ProviderInvocationError("Gemini TTS request timed out.") from exc
                except httpx.HTTPStatusError as exc:
                    if (
                        attempt >= self._MAX_ATTEMPTS
                        or exc.response.status_code not in self._RETRYABLE_STATUS_CODES
                    ):
                        response_text = exc.response.text[:500]
                        raise ProviderInvocationError(
                            f"Gemini TTS request failed with status {exc.response.status_code}: {response_text}"
                        ) from exc
                except httpx.HTTPError as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        raise ProviderInvocationError(
                            "Gemini TTS request failed before a valid response was received."
                        ) from exc

                await asyncio.sleep(0.4 * attempt)

        return self._parse_generate_content_response(response.json())

    def _build_request_payload(self, request: TtsSynthesisRequest) -> dict[str, Any]:
        style_directive = _STYLE_DIRECTIVES.get(
            request.voice_style,
            _STYLE_DIRECTIVES[VoiceStyle.CALM],
        )
        prompt = (
            f"{style_directive} "
            "Recite the following text exactly as written. "
            "Do not add extra words, introductions, or explanations.\n\n"
            f"Text: {request.text}"
        )
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self._voice_name,
                        }
                    }
                },
            },
        }

    def _parse_generate_content_response(self, payload: dict[str, Any]) -> TtsSynthesisPlan:
        candidates = payload.get("candidates") or []
        if not candidates:
            prompt_feedback = payload.get("promptFeedback")
            raise ProviderInvocationError(
                f"Gemini TTS returned no candidates. promptFeedback={prompt_feedback!r}"
            )

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason")
        if finish_reason not in {None, "STOP"}:
            raise ProviderInvocationError(
                f"Gemini TTS did not finish cleanly. finishReason={finish_reason!r}"
            )

        content = first_candidate.get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            raise ProviderInvocationError("Gemini TTS returned no content parts.")

        inline_data = parts[0].get("inlineData") if isinstance(parts[0], dict) else None
        if not isinstance(inline_data, dict):
            raise ProviderInvocationError("Gemini TTS response did not include inline audio data.")

        audio_b64 = inline_data.get("data")
        if not isinstance(audio_b64, str) or not audio_b64.strip():
            raise ProviderInvocationError("Gemini TTS response did not include audio bytes.")

        mime_type = inline_data.get("mimeType")
        return TtsSynthesisPlan(
            provider=self.provider_name,
            status="generated",
            encoding="pcm16",
            sample_rate_hz=24_000,
            channels=1,
            data_base64=audio_b64,
            mime_type=mime_type if isinstance(mime_type, str) else "audio/L16;rate=24000",
        )
