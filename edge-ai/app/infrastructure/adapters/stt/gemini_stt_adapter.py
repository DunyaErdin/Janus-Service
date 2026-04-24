from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx

from app.domain.ports.provider_errors import (
    ProviderInvocationError,
    ProviderUnavailableError,
)
from app.domain.ports.stt_port import SttPort, TranscriptionRequest, TranscriptionResult
from app.infrastructure.audio.wav_codec import (
    decode_base64_audio_chunks,
    pcm16le_to_wav_bytes,
)


class GeminiSttAdapter(SttPort):
    provider_name = "gemini_stt"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    _MAX_ATTEMPTS = 3
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

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

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        if not self._api_key:
            raise ProviderUnavailableError(
                "Gemini API key is not configured. Set EDGE_AI_GEMINI_API_KEY to enable STT."
            )
        if request.encoding != "pcm16":
            raise ProviderUnavailableError(
                f"Gemini STT adapter only supports pcm16 input right now, got {request.encoding!r}."
            )

        pcm_bytes = decode_base64_audio_chunks(request.audio_chunks)
        wav_bytes = pcm16le_to_wav_bytes(
            pcm_bytes,
            sample_rate_hz=request.sample_rate_hz,
            channels=request.channels,
        )
        request_payload = self._build_request_payload(wav_bytes)
        request_url = f"{self._BASE_URL}/models/{self._model_id}:generateContent"

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
                        raise ProviderInvocationError("Gemini STT request timed out.") from exc
                except httpx.HTTPStatusError as exc:
                    if (
                        attempt >= self._MAX_ATTEMPTS
                        or exc.response.status_code not in self._RETRYABLE_STATUS_CODES
                    ):
                        response_text = exc.response.text[:500]
                        raise ProviderInvocationError(
                            f"Gemini STT request failed with status {exc.response.status_code}: {response_text}"
                        ) from exc
                except httpx.HTTPError as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        raise ProviderInvocationError(
                            "Gemini STT request failed before a valid response was received."
                        ) from exc

                await asyncio.sleep(0.4 * attempt)

        transcript_text = self._parse_generate_content_response(response.json())
        return TranscriptionResult(text=transcript_text)

    def _build_request_payload(self, wav_bytes: bytes) -> dict[str, Any]:
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": audio_b64,
                            }
                        },
                        {
                            "text": (
                                "Transcribe the spoken audio exactly as plain text. "
                                "Return only the transcript text, without quotes, labels, markdown, or explanations. "
                                "If the speech is unclear, return an empty string."
                            )
                        },
                    ],
                }
            ]
        }

    def _parse_generate_content_response(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            prompt_feedback = payload.get("promptFeedback")
            raise ProviderInvocationError(
                f"Gemini STT returned no candidates. promptFeedback={prompt_feedback!r}"
            )

        first_candidate = candidates[0]
        finish_reason = first_candidate.get("finishReason")
        if finish_reason not in {None, "STOP"}:
            raise ProviderInvocationError(
                f"Gemini STT did not finish cleanly. finishReason={finish_reason!r}"
            )

        raw_text = self._extract_candidate_text(first_candidate)
        return self._normalize_transcript_text(raw_text)

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

    def _normalize_transcript_text(self, raw_text: str) -> str:
        normalized = raw_text.strip()
        if normalized.startswith("```") and normalized.endswith("```"):
            lines = normalized.splitlines()
            if len(lines) >= 2:
                normalized = "\n".join(lines[1:-1]).strip()

        if normalized.startswith("Transcript:"):
            normalized = normalized.partition(":")[2].strip()

        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {'"', "'"}:
            normalized = normalized[1:-1].strip()

        return normalized
