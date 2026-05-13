from __future__ import annotations

import asyncio
import logging

import httpx

from app.domain.ports.provider_errors import ProviderInvocationError, ProviderUnavailableError
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest
from app.infrastructure.audio.tts_normalizer import (
    NormalizedTtsAudio,
    encode_normalized_audio,
    normalize_pcm16le,
)

_TARGET_SAMPLE_RATE_HZ = 24_000
_TARGET_CHANNELS = 1

logger = logging.getLogger("edge_ai.tts.openai")


class OpenAiTtsAdapter(TtsPort):
    provider_name = "openai_tts"
    _API_URL = "https://api.openai.com/v1/audio/speech"
    _MAX_ATTEMPTS = 3
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        voice: str,
        request_timeout_seconds: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._request_timeout_seconds = request_timeout_seconds

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        if not self._api_key:
            raise ProviderUnavailableError(
                "OpenAI API key is not configured. Set EDGE_AI_OPENAI_API_KEY to enable TTS."
            )

        payload = {
            "model": self._model,
            "input": request.text,
            "voice": self._voice,
            "response_format": "pcm",
        }

        async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
            for attempt in range(1, self._MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        self._API_URL,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                    break
                except httpx.TimeoutException as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        raise ProviderInvocationError("OpenAI TTS request timed out.") from exc
                except httpx.HTTPStatusError as exc:
                    if (
                        attempt >= self._MAX_ATTEMPTS
                        or exc.response.status_code not in self._RETRYABLE_STATUS_CODES
                    ):
                        raise ProviderInvocationError(
                            f"OpenAI TTS request failed with status "
                            f"{exc.response.status_code}: {exc.response.text[:500]}"
                        ) from exc
                except httpx.HTTPError as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        raise ProviderInvocationError(
                            "OpenAI TTS request failed before a valid response was received."
                        ) from exc
                await asyncio.sleep(0.4 * attempt)

        # OpenAI `pcm` format: raw PCM16 LE, 24 kHz, mono — matches firmware contract exactly.
        raw_pcm = response.content
        if not raw_pcm:
            raise ProviderInvocationError("OpenAI TTS returned an empty audio payload.")
        if len(raw_pcm) % 2 != 0:
            raise ProviderInvocationError(
                "OpenAI TTS returned an odd-length PCM payload; expected 16-bit samples."
            )

        logger.info(
            "openai_tts_received",
            extra={
                "structured": {
                    "model": self._model,
                    "voice": self._voice,
                    "byte_length": len(raw_pcm),
                }
            },
        )

        # normalize_pcm16le is a no-op when rate=24000 and channels=1, but validates the payload.
        normalized_pcm = normalize_pcm16le(
            raw_pcm,
            sample_rate_hz=_TARGET_SAMPLE_RATE_HZ,
            channels=_TARGET_CHANNELS,
        )
        normalized = NormalizedTtsAudio(pcm_s16le=normalized_pcm)

        return TtsSynthesisPlan(
            provider=self.provider_name,
            status="generated",
            encoding="pcm16",
            sample_rate_hz=_TARGET_SAMPLE_RATE_HZ,
            channels=_TARGET_CHANNELS,
            data_base64=encode_normalized_audio(normalized),
            mime_type="audio/L16;rate=24000",
        )
