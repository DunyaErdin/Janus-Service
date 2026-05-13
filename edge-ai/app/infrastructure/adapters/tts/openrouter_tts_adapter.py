from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

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
# Cooldown seconds: failure 1 → 60s, failure 2 → 300s, failure 3+ → 1800s
_COOLDOWN_SCHEDULE = (60, 300, 1800)
_RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})
_STATUS_CODE_MAP = {
    401: "auth_error",
    403: "auth_error",
    402: "payment_required",
    404: "model_not_found",
    429: "rate_limited",
}

logger = logging.getLogger("edge_ai.tts.openrouter")


class OpenRouterTtsAdapter(TtsPort):
    """TTS via OpenRouter /audio/speech — PCM16 LE 24 kHz mono output.

    Non-200 responses are parsed as JSON error bodies and never forwarded to
    firmware. Retryable errors (429, 5xx) trigger an exponential cooldown so
    the provider is skipped until the backoff expires.
    """

    provider_name = "openrouter_tts"
    _MAX_ATTEMPTS = 3

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        voice: str,
        speed: float,
        request_timeout_seconds: float,
        http_referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self._speed = speed
        self._timeout = request_timeout_seconds
        self._http_referer = http_referer
        self._app_title = app_title

        self._lock: asyncio.Lock = asyncio.Lock()
        self._failure_count: int = 0
        self._cooldown_until: datetime | None = None
        self._last_error: str | None = None
        self._last_error_code: str | None = None

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        if not self._api_key:
            raise ProviderUnavailableError(
                "OpenRouter API key is not configured. Set EDGE_AI_OPENROUTER_API_KEY."
            )

        async with self._lock:
            if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
                raise ProviderUnavailableError(
                    f"OpenRouter TTS in cooldown until {self._cooldown_until.isoformat()}"
                )

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._app_title:
            headers["X-Title"] = self._app_title

        payload: dict[str, Any] = {
            "model": self._model,
            "input": request.text,
            "voice": self._voice,
            "response_format": "pcm",
            "speed": self._speed,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, self._MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        f"{self._base_url}/audio/speech",
                        headers=headers,
                        json=payload,
                    )
                except httpx.TimeoutException as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        await self._record_error("timeout", "request timed out", retryable=True)
                        raise ProviderInvocationError(
                            "OpenRouter TTS request timed out."
                        ) from exc
                    await asyncio.sleep(0.4 * attempt)
                    continue
                except httpx.HTTPError as exc:
                    if attempt >= self._MAX_ATTEMPTS:
                        await self._record_error("network_error", str(exc), retryable=True)
                        raise ProviderInvocationError(
                            "OpenRouter TTS request failed before a valid response."
                        ) from exc
                    await asyncio.sleep(0.4 * attempt)
                    continue

                if response.status_code != 200:
                    error_code, detail = self._parse_error_response(response)
                    retryable = response.status_code in _RETRYABLE_CODES

                    logger.warning(
                        "openrouter_tts_error",
                        extra={
                            "structured": {
                                "http_status": response.status_code,
                                "error_code": error_code,
                                "detail": detail,
                                "attempt": attempt,
                                "retryable": retryable,
                            }
                        },
                    )

                    if not retryable or attempt >= self._MAX_ATTEMPTS:
                        await self._record_error(error_code, detail, retryable=retryable)
                        raise ProviderInvocationError(
                            f"OpenRouter TTS error {response.status_code}"
                            f" ({error_code}): {detail}"
                        )
                    await asyncio.sleep(0.4 * attempt)
                    continue

                # 200 OK — validate PCM payload before returning.
                raw_pcm = response.content
                if not raw_pcm:
                    raise ProviderInvocationError(
                        "OpenRouter TTS returned an empty audio payload."
                    )
                if len(raw_pcm) % 2 != 0:
                    raise ProviderInvocationError(
                        "OpenRouter TTS returned an odd-length PCM payload;"
                        " expected 16-bit samples."
                    )

                async with self._lock:
                    self._failure_count = 0
                    self._cooldown_until = None
                    self._last_error = None
                    self._last_error_code = None

                logger.info(
                    "openrouter_tts_received",
                    extra={
                        "structured": {
                            "model": self._model,
                            "voice": self._voice,
                            "byte_length": len(raw_pcm),
                            "tts_provider": self.provider_name,
                            "tts_model": self._model,
                            "tts_voice": self._voice,
                            "tts_status": "generated",
                            "tts_http_status": 200,
                            "normalized_sample_rate_hz": _TARGET_SAMPLE_RATE_HZ,
                            "normalized_channels": _TARGET_CHANNELS,
                            "normalized_format": "s16le",
                            "normalized_pcm_bytes": len(raw_pcm),
                        }
                    },
                )

                normalized = NormalizedTtsAudio(
                    pcm_s16le=normalize_pcm16le(
                        raw_pcm,
                        sample_rate_hz=_TARGET_SAMPLE_RATE_HZ,
                        channels=_TARGET_CHANNELS,
                    )
                )

                return TtsSynthesisPlan(
                    provider=self.provider_name,
                    status="generated",
                    encoding="pcm16",
                    sample_rate_hz=_TARGET_SAMPLE_RATE_HZ,
                    channels=_TARGET_CHANNELS,
                    data_base64=encode_normalized_audio(normalized),
                    mime_type="audio/L16;rate=24000",
                )

        raise ProviderInvocationError("OpenRouter TTS exhausted all retry attempts.")

    async def _record_error(self, error_code: str, detail: str, *, retryable: bool) -> None:
        async with self._lock:
            self._last_error_code = error_code
            self._last_error = detail[:200]
            if retryable:
                self._failure_count += 1
                idx = min(self._failure_count - 1, len(_COOLDOWN_SCHEDULE) - 1)
                self._cooldown_until = datetime.now(timezone.utc) + timedelta(
                    seconds=_COOLDOWN_SCHEDULE[idx]
                )
                logger.warning(
                    "openrouter_tts_cooldown_set",
                    extra={
                        "structured": {
                            "failure_count": self._failure_count,
                            "cooldown_seconds": _COOLDOWN_SCHEDULE[idx],
                            "cooldown_until": self._cooldown_until.isoformat(),
                        }
                    },
                )

    def _parse_error_response(self, response: httpx.Response) -> tuple[str, str]:
        error_code = _STATUS_CODE_MAP.get(
            response.status_code,
            "provider_unavailable" if response.status_code >= 500 else "provider_error",
        )
        try:
            body = response.json()
            err = body.get("error") or {}
            detail = str(err.get("message") or response.text[:200])
        except Exception:
            detail = response.text[:200]
        return error_code, detail

    def health_dict(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        in_cooldown = bool(self._cooldown_until and now < self._cooldown_until)

        if not self._api_key:
            status = "no_key"
        elif in_cooldown:
            status = "cooldown"
        elif self._last_error_code:
            status = self._last_error_code
        else:
            status = "ok"

        return {
            "provider": self.provider_name,
            "model": self._model,
            "voice": self._voice,
            "status": status,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "last_error": self._last_error,
        }
