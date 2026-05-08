from __future__ import annotations

import base64
import re
import unicodedata

from app.domain.ports.provider_errors import ProviderUnavailableError
from app.domain.ports.stt_port import SttPort, TranscriptionRequest
from app.domain.ports.wake_detection_port import (
    WakeDetectionRequest,
    WakeDetectionResult,
    WakeDetectionService,
)


class SttWakeDetectionService(WakeDetectionService):
    provider_name = "stt_wake_detection"

    def __init__(self, stt: SttPort) -> None:
        self._stt = stt

    async def detect(self, request: WakeDetectionRequest) -> WakeDetectionResult:
        transcript = await self._stt.transcribe(
            TranscriptionRequest(
                device_id=request.device_id,
                session_id=request.interaction_id,
                audio_chunks=request.audio_chunks,
                encoding=request.encoding,
                sample_rate_hz=request.sample_rate_hz,
                channels=request.channels,
            )
        )
        normalized = normalize_wake_phrase(transcript.text)
        detected = contains_hey_janus(normalized)
        return WakeDetectionResult(
            detected=detected,
            transcript=transcript.text,
            confidence=transcript.confidence
            if transcript.confidence is not None
            else (0.80 if detected else 0.0),
            reason="wake_phrase_detected" if detected else "not_wake_word",
        )


class DevFakeWakeDetectionService(WakeDetectionService):
    provider_name = "dev_fake_wake_detection"

    async def detect(self, request: WakeDetectionRequest) -> WakeDetectionResult:
        decoded = b"".join(
            base64.b64decode(chunk, validate=False)
            for chunk in request.audio_chunks
            if chunk
        )
        haystack = decoded.decode("utf-8", errors="ignore").lower()
        detected = "hey janus" in haystack or "hey janus" in normalize_wake_phrase(
            haystack
        )
        return WakeDetectionResult(
            detected=detected,
            transcript="hey janus" if detected else "",
            confidence=1.0 if detected else 0.0,
            reason="dev_fake_match" if detected else "dev_fake_no_match",
        )


class DisabledWakeDetectionService(WakeDetectionService):
    provider_name = "disabled_wake_detection"

    async def detect(self, request: WakeDetectionRequest) -> WakeDetectionResult:
        raise ProviderUnavailableError("Wake detection is disabled by configuration.")


def normalize_wake_phrase(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    asciiish = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", asciiish).strip()


def contains_hey_janus(normalized_text: str) -> bool:
    tokens = normalized_text.split()
    if len(tokens) < 2:
        return False

    hey_tokens = {"hey", "he", "hay"}
    janus_tokens = {
        "janus",
        "januz",
        "janis",
        "janiz",
        "canus",
        "canuz",
        "canis",
        "caniz",
        "cenus",
    }
    return any(
        left in hey_tokens and right in janus_tokens
        for left, right in zip(tokens, tokens[1:])
    )
