from __future__ import annotations

import base64
import math

from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.provider_errors import (
    ProviderInvocationError,
    ProviderUnavailableError,
)
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest

GREETING_TEXT = "Size nasıl yardımcı olabilirim?"
GOLDEN_PCM_SINE_REQUEST_TEXT = "__janus_golden_pcm_sine__"


class GreetingService:
    def __init__(self, tts: TtsPort) -> None:
        self._tts = tts

    async def build_greeting(
        self,
        *,
        device_id: str,
        interaction_id: str,
        requested_text: str,
        sample_rate_hz: int,
        channels: int,
    ) -> TtsSynthesisPlan:
        if requested_text.strip() == GOLDEN_PCM_SINE_REQUEST_TEXT:
            return _build_golden_pcm_sine(interaction_id)

        text = requested_text.strip() or GREETING_TEXT
        plan = await self._tts.plan_synthesis(
            TtsSynthesisRequest(
                device_id=device_id,
                session_id=interaction_id,
                text=text,
                voice_style=VoiceStyle.CALM,
            )
        )
        if plan.data_base64 is None or plan.encoding is None:
            raise ProviderUnavailableError("Greeting TTS did not produce audio bytes.")
        if plan.encoding != "pcm16":
            raise ProviderInvocationError(
                f"Greeting TTS returned unsupported encoding {plan.encoding!r}."
            )
        if plan.sample_rate_hz not in {sample_rate_hz, 24_000}:
            raise ProviderInvocationError(
                f"Greeting TTS returned {plan.sample_rate_hz} Hz; expected {sample_rate_hz} Hz or 24000 Hz."
            )
        if plan.channels not in {channels, 1}:
            raise ProviderInvocationError(
                f"Greeting TTS returned {plan.channels} channels; expected {channels} or mono."
            )
        return plan


def _build_golden_pcm_sine(session_id: str) -> TtsSynthesisPlan:
    sample_rate_hz = 24_000
    frequency_hz = 1_000
    amplitude = 12_000
    sample_count = sample_rate_hz
    pcm = bytearray()
    for index in range(sample_count):
        sample = round(
            amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate_hz)
        )
        pcm.extend(sample.to_bytes(2, "little", signed=True))
    return TtsSynthesisPlan(
        provider="edge_golden_pcm_sine",
        status="generated",
        reference=session_id,
        encoding="pcm16",
        sample_rate_hz=sample_rate_hz,
        channels=1,
        data_base64=base64.b64encode(bytes(pcm)).decode("ascii"),
        mime_type="audio/L16;rate=24000",
    )
