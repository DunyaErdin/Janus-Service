from __future__ import annotations

from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.provider_errors import (
    ProviderInvocationError,
    ProviderUnavailableError,
)
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest

GREETING_TEXT = "Size nasıl yardımcı olabilirim?"


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
        text = (
            GREETING_TEXT
            if requested_text.strip() != GREETING_TEXT
            else requested_text.strip()
        )
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
