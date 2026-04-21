from __future__ import annotations

from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest


class PlaceholderTtsAdapter(TtsPort):
    provider_name = "placeholder_tts"

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        return TtsSynthesisPlan(
            provider=self.provider_name,
            status="skipped",
            reference=None,
        )

