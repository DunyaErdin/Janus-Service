from __future__ import annotations

import logging

from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest

logger = logging.getLogger("edge_ai.tts.fallback_text")


class FallbackTextTtsAdapter(TtsPort):
    """Substitutes request text with a fixed phrase before delegating to inner provider.

    Used as the last resort in a chain: if the dynamic AI text is not in cache
    and OpenRouter has failed, substitute with a known cached phrase so the
    user still hears audio instead of silence.
    """

    provider_name = "cached_pcm_fallback"

    def __init__(self, inner: TtsPort, fallback_text: str) -> None:
        if not fallback_text.strip():
            raise ValueError("fallback_text must not be empty.")
        self._inner = inner
        self._fallback_text = fallback_text

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        substituted = request.model_copy(update={"text": self._fallback_text})
        plan = await self._inner.plan_synthesis(substituted)
        if plan.status == "generated":
            logger.info(
                "tts_fallback_text_hit",
                extra={
                    "structured": {
                        "fallback_text": self._fallback_text,
                        "original_text_length": len(request.text),
                        "inner_provider": self._inner.provider_name,
                    }
                },
            )
        return plan
