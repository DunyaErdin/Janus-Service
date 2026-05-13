from __future__ import annotations

import logging

from app.domain.ports.provider_errors import ProviderInvocationError, ProviderUnavailableError
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest

logger = logging.getLogger("edge_ai.tts.chained")


class ChainedTtsAdapter(TtsPort):
    """Tries providers in declaration order and returns the first generated plan.

    Skips providers that return status != "generated" or raise provider errors.
    Returns status="unavailable" only when all providers are exhausted — the
    websocket route will then omit audio output and let the firmware play its
    local fallback tone.
    """

    provider_name = "chained_tts"

    def __init__(self, *providers: TtsPort) -> None:
        if not providers:
            raise ValueError("ChainedTtsAdapter requires at least one provider.")
        self._providers = providers

    @property
    def providers(self) -> tuple[TtsPort, ...]:
        return self._providers

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        last_error: str | None = None

        for provider in self._providers:
            try:
                plan = await provider.plan_synthesis(request)
            except (ProviderUnavailableError, ProviderInvocationError) as exc:
                last_error = str(exc)
                logger.warning(
                    "tts_chain_provider_error",
                    extra={
                        "structured": {
                            "provider": provider.provider_name,
                            "error": last_error,
                        }
                    },
                )
                continue

            if plan.status == "generated" and plan.data_base64:
                logger.info(
                    "tts_chain_hit",
                    extra={"structured": {"provider": provider.provider_name}},
                )
                return plan

            logger.info(
                "tts_chain_skip",
                extra={
                    "structured": {
                        "provider": provider.provider_name,
                        "status": plan.status,
                    }
                },
            )

        logger.error(
            "tts_chain_exhausted",
            extra={
                "structured": {
                    "text_preview": request.text[:60],
                    "last_error": last_error,
                }
            },
        )
        return TtsSynthesisPlan(
            provider=self.provider_name,
            status="unavailable",
        )
