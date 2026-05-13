from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.application.orchestrators.conversation_orchestrator import (
    ConversationOrchestrator,
)
from app.application.services.fallback_response_service import FallbackResponseService
from app.application.services.greeting_service import GreetingService
from app.application.services.prompt_builder import PromptBuilder
from app.application.services.response_validator import ResponseValidator
from app.application.services.touch_interpreter import TouchInterpreter
from app.application.services.wake_detection_service import (
    DevFakeWakeDetectionService,
    DisabledWakeDetectionService,
    SttWakeDetectionService,
)
from app.config import Settings, get_settings
from app.domain.ports.llm_port import LlmPort
from app.domain.ports.session_repository_port import DeviceSessionRepositoryPort
from app.domain.ports.stt_port import SttPort
from app.domain.ports.telemetry_port import TelemetryPort
from app.domain.ports.tts_port import TtsPort
from app.domain.ports.wake_detection_port import WakeDetectionService
from app.infrastructure.adapters.llm.claude_llm_adapter import ClaudeLlmAdapter
from app.infrastructure.adapters.llm.gemini_llm_adapter import GeminiLlmAdapter
from app.infrastructure.adapters.llm.mock_llm_adapter import MockLlmAdapter
from app.infrastructure.adapters.repositories.in_memory_session_repository import (
    InMemorySessionRepository,
)
from app.infrastructure.adapters.stt.gemini_stt_adapter import GeminiSttAdapter
from app.infrastructure.adapters.stt.placeholder_stt_adapter import PlaceholderSttAdapter
from app.infrastructure.adapters.telemetry.json_logger_telemetry_adapter import (
    JsonLoggerTelemetryAdapter,
)
from app.infrastructure.adapters.tts.cached_pcm_tts_adapter import CachedPcmTtsAdapter
from app.infrastructure.adapters.tts.chained_tts_adapter import ChainedTtsAdapter
from app.infrastructure.adapters.tts.fallback_text_tts_adapter import FallbackTextTtsAdapter
from app.infrastructure.adapters.tts.gemini_tts_adapter import GeminiTtsAdapter
from app.infrastructure.adapters.tts.openai_tts_adapter import OpenAiTtsAdapter
from app.infrastructure.adapters.tts.openrouter_tts_adapter import OpenRouterTtsAdapter
from app.infrastructure.adapters.tts.placeholder_tts_adapter import PlaceholderTtsAdapter
from app.infrastructure.transport.websocket.connection_manager import ConnectionManager

logger = logging.getLogger("edge_ai.dependencies")


# ── Singletons ────────────────────────────────────────────────────────────────


@lru_cache
def get_connection_manager() -> ConnectionManager:
    settings = get_settings()
    return ConnectionManager(
        stale_after_seconds=settings.websocket_receive_timeout_seconds,
        close_timeout_seconds=settings.websocket_close_timeout_seconds,
    )


@lru_cache
def get_session_repository() -> DeviceSessionRepositoryPort:
    return InMemorySessionRepository()


@lru_cache
def get_telemetry_adapter() -> TelemetryPort:
    return JsonLoggerTelemetryAdapter()


@lru_cache
def get_touch_interpreter() -> TouchInterpreter:
    return TouchInterpreter()


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    settings = get_settings()
    return PromptBuilder(
        robot_name=settings.robot_name,
        default_language=settings.default_language,
    )


@lru_cache
def get_response_validator() -> ResponseValidator:
    return ResponseValidator()


@lru_cache
def get_fallback_response_service() -> FallbackResponseService:
    return FallbackResponseService()


# ── LLM ───────────────────────────────────────────────────────────────────────


@lru_cache
def get_llm_adapter() -> LlmPort:
    settings = get_settings()

    if settings.llm_provider == "gemini":
        # gemini_enabled guard is enforced by config validator; this branch
        # can only be reached when gemini_enabled=True.
        return GeminiLlmAdapter(
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_model_id,
            request_timeout_seconds=settings.request_timeout_seconds,
        )

    if settings.llm_provider == "claude":
        if not settings.anthropic_api_key:
            logger.error(
                "provider_config_error",
                extra={
                    "structured": {
                        "provider": "claude",
                        "error": "ANTHROPIC_API_KEY is not set — Claude LLM will fail at runtime",
                    }
                },
            )
        return ClaudeLlmAdapter(
            api_key=settings.anthropic_api_key,
            model_id=settings.claude_model_id,
            max_tokens=settings.claude_max_tokens,
            request_timeout_seconds=settings.request_timeout_seconds,
        )

    return MockLlmAdapter()


# ── STT ───────────────────────────────────────────────────────────────────────


@lru_cache
def get_stt_adapter() -> SttPort:
    settings = get_settings()

    if settings.stt_provider == "gemini":
        # gemini_enabled guard is enforced by config validator.
        return GeminiSttAdapter(
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_stt_model_id,
            request_timeout_seconds=settings.request_timeout_seconds,
        )

    # "mock" and "placeholder" both resolve to no-op STT.
    return PlaceholderSttAdapter()


# ── TTS ───────────────────────────────────────────────────────────────────────


def _build_openrouter_adapter(settings: Settings) -> OpenRouterTtsAdapter:
    if not settings.openrouter_api_key:
        logger.error(
            "provider_config_error",
            extra={
                "structured": {
                    "provider": "openrouter_tts",
                    "error": "OPENROUTER_API_KEY is not set — TTS will fail at runtime",
                }
            },
        )
    return OpenRouterTtsAdapter(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_tts_base_url,
        model=settings.openrouter_tts_model,
        voice=settings.openrouter_tts_voice,
        speed=settings.openrouter_tts_speed,
        request_timeout_seconds=settings.request_timeout_seconds,
        http_referer=settings.openrouter_http_referer,
        app_title=settings.openrouter_app_title,
    )


@lru_cache
def get_tts_adapter() -> TtsPort:
    settings = get_settings()

    if settings.tts_provider == "gemini":
        # gemini_enabled guard is enforced by config validator.
        return GeminiTtsAdapter(
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_tts_model_id,
            voice_name=settings.gemini_tts_voice_name,
            request_timeout_seconds=settings.request_timeout_seconds,
        )

    if settings.tts_provider == "openai":
        # openai_enabled guard is enforced by config validator.
        return OpenAiTtsAdapter(
            api_key=settings.openai_api_key,
            model=settings.openai_tts_model,
            voice=settings.openai_tts_voice,
            request_timeout_seconds=settings.request_timeout_seconds,
        )

    if settings.tts_provider == "openrouter":
        # Chain: cached_pcm → openrouter → cached_pcm fallback (generic greeting)
        # Order controlled by tts_cache_first.
        cache = CachedPcmTtsAdapter(settings.tts_cache_dir)
        openrouter = _build_openrouter_adapter(settings)
        fallback = FallbackTextTtsAdapter(cache, "Selam adaş, buradayım.")
        if settings.tts_cache_first:
            return ChainedTtsAdapter(cache, openrouter, fallback)
        return ChainedTtsAdapter(openrouter, cache, fallback)

    return PlaceholderTtsAdapter()


# ── Provider diagnostics ──────────────────────────────────────────────────────


def build_provider_matrix(settings: Settings) -> dict[str, Any]:
    """Return a loggable/serialisable view of the active provider configuration."""
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.claude_model_id if settings.llm_provider == "claude" else None,
        "stt_provider": settings.stt_provider,
        "tts_provider": settings.tts_provider,
        "tts_cache_first": settings.tts_cache_first,
        "gemini_enabled": settings.gemini_enabled,
        "native_openai_enabled": settings.openai_enabled,
        "openrouter_configured": bool(settings.openrouter_api_key),
        "anthropic_configured": bool(settings.anthropic_api_key),
    }


def get_tts_health_status() -> dict[str, Any]:
    """Probe the active TTS adapter and return a status dict for /health/tts."""
    settings = get_settings()
    adapter = get_tts_adapter()

    openrouter: OpenRouterTtsAdapter | None = None
    cache: CachedPcmTtsAdapter | None = None

    def _walk(a: TtsPort) -> None:
        nonlocal openrouter, cache
        if isinstance(a, OpenRouterTtsAdapter):
            openrouter = a
        elif isinstance(a, CachedPcmTtsAdapter) and cache is None:
            cache = a
        elif isinstance(a, ChainedTtsAdapter):
            for p in a.providers:
                _walk(p)
        elif isinstance(a, FallbackTextTtsAdapter):
            _walk(a._inner)

    _walk(adapter)

    result: dict[str, Any] = {
        "tts_provider": settings.tts_provider,
        "tts_cache_first": settings.tts_cache_first,
        "cache_available": cache.size > 0 if cache else False,
        "cache_size": cache.size if cache else 0,
    }
    if openrouter:
        result.update(openrouter.health_dict())
    else:
        result.update({
            "provider": settings.tts_provider,
            "model": None,
            "voice": None,
            "status": "not_configured",
            "cooldown_until": None,
            "last_error": None,
        })
    return result


def get_providers_health() -> dict[str, Any]:
    """Full provider matrix for /health/providers."""
    settings = get_settings()
    tts_health = get_tts_health_status()

    claude_status = "not_selected"
    if settings.llm_provider == "claude":
        claude_status = "configured" if settings.anthropic_api_key else "missing_key"

    return {
        "claude": {
            "status": claude_status,
            "model": settings.claude_model_id if settings.llm_provider == "claude" else None,
            "configured": bool(settings.anthropic_api_key),
        },
        "openrouter_tts": tts_health,
        "gemini": {
            "status": "disabled",
            "enabled": settings.gemini_enabled,
        },
        "native_openai": {
            "status": "disabled",
            "enabled": settings.openai_enabled,
        },
        "stt": {
            "provider": settings.stt_provider,
            "status": "mock_noop" if settings.stt_provider in {"mock", "placeholder"} else "active",
        },
    }


# ── Wake detection ────────────────────────────────────────────────────────────


@lru_cache
def get_wake_detection_service() -> WakeDetectionService:
    settings = get_settings()
    if settings.wake_detector_provider == "dev_fake":
        return DevFakeWakeDetectionService()
    if settings.wake_detector_provider == "disabled":
        return DisabledWakeDetectionService()
    return SttWakeDetectionService(get_stt_adapter())


# ── Application services ──────────────────────────────────────────────────────


@lru_cache
def get_greeting_service() -> GreetingService:
    return GreetingService(get_tts_adapter())


@lru_cache
def get_conversation_orchestrator() -> ConversationOrchestrator:
    settings = get_settings()
    return ConversationOrchestrator(
        llm=get_llm_adapter(),
        stt=get_stt_adapter(),
        tts=get_tts_adapter(),
        wake_detection=get_wake_detection_service(),
        greeting_service=get_greeting_service(),
        session_repository=get_session_repository(),
        telemetry=get_telemetry_adapter(),
        prompt_builder=get_prompt_builder(),
        touch_interpreter=get_touch_interpreter(),
        response_validator=get_response_validator(),
        fallback_response_service=get_fallback_response_service(),
        max_audio_chunks_per_session=settings.max_audio_chunks_per_session,
        max_wake_chunks_per_interaction=settings.max_wake_chunks_per_interaction,
        max_wake_base64_chars_per_interaction=settings.max_wake_base64_chars_per_interaction,
        session_history_limit=settings.session_history_limit,
    )
