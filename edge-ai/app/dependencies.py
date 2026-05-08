from __future__ import annotations

from functools import lru_cache

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
from app.config import get_settings
from app.domain.ports.llm_port import LlmPort
from app.domain.ports.session_repository_port import DeviceSessionRepositoryPort
from app.domain.ports.stt_port import SttPort
from app.domain.ports.telemetry_port import TelemetryPort
from app.domain.ports.tts_port import TtsPort
from app.domain.ports.wake_detection_port import WakeDetectionService
from app.infrastructure.adapters.llm.gemini_llm_adapter import GeminiLlmAdapter
from app.infrastructure.adapters.llm.mock_llm_adapter import MockLlmAdapter
from app.infrastructure.adapters.repositories.in_memory_session_repository import (
    InMemorySessionRepository,
)
from app.infrastructure.adapters.stt.gemini_stt_adapter import GeminiSttAdapter
from app.infrastructure.adapters.stt.placeholder_stt_adapter import (
    PlaceholderSttAdapter,
)
from app.infrastructure.adapters.telemetry.json_logger_telemetry_adapter import (
    JsonLoggerTelemetryAdapter,
)
from app.infrastructure.adapters.tts.gemini_tts_adapter import GeminiTtsAdapter
from app.infrastructure.adapters.tts.placeholder_tts_adapter import (
    PlaceholderTtsAdapter,
)
from app.infrastructure.transport.websocket.connection_manager import ConnectionManager


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


@lru_cache
def get_llm_adapter() -> LlmPort:
    settings = get_settings()
    if settings.llm_provider == "gemini":
        return GeminiLlmAdapter(
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_model_id,
            request_timeout_seconds=settings.request_timeout_seconds,
        )
    return MockLlmAdapter()


@lru_cache
def get_stt_adapter() -> SttPort:
    settings = get_settings()
    if settings.stt_provider == "gemini":
        return GeminiSttAdapter(
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_stt_model_id,
            request_timeout_seconds=settings.request_timeout_seconds,
        )
    return PlaceholderSttAdapter()


@lru_cache
def get_tts_adapter() -> TtsPort:
    settings = get_settings()
    if settings.tts_provider == "gemini":
        return GeminiTtsAdapter(
            api_key=settings.gemini_api_key,
            model_id=settings.gemini_tts_model_id,
            voice_name=settings.gemini_tts_voice_name,
            request_timeout_seconds=settings.request_timeout_seconds,
        )
    return PlaceholderTtsAdapter()


@lru_cache
def get_wake_detection_service() -> WakeDetectionService:
    settings = get_settings()
    if settings.wake_detector_provider == "dev_fake":
        return DevFakeWakeDetectionService()
    if settings.wake_detector_provider == "disabled":
        return DisabledWakeDetectionService()
    return SttWakeDetectionService(get_stt_adapter())


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
