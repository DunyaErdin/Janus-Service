from __future__ import annotations

import asyncio

from app.application.orchestrators.conversation_orchestrator import ConversationOrchestrator
from app.application.services.fallback_response_service import FallbackResponseService
from app.application.services.prompt_builder import PromptBuilder
from app.application.services.response_validator import ResponseValidator
from app.application.services.touch_interpreter import TouchInterpreter
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.device_event import AudioChunkEvent, AudioEncoding, SessionStartEvent
from app.domain.ports.llm_port import LlmPort, LlmPromptInput
from app.domain.ports.provider_errors import ProviderUnavailableError
from app.domain.ports.stt_port import SttPort, TranscriptionRequest, TranscriptionResult
from app.domain.ports.telemetry_port import TelemetryEvent, TelemetryPort
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest
from app.infrastructure.adapters.repositories.in_memory_session_repository import (
    InMemorySessionRepository,
)


class DummyLlmAdapter(LlmPort):
    provider_name = "dummy_llm"

    async def generate_response(self, prompt: LlmPromptInput):
        raise AssertionError("LLM should not be invoked when STT is unavailable.")


class UnavailableSttAdapter(SttPort):
    provider_name = "unavailable_stt"

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise ProviderUnavailableError("stt unavailable")


class CapturingTtsAdapter(TtsPort):
    provider_name = "capturing_tts"

    def __init__(self) -> None:
        self.requests: list[TtsSynthesisRequest] = []

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        self.requests.append(request)
        return TtsSynthesisPlan(
            provider=self.provider_name,
            status="generated",
            encoding="pcm16",
            sample_rate_hz=24000,
            channels=1,
            data_base64="AQID",
            mime_type="audio/L16;rate=24000",
        )


class NoopTelemetry(TelemetryPort):
    async def publish(self, event: TelemetryEvent) -> None:
        return None


def test_audio_final_with_unavailable_stt_returns_fallback_plan_and_tts() -> None:
    async def scenario() -> None:
        tts = CapturingTtsAdapter()
        orchestrator = ConversationOrchestrator(
            llm=DummyLlmAdapter(),
            stt=UnavailableSttAdapter(),
            tts=tts,
            session_repository=InMemorySessionRepository(),
            telemetry=NoopTelemetry(),
            prompt_builder=PromptBuilder(robot_name="Janus", default_language="tr-TR"),
            touch_interpreter=TouchInterpreter(),
            response_validator=ResponseValidator(),
            fallback_response_service=FallbackResponseService(),
            max_audio_chunks_per_session=64,
            session_history_limit=20,
        )

        await orchestrator.handle_event(
            SessionStartEvent(
                device_id="janus-esp-01",
                requested_session_id="session-1",
                trigger="panel",
            )
        )

        result = await orchestrator.handle_event(
            AudioChunkEvent(
                device_id="janus-esp-01",
                chunk_id=0,
                encoding=AudioEncoding.PCM16,
                sample_rate_hz=16000,
                channels=1,
                data_base64="AAAA",
                is_final=True,
            )
        )

        assert result.response_plan is not None
        assert result.response_plan.spoken_text == "Seni duydum ama su an kucuk bir sorun yasadim."
        assert result.tts_plan is not None
        assert result.tts_plan.data_base64 == "AQID"
        assert tts.requests
        assert tts.requests[0].voice_style == VoiceStyle.CALM

    asyncio.run(scenario())
