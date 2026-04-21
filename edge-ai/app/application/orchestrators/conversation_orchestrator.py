from __future__ import annotations

import time
from datetime import datetime, timezone

from pydantic import BaseModel

from app.application.services.fallback_response_service import FallbackResponseService
from app.application.services.prompt_builder import PromptBuilder
from app.application.services.response_validator import (
    ResponseValidationError,
    ResponseValidator,
)
from app.application.services.touch_interpreter import TouchInterpreter
from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.models.device_event import (
    AudioChunkEvent,
    DeviceEvent,
    HeartbeatEvent,
    HelloEvent,
    SessionEndEvent,
    SessionStartEvent,
    StatusEvent,
    TouchEvent,
)
from app.domain.models.session_context import (
    AudioBufferState,
    ConversationTurn,
    DeviceSessionContext,
)
from app.domain.models.touch_context import TouchContext, TouchInterpretation
from app.domain.ports.llm_port import LlmPort
from app.domain.ports.provider_errors import (
    ProviderInvocationError,
    ProviderUnavailableError,
)
from app.domain.ports.session_repository_port import DeviceSessionRepositoryPort
from app.domain.ports.stt_port import SttPort, TranscriptionRequest
from app.domain.ports.telemetry_port import TelemetryEvent, TelemetryPort
from app.domain.ports.tts_port import TtsPort, TtsSynthesisRequest


class OrchestrationResult(BaseModel):
    device_id: str
    session_id: str | None
    ack_message: str
    response_plan: AIResponsePlan | None = None
    provider_used: str | None = None
    error_category: str | None = None


class ConversationOrchestrator:
    def __init__(
        self,
        *,
        llm: LlmPort,
        stt: SttPort,
        tts: TtsPort,
        session_repository: DeviceSessionRepositoryPort,
        telemetry: TelemetryPort,
        prompt_builder: PromptBuilder,
        touch_interpreter: TouchInterpreter,
        response_validator: ResponseValidator,
        fallback_response_service: FallbackResponseService,
        max_audio_chunks_per_session: int,
        session_history_limit: int,
    ) -> None:
        self._llm = llm
        self._stt = stt
        self._tts = tts
        self._session_repository = session_repository
        self._telemetry = telemetry
        self._prompt_builder = prompt_builder
        self._touch_interpreter = touch_interpreter
        self._response_validator = response_validator
        self._fallback_response_service = fallback_response_service
        self._max_audio_chunks_per_session = max_audio_chunks_per_session
        self._session_history_limit = session_history_limit

    async def handle_event(self, event: DeviceEvent) -> OrchestrationResult:
        started = time.perf_counter()
        session = await self._resolve_session(event)

        if session is None:
            await self._telemetry.publish(
                TelemetryEvent(
                    event_name="conversation_orchestrator",
                    device_id=event.device_id,
                    session_id=None,
                    event_type=event.event_type,
                    orchestrator_phase="rejected",
                    error_category="session_missing",
                    details={"reason": "session_end_requested_without_active_session"},
                )
            )
            return OrchestrationResult(
                device_id=event.device_id,
                session_id=None,
                ack_message="session_missing",
                error_category="session_missing",
            )

        session.last_event_at = datetime.now(timezone.utc)
        await self._telemetry.publish(
            TelemetryEvent(
                event_name="conversation_orchestrator",
                device_id=session.device_id,
                session_id=session.session_id,
                event_type=event.event_type,
                orchestrator_phase="received",
                details={"correlation_id": event.correlation_id},
            )
        )

        ack_message = f"{event.event_type}_ack"
        response_plan: AIResponsePlan | None = None
        provider_used: str | None = None
        error_category: str | None = None

        try:
            if isinstance(event, HelloEvent):
                self._handle_hello(session, event)
                ack_message = "hello_ack"
            elif isinstance(event, SessionStartEvent):
                if event.trigger is not None:
                    session.metadata["session_trigger"] = event.trigger
                ack_message = "session_started"
            elif isinstance(event, SessionEndEvent):
                session = await self._session_repository.end_session(event.device_id, event.reason) or session
                ack_message = "session_ended"
            elif isinstance(event, HeartbeatEvent):
                if event.sequence is not None:
                    session.metadata["last_heartbeat_sequence"] = str(event.sequence)
                ack_message = "heartbeat_ack"
            elif isinstance(event, StatusEvent):
                session.latest_status = event.status
                ack_message = "status_recorded"
            elif isinstance(event, TouchEvent):
                interpreted_touch = self._touch_interpreter.interpret(event.touch)
                session.last_touch = interpreted_touch
                session.touch_history = (session.touch_history + [interpreted_touch])[-self._session_history_limit :]
                response_plan, provider_used = await self._generate_response(
                    session=session,
                    latest_touch=interpreted_touch,
                    transcript_text=None,
                )
                session.conversation_history = self._append_turn(
                    session.conversation_history,
                    ConversationTurn(role="assistant", text=response_plan.spoken_text),
                )
                ack_message = "touch_processed"
            elif isinstance(event, AudioChunkEvent):
                self._buffer_audio(session, event)
                ack_message = "audio_buffered"
                if event.is_final:
                    transcript = await self._transcribe_audio(session, event)
                    session.conversation_history = self._append_turn(
                        session.conversation_history,
                        ConversationTurn(role="user", text=transcript),
                    )
                    response_plan, provider_used = await self._generate_response(
                        session=session,
                        latest_touch=session.last_touch,
                        transcript_text=transcript,
                    )
                    session.conversation_history = self._append_turn(
                        session.conversation_history,
                        ConversationTurn(role="assistant", text=response_plan.spoken_text),
                    )
                    session.audio_buffer = AudioBufferState()
                    ack_message = "audio_finalized"
        except ResponseValidationError:
            error_category = "response_validation"
            provider_used = self._llm.provider_name
            response_plan = self._fallback_response_service.build(
                touch_interpretation=self._safe_touch_value(session.last_touch)
            )
        except (ProviderUnavailableError, ProviderInvocationError):
            error_category = "provider_failure"
            provider_used = self._llm.provider_name
            response_plan = self._fallback_response_service.build(
                touch_interpretation=self._safe_touch_value(session.last_touch)
            )
        except Exception:
            error_category = "orchestrator_unhandled_error"
            response_plan = self._fallback_response_service.build(
                touch_interpretation=self._safe_touch_value(session.last_touch)
            )

        if isinstance(event, AudioChunkEvent) and event.is_final:
            session.audio_buffer = AudioBufferState()

        await self._session_repository.save(session)
        await self._telemetry.publish(
            TelemetryEvent(
                event_name="conversation_orchestrator",
                device_id=session.device_id,
                session_id=session.session_id,
                event_type=event.event_type,
                orchestrator_phase="completed" if error_category is None else "fallback",
                latency_ms=(time.perf_counter() - started) * 1000,
                provider=provider_used,
                error_category=error_category,
                details={"ack_message": ack_message},
            )
        )

        return OrchestrationResult(
            device_id=session.device_id,
            session_id=session.session_id,
            ack_message=ack_message,
            response_plan=response_plan,
            provider_used=provider_used,
            error_category=error_category,
        )

    async def _resolve_session(self, event: DeviceEvent) -> DeviceSessionContext | None:
        if isinstance(event, SessionStartEvent):
            return await self._session_repository.start_new_session(
                device_id=event.device_id,
                requested_session_id=event.requested_session_id,
            )

        if isinstance(event, SessionEndEvent):
            return await self._session_repository.get_active(event.device_id)

        return await self._session_repository.get_or_create(event.device_id)

    def _handle_hello(self, session: DeviceSessionContext, event: HelloEvent) -> None:
        session.device_capabilities = event.capabilities
        session.metadata["protocol_version"] = event.protocol_version
        if event.firmware_version is not None:
            session.metadata["firmware_version"] = event.firmware_version

    def _buffer_audio(self, session: DeviceSessionContext, event: AudioChunkEvent) -> None:
        buffered_chunks = session.audio_buffer.buffered_chunks + [event.data_base64]
        if len(buffered_chunks) > self._max_audio_chunks_per_session:
            buffered_chunks = buffered_chunks[-self._max_audio_chunks_per_session :]

        session.audio_buffer.buffered_chunks = buffered_chunks
        session.audio_buffer.chunk_count = len(buffered_chunks)
        session.audio_buffer.last_chunk_at = event.sent_at
        session.audio_buffer.encoding = event.encoding.value
        session.audio_buffer.sample_rate_hz = event.sample_rate_hz
        session.audio_buffer.channels = event.channels
        session.audio_buffer.final_chunk_received = event.is_final

    async def _transcribe_audio(
        self,
        session: DeviceSessionContext,
        event: AudioChunkEvent,
    ) -> str:
        result = await self._stt.transcribe(
            TranscriptionRequest(
                device_id=session.device_id,
                session_id=session.session_id,
                audio_chunks=session.audio_buffer.buffered_chunks,
                encoding=event.encoding.value,
                sample_rate_hz=event.sample_rate_hz,
                channels=event.channels,
            )
        )
        return result.text

    async def _generate_response(
        self,
        *,
        session: DeviceSessionContext,
        latest_touch: TouchContext | None,
        transcript_text: str | None,
    ) -> tuple[AIResponsePlan, str]:
        prompt = self._prompt_builder.build(
            session=session,
            latest_touch=latest_touch,
            transcript_text=transcript_text,
        )
        candidate_plan = await self._llm.generate_response(prompt)
        validated_plan = self._response_validator.validate(candidate_plan)

        try:
            await self._tts.plan_synthesis(
                TtsSynthesisRequest(
                    device_id=session.device_id,
                    session_id=session.session_id,
                    text=validated_plan.spoken_text,
                    voice_style=validated_plan.voice_style,
                )
            )
        except (ProviderUnavailableError, ProviderInvocationError):
            await self._telemetry.publish(
                TelemetryEvent(
                    event_name="tts_plan",
                    device_id=session.device_id,
                    session_id=session.session_id,
                    event_type="tts",
                    orchestrator_phase="degraded",
                    provider=self._tts.provider_name,
                    error_category="tts_unavailable",
                )
            )

        return validated_plan, self._llm.provider_name

    def _append_turn(
        self,
        turns: list[ConversationTurn],
        new_turn: ConversationTurn,
    ) -> list[ConversationTurn]:
        return (turns + [new_turn])[-self._session_history_limit :]

    def _safe_touch_value(self, touch: TouchContext | None) -> TouchInterpretation:
        if touch is None:
            return TouchInterpretation.UNKNOWN
        return touch.interpreted_as
