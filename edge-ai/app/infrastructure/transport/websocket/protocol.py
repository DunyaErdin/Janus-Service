from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.models.device_event import (
    AudioChunkEvent,
    DeviceEvent,
    GreetingRequestEvent,
    HeartbeatEvent,
    HelloEvent,
    SessionEndEvent,
    SessionStartEvent,
    StatusEvent,
    TouchEvent,
    WakeAudioChunkEvent,
    WakeListeningStartedEvent,
)
from app.schemas.websocket_messages import (
    AIResponsePlanMessage,
    AckMessage,
    AudioOutputMessage,
    AudioOutputChunkMessage,
    AudioOutputEndMessage,
    AudioChunkMessage,
    ErrorMessage,
    GreetingRequestMessage,
    HeartbeatMessage,
    HelloMessage,
    IncomingDeviceMessage,
    OutgoingDeviceMessage,
    SessionEndMessage,
    SessionStartMessage,
    StatusMessage,
    TouchEventMessage,
    WakeAudioChunkMessage,
    WakeDetectedMessage,
    WakeListeningStartedMessage,
    WakeRejectedMessage,
)

_INCOMING_MESSAGE_ADAPTER = TypeAdapter(IncomingDeviceMessage)
_AUDIO_OUTPUT_CHUNK_BASE64_CHARS = 2048


class ProtocolDecodeError(ValueError):
    pass


def parse_incoming_message(raw_message: str) -> IncomingDeviceMessage:
    try:
        return _INCOMING_MESSAGE_ADAPTER.validate_json(raw_message)
    except ValidationError as exc:
        raise ProtocolDecodeError(
            "Incoming websocket message failed schema validation."
        ) from exc


def serialize_outgoing_message(message: OutgoingDeviceMessage) -> str:
    return message.model_dump_json(exclude_none=True)


def to_domain_event(message: IncomingDeviceMessage) -> DeviceEvent:
    if isinstance(message, HelloMessage):
        return HelloEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            protocol_version=message.protocol_version,
            firmware_version=message.firmware_version,
            capabilities=message.capabilities,
        )

    if isinstance(message, HeartbeatMessage):
        return HeartbeatEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            sent_at=message.sent_at,
            sequence=message.sequence,
            uptime_ms=message.uptime_ms,
            wifi_connected=message.wifi_connected,
            transport_connected=message.transport_connected,
            active_session=message.active_session,
        )

    if isinstance(message, TouchEventMessage):
        return TouchEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            touch=message.touch,
        )

    if isinstance(message, AudioChunkMessage):
        return AudioChunkEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            chunk_id=message.chunk_id,
            encoding=message.encoding,
            sample_rate_hz=message.sample_rate_hz,
            channels=message.channels,
            data_base64=message.data_base64,
            is_final=message.is_final,
            sent_at=message.sent_at,
        )

    if isinstance(message, WakeListeningStartedMessage):
        return WakeListeningStartedEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            interaction_id=message.interaction_id,
            encoding=message.encoding,
            sample_rate_hz=message.sample_rate_hz,
            channels=message.channels,
            window_ms=message.window_ms,
            prefilter=message.prefilter,
        )

    if isinstance(message, WakeAudioChunkMessage):
        return WakeAudioChunkEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            interaction_id=message.interaction_id,
            chunk_id=message.chunk_id,
            encoding=message.encoding,
            sample_rate_hz=message.sample_rate_hz,
            channels=message.channels,
            data_base64=message.data_base64,
            is_final=message.is_final,
            rms=message.rms,
            peak_abs=message.peak_abs,
            sent_at=message.sent_at,
        )

    if isinstance(message, GreetingRequestMessage):
        return GreetingRequestEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            interaction_id=message.interaction_id,
            text=message.text,
            encoding=message.encoding,
            sample_rate_hz=message.sample_rate_hz,
            channels=message.channels,
        )

    if isinstance(message, SessionStartMessage):
        return SessionStartEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            requested_session_id=message.requested_session_id,
            trigger=message.trigger,
            encoding=message.encoding,
            sample_rate_hz=message.sample_rate_hz,
            channels=message.channels,
        )

    if isinstance(message, SessionEndMessage):
        return SessionEndEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            session_id=message.session_id,
            reason=message.reason,
            elapsed_ms=message.elapsed_ms,
            chunk_count=message.chunk_count,
            trigger=message.trigger,
        )

    if isinstance(message, StatusMessage):
        return StatusEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            status=message.status,
        )

    raise ProtocolDecodeError("Unsupported websocket message type.")


def build_ack_message(
    *,
    device_id: str | None,
    session_id: str | None,
    correlation_id: str | None,
    ack_for: str,
    message: str,
) -> AckMessage:
    return AckMessage(
        device_id=device_id,
        session_id=session_id,
        correlation_id=correlation_id,
        acked_message_type=ack_for,
        ack_for=ack_for,
        detail=message,
        message=message,
    )


def build_ai_response_message(
    *,
    device_id: str,
    session_id: str,
    correlation_id: str | None,
    response_plan: AIResponsePlan,
) -> AIResponsePlanMessage:
    return AIResponsePlanMessage(
        device_id=device_id,
        session_id=session_id,
        correlation_id=correlation_id,
        response_plan=response_plan,
        spoken_text=response_plan.spoken_text,
        emotion=response_plan.emotion,
        face_expression=response_plan.face_expression,
        voice_style=response_plan.voice_style,
        should_speak=True,
    )


def build_error_message(
    *,
    code: str,
    message: str,
    retryable: bool,
    device_id: str | None = None,
    correlation_id: str | None = None,
) -> ErrorMessage:
    return ErrorMessage(
        device_id=device_id,
        correlation_id=correlation_id,
        code=code,
        detail=message,
        message=message,
        retryable=retryable,
    )


def build_audio_output_message(
    *,
    device_id: str,
    session_id: str,
    correlation_id: str | None,
    encoding: str,
    sample_rate_hz: int,
    channels: int,
    data_base64: str,
    mime_type: str | None,
) -> AudioOutputMessage:
    return AudioOutputMessage(
        device_id=device_id,
        session_id=session_id,
        correlation_id=correlation_id,
        encoding=encoding,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        data_base64=data_base64,
        mime_type=mime_type,
    )


def build_audio_output_chunk_messages(
    *,
    device_id: str,
    session_id: str,
    correlation_id: str | None,
    encoding: str,
    sample_rate_hz: int,
    channels: int,
    data_base64: str,
    mime_type: str | None,
) -> list[AudioOutputChunkMessage]:
    normalized_audio = "".join(data_base64.split())
    chunk_width = _AUDIO_OUTPUT_CHUNK_BASE64_CHARS
    chunk_width -= chunk_width % 4
    if chunk_width <= 0:
        raise ValueError("audio output chunk width must be a positive base64 multiple.")

    chunks = [
        normalized_audio[start : start + chunk_width]
        for start in range(0, len(normalized_audio), chunk_width)
        if normalized_audio[start : start + chunk_width]
    ]

    return [
        AudioOutputChunkMessage(
            device_id=device_id,
            session_id=session_id,
            interaction_id=session_id if session_id.startswith("wake-") else None,
            correlation_id=correlation_id,
            chunk_id=index,
            encoding=encoding,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            data_base64=chunk,
            is_final=index == len(chunks) - 1,
            mime_type=mime_type,
        )
        for index, chunk in enumerate(chunks)
    ]


def build_wake_detected_message(
    *,
    device_id: str,
    interaction_id: str,
    correlation_id: str | None,
    transcript: str | None,
    confidence: float | None,
) -> WakeDetectedMessage:
    return WakeDetectedMessage(
        device_id=device_id,
        interaction_id=interaction_id,
        correlation_id=correlation_id,
        transcript=transcript,
        confidence=confidence,
    )


def build_wake_rejected_message(
    *,
    device_id: str,
    interaction_id: str,
    correlation_id: str | None,
    reason: str,
) -> WakeRejectedMessage:
    return WakeRejectedMessage(
        device_id=device_id,
        interaction_id=interaction_id,
        correlation_id=correlation_id,
        reason=reason,
    )


def build_audio_output_end_message(
    *,
    device_id: str,
    session_id: str | None,
    interaction_id: str | None,
    correlation_id: str | None,
    reason: str = "completed",
) -> AudioOutputEndMessage:
    return AudioOutputEndMessage(
        device_id=device_id,
        session_id=session_id,
        interaction_id=interaction_id,
        correlation_id=correlation_id,
        reason=reason,
    )
    (GreetingRequestMessage,)
