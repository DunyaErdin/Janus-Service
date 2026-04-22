from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

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
from app.schemas.websocket_messages import (
    AIResponsePlanMessage,
    AckMessage,
    AudioChunkMessage,
    ErrorMessage,
    HeartbeatMessage,
    HelloMessage,
    IncomingDeviceMessage,
    OutgoingDeviceMessage,
    SessionEndMessage,
    SessionStartMessage,
    StatusMessage,
    TouchEventMessage,
)

_INCOMING_MESSAGE_ADAPTER = TypeAdapter(IncomingDeviceMessage)


class ProtocolDecodeError(ValueError):
    pass


def parse_incoming_message(raw_message: str) -> IncomingDeviceMessage:
    try:
        return _INCOMING_MESSAGE_ADAPTER.validate_json(raw_message)
    except ValidationError as exc:
        raise ProtocolDecodeError("Incoming websocket message failed schema validation.") from exc


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
            chunk_id=message.chunk_id,
            encoding=message.encoding,
            sample_rate_hz=message.sample_rate_hz,
            channels=message.channels,
            data_base64=message.data_base64,
            is_final=message.is_final,
            sent_at=message.sent_at,
        )

    if isinstance(message, SessionStartMessage):
        return SessionStartEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            requested_session_id=message.requested_session_id,
            trigger=message.trigger,
        )

    if isinstance(message, SessionEndMessage):
        return SessionEndEvent(
            device_id=message.device_id,
            correlation_id=message.correlation_id,
            reason=message.reason,
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
        ack_for=ack_for,
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
        message=message,
        retryable=retryable,
    )
