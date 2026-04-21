from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.models.device_event import AudioEncoding
from app.domain.models.session_context import DeviceStatusSnapshot
from app.domain.models.touch_context import TouchContext


class MessageBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str | None = Field(default=None, max_length=128)


class HelloMessage(MessageBase):
    message_type: Literal["hello"] = "hello"
    device_id: str = Field(min_length=1, max_length=128)
    protocol_version: str = Field(default="1.0", max_length=32)
    firmware_version: str | None = Field(default=None, max_length=64)
    capabilities: list[str] = Field(default_factory=list)


class HeartbeatMessage(MessageBase):
    message_type: Literal["heartbeat"] = "heartbeat"
    device_id: str = Field(min_length=1, max_length=128)
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int | None = Field(default=None, ge=0)


class TouchEventMessage(MessageBase):
    message_type: Literal["touch_event"] = "touch_event"
    device_id: str = Field(min_length=1, max_length=128)
    touch: TouchContext


class AudioChunkMessage(MessageBase):
    message_type: Literal["audio_chunk"] = "audio_chunk"
    device_id: str = Field(min_length=1, max_length=128)
    chunk_id: int = Field(ge=0)
    encoding: AudioEncoding
    sample_rate_hz: int = Field(ge=8000, le=96000)
    channels: int = Field(default=1, ge=1, le=2)
    data_base64: str = Field(min_length=1)
    is_final: bool = False
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStartMessage(MessageBase):
    message_type: Literal["session_start"] = "session_start"
    device_id: str = Field(min_length=1, max_length=128)
    requested_session_id: str | None = Field(default=None, max_length=128)
    trigger: str | None = Field(default=None, max_length=64)


class SessionEndMessage(MessageBase):
    message_type: Literal["session_end"] = "session_end"
    device_id: str = Field(min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=128)


class StatusMessage(MessageBase):
    message_type: Literal["status"] = "status"
    device_id: str = Field(min_length=1, max_length=128)
    status: DeviceStatusSnapshot


class AckMessage(MessageBase):
    message_type: Literal["ack"] = "ack"
    device_id: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    message: str = Field(min_length=1, max_length=128)


class AIResponsePlanMessage(MessageBase):
    message_type: Literal["ai_response_plan"] = "ai_response_plan"
    device_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    response_plan: AIResponsePlan


class ErrorMessage(MessageBase):
    message_type: Literal["error"] = "error"
    device_id: str | None = Field(default=None, max_length=128)
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=240)
    retryable: bool = False


IncomingDeviceMessage = Annotated[
    Union[
        HelloMessage,
        HeartbeatMessage,
        TouchEventMessage,
        AudioChunkMessage,
        SessionStartMessage,
        SessionEndMessage,
        StatusMessage,
    ],
    Field(discriminator="message_type"),
]

OutgoingDeviceMessage = Annotated[
    Union[AckMessage, AIResponsePlanMessage, ErrorMessage],
    Field(discriminator="message_type"),
]

