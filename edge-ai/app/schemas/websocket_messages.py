from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.models.device_event import AudioEncoding
from app.domain.models.session_context import DeviceStatusSnapshot
from app.domain.models.touch_context import (
    RawTouchSensor,
    TouchContext,
    TouchGesture,
)


def _map_touch_sensor(sensor_name: str) -> RawTouchSensor:
    normalized = sensor_name.strip().lower()
    if normalized in {"pet", "petting_surface", "pet_touch", "pet_touch_sensor"}:
        return RawTouchSensor.PETTING_SURFACE
    if normalized in {"record", "record_button", "record_touch", "record_touch_sensor"}:
        return RawTouchSensor.RECORD_BUTTON
    return RawTouchSensor.UNKNOWN


def _build_status_note(level: str | None, component: str | None, detail: str) -> str:
    prefix_parts = [part for part in [level, component] if part]
    prefix = "/".join(prefix_parts)
    note = f"{prefix}: {detail}" if prefix else detail
    return note[:200]


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
    uptime_ms: int | None = Field(default=None, ge=0)
    wifi_connected: bool | None = None
    transport_connected: bool | None = None
    active_session: bool | None = None


class TouchEventMessage(MessageBase):
    message_type: Literal["touch_event"] = "touch_event"
    device_id: str = Field(min_length=1, max_length=128)
    touch: TouchContext | None = None
    sensor: str | None = Field(default=None, max_length=32)
    pressed: bool | None = None
    source: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def populate_touch_context(self) -> "TouchEventMessage":
        if self.touch is not None:
            return self

        if self.sensor is None or self.pressed is None:
            raise ValueError("touch_event must include either touch or sensor/pressed fields.")

        self.touch = TouchContext(
            sensor=_map_touch_sensor(self.sensor),
            gesture=TouchGesture.PRESS if self.pressed else TouchGesture.RELEASE,
        )
        return self


class AudioChunkMessage(MessageBase):
    message_type: Literal["audio_chunk"] = "audio_chunk"
    device_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
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
    session_id: str | None = Field(default=None, max_length=128)
    trigger: str | None = Field(default=None, max_length=64)
    encoding: str | None = Field(default=None, max_length=32)
    sample_rate_hz: int | None = Field(default=None, ge=8000, le=96000)
    channels: int | None = Field(default=None, ge=1, le=2)

    @model_validator(mode="after")
    def populate_requested_session_id(self) -> "SessionStartMessage":
        if self.requested_session_id is None and self.session_id is not None:
            self.requested_session_id = self.session_id
        return self


class SessionEndMessage(MessageBase):
    message_type: Literal["session_end"] = "session_end"
    device_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    reason: str | None = Field(default=None, max_length=128)
    elapsed_ms: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=0)
    trigger: str | None = Field(default=None, max_length=64)


class StatusMessage(MessageBase):
    message_type: Literal["status"] = "status"
    device_id: str = Field(min_length=1, max_length=128)
    status: DeviceStatusSnapshot | None = None
    level: str | None = Field(default=None, max_length=16)
    component: str | None = Field(default=None, max_length=64)
    detail: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def populate_status_snapshot(self) -> "StatusMessage":
        if self.status is not None:
            return self

        if self.detail is None:
            raise ValueError("status message must include either status or detail.")

        self.status = DeviceStatusSnapshot(
            note=_build_status_note(self.level, self.component, self.detail),
        )
        return self


class AckMessage(MessageBase):
    message_type: Literal["ack"] = "ack"
    device_id: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    acked_message_type: str = Field(min_length=1, max_length=64)
    ack_for: str = Field(min_length=1, max_length=64)
    accepted: bool = True
    detail: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=128)
    server_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AIResponsePlanMessage(MessageBase):
    message_type: Literal["ai_response_plan"] = "ai_response_plan"
    device_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    response_plan: AIResponsePlan | None = None
    spoken_text: str = Field(min_length=1, max_length=240)
    emotion: Emotion
    face_expression: FaceExpression
    voice_style: VoiceStyle
    should_speak: bool = True
    server_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorMessage(MessageBase):
    message_type: Literal["error"] = "error"
    device_id: str | None = Field(default=None, max_length=128)
    code: str = Field(min_length=1, max_length=128)
    detail: str | None = Field(default=None, max_length=240)
    message: str = Field(min_length=1, max_length=240)
    retryable: bool = False
    server_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AudioOutputMessage(MessageBase):
    message_type: Literal["audio_output"] = "audio_output"
    device_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    encoding: str = Field(min_length=1, max_length=32)
    sample_rate_hz: int = Field(ge=8000, le=96000)
    channels: int = Field(default=1, ge=1, le=2)
    data_base64: str = Field(min_length=1)
    is_final: bool = True
    mime_type: str | None = Field(default=None, max_length=64)
    server_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    Union[AckMessage, AIResponsePlanMessage, ErrorMessage, AudioOutputMessage],
    Field(discriminator="message_type"),
]
