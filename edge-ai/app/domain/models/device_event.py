from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.models.session_context import DeviceStatusSnapshot
from app.domain.models.touch_context import TouchContext


class AudioEncoding(str, Enum):
    PCM16 = "pcm16"
    OPUS = "opus"


class BaseDeviceEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    device_id: str = Field(min_length=1, max_length=128)
    correlation_id: str | None = Field(default=None, max_length=128)
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HelloEvent(BaseDeviceEvent):
    event_type: Literal["hello"] = "hello"
    protocol_version: str = Field(default="1.0", max_length=32)
    firmware_version: str | None = Field(default=None, max_length=64)
    capabilities: list[str] = Field(default_factory=list)


class HeartbeatEvent(BaseDeviceEvent):
    event_type: Literal["heartbeat"] = "heartbeat"
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int | None = Field(default=None, ge=0)
    uptime_ms: int | None = Field(default=None, ge=0)
    wifi_connected: bool | None = None
    transport_connected: bool | None = None
    active_session: bool | None = None


class TouchEvent(BaseDeviceEvent):
    event_type: Literal["touch_event"] = "touch_event"
    touch: TouchContext


class AudioChunkEvent(BaseDeviceEvent):
    event_type: Literal["audio_chunk"] = "audio_chunk"
    session_id: str | None = Field(default=None, max_length=128)
    chunk_id: int = Field(ge=0)
    encoding: AudioEncoding
    sample_rate_hz: int = Field(ge=8000, le=96000)
    channels: int = Field(default=1, ge=1, le=2)
    data_base64: str = Field(min_length=1)
    is_final: bool = False
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionStartEvent(BaseDeviceEvent):
    event_type: Literal["session_start"] = "session_start"
    requested_session_id: str | None = Field(default=None, max_length=128)
    trigger: str | None = Field(default=None, max_length=64)
    encoding: str | None = Field(default=None, max_length=32)
    sample_rate_hz: int | None = Field(default=None, ge=8000, le=96000)
    channels: int | None = Field(default=None, ge=1, le=2)


class SessionEndEvent(BaseDeviceEvent):
    event_type: Literal["session_end"] = "session_end"
    session_id: str | None = Field(default=None, max_length=128)
    reason: str | None = Field(default=None, max_length=128)
    elapsed_ms: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=0)
    trigger: str | None = Field(default=None, max_length=64)


class StatusEvent(BaseDeviceEvent):
    event_type: Literal["status"] = "status"
    status: DeviceStatusSnapshot


DeviceEvent = Annotated[
    Union[
        HelloEvent,
        HeartbeatEvent,
        TouchEvent,
        AudioChunkEvent,
        SessionStartEvent,
        SessionEndEvent,
        StatusEvent,
    ],
    Field(discriminator="event_type"),
]
