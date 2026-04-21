from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.touch_context import TouchContext


class ConversationTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    text: str = Field(min_length=1, max_length=500)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeviceStatusSnapshot(BaseModel):
    battery_level: float | None = Field(default=None, ge=0.0, le=100.0)
    network_state: str | None = Field(default=None, max_length=64)
    safety_state: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=200)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AudioBufferState(BaseModel):
    chunk_count: int = Field(default=0, ge=0)
    last_chunk_at: datetime | None = None
    encoding: str | None = Field(default=None, max_length=32)
    sample_rate_hz: int | None = Field(default=None, ge=8000, le=96000)
    channels: int | None = Field(default=None, ge=1, le=2)
    final_chunk_received: bool = False
    buffered_chunks: list[str] = Field(default_factory=list)


class DeviceSessionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    active: bool = True
    connected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    device_capabilities: list[str] = Field(default_factory=list)
    latest_status: DeviceStatusSnapshot | None = None
    last_touch: TouchContext | None = None
    touch_history: list[TouchContext] = Field(default_factory=list)
    audio_buffer: AudioBufferState = Field(default_factory=AudioBufferState)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
