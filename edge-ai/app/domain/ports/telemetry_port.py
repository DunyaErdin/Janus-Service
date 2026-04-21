from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TelemetryEvent(BaseModel):
    event_name: str = Field(min_length=1, max_length=64)
    device_id: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    event_type: str | None = Field(default=None, max_length=64)
    orchestrator_phase: str | None = Field(default=None, max_length=64)
    latency_ms: float | None = Field(default=None, ge=0.0)
    provider: str | None = Field(default=None, max_length=64)
    error_category: str | None = Field(default=None, max_length=64)
    details: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetryPort(ABC):
    @abstractmethod
    async def publish(self, event: TelemetryEvent) -> None:
        raise NotImplementedError

