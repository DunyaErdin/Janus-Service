from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class WakeDetectionRequest(BaseModel):
    device_id: str
    interaction_id: str
    audio_chunks: list[str] = Field(default_factory=list, min_length=1)
    encoding: str = Field(min_length=1, max_length=32)
    sample_rate_hz: int = Field(ge=8000, le=96000)
    channels: int = Field(ge=1, le=2)


class WakeDetectionResult(BaseModel):
    detected: bool
    transcript: str | None = Field(default=None, max_length=240)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(default="not_wake_word", max_length=128)


class WakeDetectionService(ABC):
    provider_name: str

    @abstractmethod
    async def detect(self, request: WakeDetectionRequest) -> WakeDetectionResult:
        raise NotImplementedError
