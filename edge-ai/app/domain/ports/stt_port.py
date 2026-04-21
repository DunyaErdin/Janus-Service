from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    device_id: str
    session_id: str
    audio_chunks: list[str] = Field(default_factory=list, min_length=1)
    encoding: str = Field(min_length=1, max_length=32)
    sample_rate_hz: int = Field(ge=8000, le=96000)
    channels: int = Field(ge=1, le=2)


class TranscriptionResult(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    language: str | None = Field(default=None, max_length=32)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SttPort(ABC):
    provider_name: str

    @abstractmethod
    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise NotImplementedError

