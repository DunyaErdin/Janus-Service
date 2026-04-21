from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.enums.voice_style import VoiceStyle


class TtsSynthesisRequest(BaseModel):
    device_id: str
    session_id: str
    text: str = Field(min_length=1, max_length=240)
    voice_style: VoiceStyle


class TtsSynthesisPlan(BaseModel):
    provider: str
    status: Literal["planned", "skipped", "unavailable"]
    reference: str | None = Field(default=None, max_length=256)


class TtsPort(ABC):
    provider_name: str

    @abstractmethod
    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        raise NotImplementedError

