from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.domain.models.ai_response_plan import AIResponsePlan


class LlmPromptInput(BaseModel):
    device_id: str
    session_id: str
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)


class LlmPort(ABC):
    provider_name: str

    @abstractmethod
    async def generate_response(self, prompt: LlmPromptInput) -> AIResponsePlan:
        raise NotImplementedError

