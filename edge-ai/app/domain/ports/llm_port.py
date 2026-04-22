from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.ai_response_plan import AIResponsePlan


class LlmRuntimeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_name: str = Field(min_length=1, max_length=64)
    language: str = Field(min_length=2, max_length=32)
    interaction_mode: Literal["idle", "listening", "replying"]
    device_state: dict[str, Any] = Field(default_factory=dict)
    touch_context: dict[str, Any] = Field(default_factory=dict)
    conversation_summary: str = Field(default="", max_length=1200)
    latest_user_utterance: str | None = Field(default=None, max_length=500)


class LlmPromptExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_summary: str = Field(min_length=1)
    output_payload: dict[str, Any] = Field(default_factory=dict)


class LlmPromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str
    session_id: str
    runtime_context: LlmRuntimeContext
    system_prompt: str = Field(min_length=1)
    developer_prompt: str = Field(min_length=1)
    dynamic_context: str = Field(min_length=1)
    output_instruction: str = Field(min_length=1)
    response_schema: dict[str, Any] = Field(default_factory=dict)
    few_shot_examples: list[LlmPromptExample] = Field(default_factory=list)

    def render_system_instruction(self) -> str:
        return "\n\n".join(
            [
                self.system_prompt.strip(),
                self.developer_prompt.strip(),
            ]
        ).strip()

    def render_user_prompt(self) -> str:
        blocks = [
            self.dynamic_context.strip(),
            self.output_instruction.strip(),
        ]

        if self.few_shot_examples:
            formatted_examples = []
            for index, example in enumerate(self.few_shot_examples, start=1):
                formatted_examples.append(
                    "\n".join(
                        [
                            f"Example {index} Input:",
                            example.input_summary.strip(),
                            f"Example {index} Output:",
                            json.dumps(example.output_payload, ensure_ascii=False),
                        ]
                    )
                )
            blocks.append("Reference Examples:\n" + "\n\n".join(formatted_examples))

        return "\n\n".join(blocks).strip()


class LlmPort(ABC):
    provider_name: str

    @abstractmethod
    async def generate_response(self, prompt: LlmPromptInput) -> AIResponsePlan:
        raise NotImplementedError
