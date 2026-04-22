from __future__ import annotations

import json
from textwrap import dedent

from app.application.prompts.developer_prompt import build_developer_prompt
from app.application.prompts.output_contract import (
    get_few_shot_examples,
    get_strict_output_instruction,
)
from app.application.prompts.system_prompt import build_system_prompt
from app.domain.models.session_context import DeviceStatusSnapshot
from app.domain.models.touch_context import TouchContext, TouchInterpretation
from app.domain.ports.llm_port import (
    LlmPromptExample,
    LlmPromptInput,
    LlmRuntimeContext,
)
from app.schemas.llm_response_schema import get_robot_structured_response_json_schema


class PromptBuilder:
    def __init__(self, *, robot_name: str, default_language: str) -> None:
        self._robot_name = robot_name
        self._default_language = default_language

    def build(
        self,
        *,
        device_id: str,
        session_id: str,
        language: str | None,
        interaction_mode: str,
        device_state: DeviceStatusSnapshot | None,
        touch_context: TouchContext | None,
        conversation_summary: str,
        latest_user_utterance: str | None,
    ) -> LlmPromptInput:
        effective_language = language or self._default_language
        normalized_interaction_mode = interaction_mode if interaction_mode in {
            "idle",
            "listening",
            "replying",
        } else "replying"
        touch_payload = self._serialize_touch_context(touch_context)
        device_state_payload = device_state.model_dump(mode="json") if device_state else {
            "status": "unknown"
        }
        conversation_summary_text = conversation_summary.strip() or "Yeni oturum, önceki konuşma özeti yok."
        latest_utterance = latest_user_utterance.strip() if latest_user_utterance else ""

        dynamic_context = dedent(
            f"""
            Robot Name: {self._robot_name}
            Language: {effective_language}
            Interaction Mode: {normalized_interaction_mode}
            Current Device State: {json.dumps(device_state_payload, ensure_ascii=False)}
            Recent Touch Context: {json.dumps(touch_payload, ensure_ascii=False)}
            Recent Conversation Summary: {conversation_summary_text}
            Latest User Utterance: {latest_utterance or "<empty>"}
            Response Constraints: speak naturally, short, warm, valid JSON only

            Interpretation Notes:
            - If Latest User Utterance is empty and Recent Touch Context is meaningful, respond to the touch or social cue.
            - If Recent Conversation Summary is empty, treat this as a fresh turn.
            - If Interaction Mode is listening, prefer listening-compatible emotion, face_expression, and voice_style.
            - If Current Device State suggests quiet or rest mode, prefer calmer wording and calmer styles.
            - If context is weak or ambiguous, prefer a short clarification response over guessing.
            """
        ).strip()

        return LlmPromptInput(
            device_id=device_id,
            session_id=session_id,
            runtime_context=LlmRuntimeContext(
                robot_name=self._robot_name,
                language=effective_language,
                interaction_mode=normalized_interaction_mode,
                device_state=device_state_payload,
                touch_context=touch_payload,
                conversation_summary=conversation_summary_text,
                latest_user_utterance=latest_utterance or None,
            ),
            system_prompt=build_system_prompt(
                robot_name=self._robot_name,
                language=effective_language,
            ),
            developer_prompt=build_developer_prompt(
                default_language=self._default_language,
            ),
            dynamic_context=dynamic_context,
            output_instruction=get_strict_output_instruction(),
            response_schema=get_robot_structured_response_json_schema(),
            few_shot_examples=[
                LlmPromptExample.model_validate(example)
                for example in get_few_shot_examples()
            ],
        )

    def _serialize_touch_context(
        self,
        touch_context: TouchContext | None,
    ) -> dict[str, object]:
        if touch_context is None:
            return {
                "present": False,
                "semantic_label": TouchInterpretation.NONE.value,
            }

        return {
            "present": True,
            "sensor": touch_context.sensor.value,
            "gesture": touch_context.gesture.value,
            "duration_ms": touch_context.duration_ms,
            "repeat_count": touch_context.repeat_count,
            "intensity": touch_context.intensity,
            "semantic_label": touch_context.interpreted_as.value,
            "occurred_at": touch_context.occurred_at.isoformat(),
        }
