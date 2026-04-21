from __future__ import annotations

import json
from textwrap import dedent

from app.domain.enums.action_type import ActionType
from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.session_context import DeviceSessionContext
from app.domain.models.touch_context import TouchContext, TouchInterpretation
from app.domain.ports.llm_port import LlmPromptInput
from app.schemas.llm_response_schema import LlmStructuredResponse


class PromptBuilder:
    def __init__(self, *, robot_name: str, default_language: str) -> None:
        self._robot_name = robot_name
        self._default_language = default_language

    def build(
        self,
        *,
        session: DeviceSessionContext,
        latest_touch: TouchContext | None,
        transcript_text: str | None,
    ) -> LlmPromptInput:
        recent_turns = session.conversation_history[-4:]
        history_summary = [
            {"role": turn.role, "text": turn.text}
            for turn in recent_turns
        ]
        touch_summary = {
            "sensor": latest_touch.sensor.value if latest_touch else "unknown",
            "gesture": latest_touch.gesture.value if latest_touch else "unknown",
            "duration_ms": latest_touch.duration_ms if latest_touch else None,
            "repeat_count": latest_touch.repeat_count if latest_touch else None,
            "interpreted_as": latest_touch.interpreted_as.value if latest_touch else "unknown",
        }

        system_prompt = dedent(
            f"""
            You are {self._robot_name}, a local edge AI assistant for a home robot.
            You do not control hardware directly.
            You must return only structured semantic response plans.

            Safety rules:
            - Never emit GPIO, PWM, pin, bus, or register-level instructions.
            - Never emit transport-level commands or firmware implementation details.
            - Keep spoken text brief, warm, and suitable for a home assistant robot.
            - If context is limited, stay concise and safe.
            - Prefer semantic actions over literal device instructions.

            Allowed emotion values: {", ".join(item.value for item in Emotion)}
            Allowed face_expression values: {", ".join(item.value for item in FaceExpression)}
            Allowed voice_style values: {", ".join(item.value for item in VoiceStyle)}
            Allowed touch_interpretation values: {", ".join(item.value for item in TouchInterpretation)}
            Allowed action.type values: {", ".join(item.value for item in ActionType)}

            Output language: {self._default_language}
            Return JSON only, matching this schema exactly:
            {json.dumps(LlmStructuredResponse.model_json_schema(), ensure_ascii=False)}
            """
        ).strip()

        user_prompt = dedent(
            f"""
            Session context:
            {json.dumps(
                {
                    "device_id": session.device_id,
                    "session_id": session.session_id,
                    "device_capabilities": session.device_capabilities,
                    "latest_status": session.latest_status.model_dump(mode="json") if session.latest_status else None,
                    "touch_context": touch_summary,
                    "recent_conversation": history_summary,
                    "latest_transcript": transcript_text,
                },
                ensure_ascii=False,
            )}

            Produce a single safe response plan for the next robot reaction.
            """
        ).strip()

        return LlmPromptInput(
            device_id=session.device_id,
            session_id=session.session_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context={
                "touch_interpretation": touch_summary["interpreted_as"],
                "transcript_text": transcript_text,
                "device_capabilities": session.device_capabilities,
                "history_summary": history_summary,
            },
        )
