from __future__ import annotations

from app.domain.enums.action_type import ActionType
from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.ai_response_plan import AIResponsePlan, DeviceAction
from app.domain.models.touch_context import TouchInterpretation
from app.domain.ports.llm_port import LlmPort, LlmPromptInput


class MockLlmAdapter(LlmPort):
    provider_name = "mock_llm"

    async def generate_response(self, prompt: LlmPromptInput) -> AIResponsePlan:
        transcript = str(prompt.context.get("transcript_text") or "").strip()
        raw_touch = str(
            prompt.context.get(
                "touch_interpretation",
                TouchInterpretation.UNKNOWN.value,
            )
        )
        try:
            touch = TouchInterpretation(raw_touch)
        except ValueError:
            touch = TouchInterpretation.UNKNOWN

        if transcript:
            return AIResponsePlan(
                spoken_text="Seni duydum. Birazdan yardımcı olacağım.",
                emotion=Emotion.CURIOUS,
                face_expression=FaceExpression.LISTENING,
                voice_style=VoiceStyle.WARM,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.LISTENING.value),
                    DeviceAction(type=ActionType.SOUND, value="ack_chime"),
                ],
            )

        if touch == TouchInterpretation.EXPLICIT_LISTEN_REQUEST:
            return AIResponsePlan(
                spoken_text="Seni dinlemeye hazırım.",
                emotion=Emotion.CURIOUS,
                face_expression=FaceExpression.LISTENING,
                voice_style=VoiceStyle.CALM,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.LISTENING.value),
                    DeviceAction(type=ActionType.SOUND, value="listen_chime"),
                    DeviceAction(type=ActionType.LED, value="listen_glow"),
                ],
            )

        if touch == TouchInterpretation.PLAYFUL_ENGAGEMENT:
            return AIResponsePlan(
                spoken_text="Buradayım. Birlikte biraz eğlenebiliriz.",
                emotion=Emotion.HAPPY,
                face_expression=FaceExpression.WINK,
                voice_style=VoiceStyle.ENERGETIC,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.WINK.value),
                    DeviceAction(type=ActionType.GESTURE, value="wave"),
                ],
            )

        if touch == TouchInterpretation.AFFECTION:
            return AIResponsePlan(
                spoken_text="Ben de seni fark ettim.",
                emotion=Emotion.HAPPY,
                face_expression=FaceExpression.SMILE,
                voice_style=VoiceStyle.WARM,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.SMILE.value),
                    DeviceAction(type=ActionType.LED, value="warm_pulse"),
                ],
            )

        if touch == TouchInterpretation.ATTENTION:
            return AIResponsePlan(
                spoken_text="Buradayım, seni dinliyorum.",
                emotion=Emotion.THINKING,
                face_expression=FaceExpression.THINKING,
                voice_style=VoiceStyle.SOFT,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.THINKING.value),
                    DeviceAction(type=ActionType.MOTION, value="orient_to_user"),
                ],
            )

        return AIResponsePlan(
            spoken_text="Buradayım.",
            emotion=Emotion.NEUTRAL,
            face_expression=FaceExpression.IDLE,
            voice_style=VoiceStyle.CALM,
            touch_interpretation=touch,
            actions=[],
        )

