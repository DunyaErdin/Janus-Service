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
        transcript = str(prompt.runtime_context.latest_user_utterance or "").strip()
        raw_touch = str(
            prompt.runtime_context.touch_context.get(
                "semantic_label",
                TouchInterpretation.NONE.value,
            )
        )
        try:
            touch = TouchInterpretation(raw_touch)
        except ValueError:
            touch = TouchInterpretation.UNKNOWN

        if transcript:
            return AIResponsePlan(
                spoken_text="Seni duydum. Yardımcı olayım.",
                emotion=Emotion.CURIOUS,
                face_expression=FaceExpression.LISTENING,
                voice_style=VoiceStyle.WARM,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.LISTENING.value),
                    DeviceAction(type=ActionType.SOUND, value="soft_ack"),
                ],
            )

        if touch == TouchInterpretation.EXPLICIT_LISTEN_REQUEST:
            return AIResponsePlan(
                spoken_text="Hazırım, seni dinliyorum.",
                emotion=Emotion.LISTENING,
                face_expression=FaceExpression.LISTENING,
                voice_style=VoiceStyle.SOFT,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.SOUND, value="listen_beep"),
                ],
            )

        if touch == TouchInterpretation.AFFECTION:
            return AIResponsePlan(
                spoken_text="Canım, buradayım. Seni sevgiyle dinliyorum.",
                emotion=Emotion.AFFECTIONATE,
                face_expression=FaceExpression.SMILE,
                voice_style=VoiceStyle.WARM,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.SMILE.value),
                ],
            )

        if touch == TouchInterpretation.PETTING:
            return AIResponsePlan(
                spoken_text="Buradayım. Bu nazik dokunuşunu fark ettim.",
                emotion=Emotion.AFFECTIONATE,
                face_expression=FaceExpression.HAPPY_EYES,
                voice_style=VoiceStyle.SOFT,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.FACE, value=FaceExpression.HAPPY_EYES.value),
                ],
            )

        if touch == TouchInterpretation.ATTENTION_REQUEST:
            return AIResponsePlan(
                spoken_text="Buradayım, seni dikkatle dinliyorum.",
                emotion=Emotion.LISTENING,
                face_expression=FaceExpression.LISTENING,
                voice_style=VoiceStyle.SOFT,
                touch_interpretation=touch,
                actions=[
                    DeviceAction(type=ActionType.GESTURE, value="nod"),
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
