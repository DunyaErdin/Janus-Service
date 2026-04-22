from __future__ import annotations

from typing import Literal

from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.models.touch_context import TouchInterpretation


class FallbackResponseService:
    def build(
        self,
        *,
        touch_interpretation: TouchInterpretation = TouchInterpretation.UNKNOWN,
        reason: Literal["processing_failure", "unclear_input"] = "processing_failure",
    ) -> AIResponsePlan:
        if reason == "unclear_input":
            return AIResponsePlan(
                spoken_text="Seni tam anlayamadım. Bir kez daha söyler misin?",
                emotion=Emotion.LISTENING,
                face_expression=FaceExpression.LISTENING,
                voice_style=VoiceStyle.SOFT,
                touch_interpretation=touch_interpretation,
                actions=[],
            )

        return AIResponsePlan(
            spoken_text="Seni duydum ama şu an küçük bir sorun yaşadım.",
            emotion=Emotion.THINKING,
            face_expression=FaceExpression.THINKING,
            voice_style=VoiceStyle.CALM,
            touch_interpretation=touch_interpretation,
            actions=[],
        )
