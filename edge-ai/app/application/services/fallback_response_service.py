from __future__ import annotations

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
    ) -> AIResponsePlan:
        return AIResponsePlan(
            spoken_text="Şu anda seni anladım ama cevap oluştururken bir sorun yaşadım.",
            emotion=Emotion.NEUTRAL,
            face_expression=FaceExpression.THINKING,
            voice_style=VoiceStyle.CALM,
            touch_interpretation=touch_interpretation,
            actions=[],
        )

