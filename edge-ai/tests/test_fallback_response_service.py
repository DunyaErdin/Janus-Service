from app.application.services.fallback_response_service import FallbackResponseService
from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.touch_context import TouchInterpretation


def test_fallback_response_is_safe_and_structured() -> None:
    service = FallbackResponseService()

    plan = service.build(
        touch_interpretation=TouchInterpretation.EXPLICIT_LISTEN_REQUEST
    )

    assert plan.spoken_text == "Seni duydum ama su an kucuk bir sorun yasadim."
    assert plan.emotion == Emotion.THINKING
    assert plan.face_expression == FaceExpression.THINKING
    assert plan.voice_style == VoiceStyle.CALM
    assert plan.touch_interpretation == TouchInterpretation.EXPLICIT_LISTEN_REQUEST
    assert plan.actions == []
