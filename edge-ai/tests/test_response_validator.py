from app.application.services.response_validator import ResponseValidator
from app.domain.enums.action_type import ActionType
from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.ai_response_plan import AIResponsePlan, DeviceAction
from app.domain.models.touch_context import TouchInterpretation


def test_response_validator_normalizes_face_voice_and_face_action() -> None:
    validator = ResponseValidator()

    normalized = validator.validate(
        AIResponsePlan(
            spoken_text="Merhaba, ben buradayim.",
            emotion=Emotion.HAPPY,
            face_expression=FaceExpression.THINKING,
            voice_style=VoiceStyle.SERIOUS,
            touch_interpretation=TouchInterpretation.NONE,
            actions=[
                DeviceAction(type=ActionType.FACE, value=FaceExpression.THINKING.value),
            ],
        )
    )

    assert normalized.face_expression == FaceExpression.SMILE
    assert normalized.voice_style == VoiceStyle.WARM
    assert normalized.actions[0].value == FaceExpression.SMILE.value
