import pytest

from app.application.services.response_validator import (
    ResponseValidationError,
    ResponseValidator,
)
from app.domain.enums.action_type import ActionType
from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.ai_response_plan import AIResponsePlan, DeviceAction
from app.domain.models.touch_context import TouchInterpretation


def test_response_validator_accepts_valid_plan() -> None:
    validator = ResponseValidator()
    plan = AIResponsePlan(
        spoken_text="Merhaba, seni duydum.",
        emotion=Emotion.HAPPY,
        face_expression=FaceExpression.SMILE,
        voice_style=VoiceStyle.WARM,
        touch_interpretation=TouchInterpretation.AFFECTION,
        actions=[DeviceAction(type=ActionType.FACE, value=FaceExpression.SMILE.value)],
    )

    validated = validator.validate(plan)

    assert validated.face_expression == FaceExpression.SMILE


def test_response_validator_rejects_invalid_action_value() -> None:
    validator = ResponseValidator()

    with pytest.raises(ResponseValidationError):
        validator.validate(
            {
                "spoken_text": "Merhaba.",
                "emotion": "happy",
                "face_expression": "smile",
                "voice_style": "warm",
                "touch_interpretation": "affection",
                "actions": [{"type": "gesture", "value": "dance"}],
            }
        )


def test_response_validator_rejects_low_level_text() -> None:
    validator = ResponseValidator()

    with pytest.raises(ResponseValidationError):
        validator.validate(
            {
                "spoken_text": "GPIO 12 pinini etkinleştiriyorum.",
                "emotion": "neutral",
                "face_expression": "idle",
                "voice_style": "serious",
                "touch_interpretation": "unknown",
                "actions": [],
            }
        )

