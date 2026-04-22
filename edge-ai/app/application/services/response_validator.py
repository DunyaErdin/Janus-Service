from __future__ import annotations

import re

from pydantic import ValidationError

from app.domain.enums.action_type import ActionType
from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.ai_response_plan import AIResponsePlan

_ALLOWED_FACES_BY_EMOTION: dict[Emotion, set[FaceExpression]] = {
    Emotion.NEUTRAL: {FaceExpression.IDLE, FaceExpression.BLINK},
    Emotion.HAPPY: {FaceExpression.SMILE, FaceExpression.HAPPY_EYES, FaceExpression.BLINK},
    Emotion.CURIOUS: {FaceExpression.THINKING, FaceExpression.SURPRISED, FaceExpression.LISTENING},
    Emotion.SLEEPY: {FaceExpression.SLEEPY, FaceExpression.BLINK},
    Emotion.EXCITED: {FaceExpression.SURPRISED, FaceExpression.HAPPY_EYES, FaceExpression.SMILE},
    Emotion.THINKING: {FaceExpression.THINKING, FaceExpression.LISTENING},
    Emotion.SAD: {FaceExpression.SAD_EYES, FaceExpression.IDLE},
    Emotion.AFFECTIONATE: {FaceExpression.SMILE, FaceExpression.HAPPY_EYES},
    Emotion.PLAYFUL: {FaceExpression.WINK, FaceExpression.SMILE, FaceExpression.HAPPY_EYES},
    Emotion.LISTENING: {FaceExpression.LISTENING, FaceExpression.THINKING},
}

_ALLOWED_VOICES_BY_EMOTION: dict[Emotion, set[VoiceStyle]] = {
    Emotion.NEUTRAL: {VoiceStyle.CALM},
    Emotion.HAPPY: {VoiceStyle.WARM, VoiceStyle.CHEERFUL},
    Emotion.CURIOUS: {VoiceStyle.WARM, VoiceStyle.SOFT},
    Emotion.SLEEPY: {VoiceStyle.SLEEPY, VoiceStyle.SOFT},
    Emotion.EXCITED: {VoiceStyle.ENERGETIC, VoiceStyle.CHEERFUL},
    Emotion.THINKING: {VoiceStyle.CALM, VoiceStyle.SERIOUS},
    Emotion.SAD: {VoiceStyle.SOFT, VoiceStyle.CALM},
    Emotion.AFFECTIONATE: {VoiceStyle.WARM, VoiceStyle.SOFT},
    Emotion.PLAYFUL: {VoiceStyle.PLAYFUL, VoiceStyle.CHEERFUL},
    Emotion.LISTENING: {VoiceStyle.SOFT, VoiceStyle.CALM},
}


class ResponseValidationError(ValueError):
    pass


class ResponseValidator:
    _LOW_LEVEL_TOKENS = (
        "gpio",
        "pin ",
        "i2c",
        "spi",
        "pwm",
        "uart",
        "firmware",
        "esp32",
        "register",
    )

    def validate(self, candidate: AIResponsePlan | dict[str, object]) -> AIResponsePlan:
        try:
            plan = AIResponsePlan.model_validate(candidate)
        except ValidationError as exc:
            raise ResponseValidationError("AI response failed schema validation.") from exc

        if any(token in plan.spoken_text.lower() for token in self._LOW_LEVEL_TOKENS):
            raise ResponseValidationError("Spoken text referenced low-level device details.")

        if len(plan.spoken_text.split()) > 40:
            raise ResponseValidationError("Spoken text exceeded the allowed brevity limit.")

        sentence_count = len(
            [sentence for sentence in re.split(r"[.!?]+", plan.spoken_text) if sentence.strip()]
        )
        if sentence_count > 3:
            raise ResponseValidationError("Spoken text exceeded the sentence limit.")

        allowed_faces = _ALLOWED_FACES_BY_EMOTION[plan.emotion]
        if plan.face_expression not in allowed_faces:
            raise ResponseValidationError("Face expression does not align with the selected emotion.")

        allowed_voices = _ALLOWED_VOICES_BY_EMOTION[plan.emotion]
        if plan.voice_style not in allowed_voices:
            raise ResponseValidationError("Voice style does not align with the selected emotion.")

        face_actions = [action for action in plan.actions if action.type == ActionType.FACE]
        if face_actions and all(action.value != plan.face_expression.value for action in face_actions):
            raise ResponseValidationError(
                "Face actions must align with the selected face expression."
            )

        if len({action.type for action in plan.actions}) != len(plan.actions):
            raise ResponseValidationError("Action types must be unique within one response plan.")

        return plan
