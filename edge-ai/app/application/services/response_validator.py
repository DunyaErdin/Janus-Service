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

_DEFAULT_FACE_BY_EMOTION: dict[Emotion, FaceExpression] = {
    Emotion.NEUTRAL: FaceExpression.IDLE,
    Emotion.HAPPY: FaceExpression.SMILE,
    Emotion.CURIOUS: FaceExpression.LISTENING,
    Emotion.SLEEPY: FaceExpression.SLEEPY,
    Emotion.EXCITED: FaceExpression.HAPPY_EYES,
    Emotion.THINKING: FaceExpression.THINKING,
    Emotion.SAD: FaceExpression.SAD_EYES,
    Emotion.AFFECTIONATE: FaceExpression.SMILE,
    Emotion.PLAYFUL: FaceExpression.WINK,
    Emotion.LISTENING: FaceExpression.LISTENING,
}

_DEFAULT_VOICE_BY_EMOTION: dict[Emotion, VoiceStyle] = {
    Emotion.NEUTRAL: VoiceStyle.CALM,
    Emotion.HAPPY: VoiceStyle.WARM,
    Emotion.CURIOUS: VoiceStyle.WARM,
    Emotion.SLEEPY: VoiceStyle.SLEEPY,
    Emotion.EXCITED: VoiceStyle.ENERGETIC,
    Emotion.THINKING: VoiceStyle.CALM,
    Emotion.SAD: VoiceStyle.SOFT,
    Emotion.AFFECTIONATE: VoiceStyle.WARM,
    Emotion.PLAYFUL: VoiceStyle.PLAYFUL,
    Emotion.LISTENING: VoiceStyle.SOFT,
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
            plan = plan.model_copy(
                update={"face_expression": _DEFAULT_FACE_BY_EMOTION[plan.emotion]}
            )

        allowed_voices = _ALLOWED_VOICES_BY_EMOTION[plan.emotion]
        if plan.voice_style not in allowed_voices:
            plan = plan.model_copy(
                update={"voice_style": _DEFAULT_VOICE_BY_EMOTION[plan.emotion]}
            )

        normalized_actions = []
        seen_action_types: set[ActionType] = set()
        for action in plan.actions:
            if action.type in seen_action_types:
                continue
            seen_action_types.add(action.type)
            if action.type == ActionType.FACE and action.value != plan.face_expression.value:
                normalized_actions.append(
                    action.model_copy(update={"value": plan.face_expression.value})
                )
                continue
            normalized_actions.append(action)

        if normalized_actions != plan.actions:
            plan = plan.model_copy(update={"actions": normalized_actions})

        return plan
