from __future__ import annotations

from pydantic import ValidationError

from app.domain.enums.action_type import ActionType
from app.domain.models.ai_response_plan import AIResponsePlan


class ResponseValidationError(ValueError):
    pass


class ResponseValidator:
    _LOW_LEVEL_TOKENS = ("gpio", "pin ", "i2c", "spi", "pwm", "uart")

    def validate(self, candidate: AIResponsePlan | dict[str, object]) -> AIResponsePlan:
        try:
            plan = AIResponsePlan.model_validate(candidate)
        except ValidationError as exc:
            raise ResponseValidationError("AI response failed schema validation.") from exc

        if any(token in plan.spoken_text.lower() for token in self._LOW_LEVEL_TOKENS):
            raise ResponseValidationError("Spoken text referenced low-level device details.")

        if len(plan.spoken_text.split()) > 40:
            raise ResponseValidationError("Spoken text exceeded the allowed brevity limit.")

        face_actions = [action for action in plan.actions if action.type == ActionType.FACE]
        if face_actions and all(action.value != plan.face_expression.value for action in face_actions):
            raise ResponseValidationError(
                "Face actions must align with the selected face expression."
            )

        return plan

