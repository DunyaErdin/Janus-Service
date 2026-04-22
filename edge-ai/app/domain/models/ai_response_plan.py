from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.domain.enums.action_type import ActionType
from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.touch_context import TouchInterpretation

SpokenText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]

_ALLOWED_ACTION_VALUES: dict[ActionType, frozenset[str]] = {
    ActionType.FACE: frozenset(item.value for item in FaceExpression),
    ActionType.GESTURE: frozenset({"nod", "wave", "lean_in", "head_tilt"}),
    ActionType.MOTION: frozenset({"small_forward", "small_back", "orient_user", "settle_idle"}),
    ActionType.SOUND: frozenset({"chime", "soft_ack", "listen_beep"}),
}


class DeviceAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ActionType
    value: str = Field(min_length=1, max_length=64)

    @field_validator("value")
    @classmethod
    def validate_semantic_value(cls, value: str, info: ValidationInfo) -> str:
        action_type = info.data.get("type")
        if action_type is None:
            return value

        normalized = value.strip()
        allowed_values = _ALLOWED_ACTION_VALUES[action_type]
        if normalized not in allowed_values:
            raise ValueError(
                f"Unsupported value '{normalized}' for action type '{action_type.value}'."
            )
        return normalized


class AIResponsePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spoken_text: SpokenText
    emotion: Emotion
    face_expression: FaceExpression
    voice_style: VoiceStyle
    touch_interpretation: TouchInterpretation
    actions: list[DeviceAction] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_action_set(self) -> "AIResponsePlan":
        if len({action.type for action in self.actions}) != len(self.actions):
            raise ValueError("Action types must not repeat within one response plan.")
        return self
