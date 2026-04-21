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
    ActionType.GESTURE: frozenset({"nod", "wave", "tilt_head", "lean_in"}),
    ActionType.MOTION: frozenset({"orient_to_user", "pause_motion", "resume_idle"}),
    ActionType.SOUND: frozenset({"ack_chime", "listen_chime", "soft_alert"}),
    ActionType.LED: frozenset({"idle_glow", "listen_glow", "warm_pulse"}),
    ActionType.NONE: frozenset({"noop"}),
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
    actions: list[DeviceAction] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_action_set(self) -> "AIResponsePlan":
        none_actions = [action for action in self.actions if action.type == ActionType.NONE]
        if none_actions and len(self.actions) > 1:
            raise ValueError("The 'none' action cannot be combined with other actions.")
        return self

