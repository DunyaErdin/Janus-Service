from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class RawTouchSensor(str, Enum):
    PETTING_SURFACE = "petting_surface"
    RECORD_BUTTON = "record_button"
    UNKNOWN = "unknown"


class TouchGesture(str, Enum):
    TAP = "tap"
    PRESS = "press"
    RELEASE = "release"
    STROKE = "stroke"
    HOLD = "hold"
    UNKNOWN = "unknown"


class TouchInterpretation(str, Enum):
    UNKNOWN = "unknown"
    AFFECTION = "affection"
    ATTENTION = "attention"
    PLAYFUL_ENGAGEMENT = "playful_engagement"
    EXPLICIT_LISTEN_REQUEST = "explicit_listen_request"


class TouchContext(BaseModel):
    sensor: RawTouchSensor = RawTouchSensor.UNKNOWN
    gesture: TouchGesture = TouchGesture.UNKNOWN
    duration_ms: int | None = Field(default=None, ge=0, le=60000)
    repeat_count: int = Field(default=1, ge=1, le=20)
    intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    interpreted_as: TouchInterpretation = TouchInterpretation.UNKNOWN
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

