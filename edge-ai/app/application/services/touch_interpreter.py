from __future__ import annotations

from app.domain.models.touch_context import (
    RawTouchSensor,
    TouchContext,
    TouchGesture,
    TouchInterpretation,
)


class TouchInterpreter:
    def interpret(self, touch: TouchContext) -> TouchContext:
        interpretation = TouchInterpretation.UNKNOWN

        if touch.sensor == RawTouchSensor.RECORD_BUTTON and touch.gesture in {
            TouchGesture.PRESS,
            TouchGesture.HOLD,
        }:
            interpretation = TouchInterpretation.EXPLICIT_LISTEN_REQUEST
        elif touch.sensor == RawTouchSensor.PETTING_SURFACE and touch.repeat_count >= 3:
            interpretation = TouchInterpretation.PLAYFUL_ENGAGEMENT
        elif touch.sensor == RawTouchSensor.PETTING_SURFACE and touch.gesture in {
            TouchGesture.STROKE,
            TouchGesture.TAP,
        }:
            interpretation = TouchInterpretation.AFFECTION
        elif touch.sensor == RawTouchSensor.PETTING_SURFACE and touch.gesture == TouchGesture.PRESS:
            interpretation = TouchInterpretation.ATTENTION

        return touch.model_copy(update={"interpreted_as": interpretation})

