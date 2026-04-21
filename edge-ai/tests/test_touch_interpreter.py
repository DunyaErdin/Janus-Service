from app.application.services.touch_interpreter import TouchInterpreter
from app.domain.models.touch_context import (
    RawTouchSensor,
    TouchContext,
    TouchGesture,
    TouchInterpretation,
)


def test_short_pet_touch_maps_to_affection() -> None:
    interpreter = TouchInterpreter()
    interpreted = interpreter.interpret(
        TouchContext(
            sensor=RawTouchSensor.PETTING_SURFACE,
            gesture=TouchGesture.STROKE,
            duration_ms=350,
            repeat_count=1,
        )
    )

    assert interpreted.interpreted_as == TouchInterpretation.AFFECTION


def test_repeated_pet_touch_maps_to_playful_engagement() -> None:
    interpreter = TouchInterpreter()
    interpreted = interpreter.interpret(
        TouchContext(
            sensor=RawTouchSensor.PETTING_SURFACE,
            gesture=TouchGesture.STROKE,
            duration_ms=1200,
            repeat_count=4,
        )
    )

    assert interpreted.interpreted_as == TouchInterpretation.PLAYFUL_ENGAGEMENT


def test_record_press_maps_to_explicit_listen_request() -> None:
    interpreter = TouchInterpreter()
    interpreted = interpreter.interpret(
        TouchContext(
            sensor=RawTouchSensor.RECORD_BUTTON,
            gesture=TouchGesture.PRESS,
            duration_ms=100,
        )
    )

    assert interpreted.interpreted_as == TouchInterpretation.EXPLICIT_LISTEN_REQUEST


def test_unknown_signal_stays_unknown() -> None:
    interpreter = TouchInterpreter()
    interpreted = interpreter.interpret(
        TouchContext(
            sensor=RawTouchSensor.UNKNOWN,
            gesture=TouchGesture.UNKNOWN,
        )
    )

    assert interpreted.interpreted_as == TouchInterpretation.UNKNOWN

