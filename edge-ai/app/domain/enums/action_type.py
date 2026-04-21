from enum import Enum


class ActionType(str, Enum):
    FACE = "face"
    GESTURE = "gesture"
    MOTION = "motion"
    SOUND = "sound"
    LED = "led"
    NONE = "none"

