from enum import Enum


class FaceExpression(str, Enum):
    IDLE = "idle"
    SMILE = "smile"
    BLINK = "blink"
    WINK = "wink"
    LISTENING = "listening"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"
    THINKING = "thinking"
    HAPPY_EYES = "happy_eyes"
    SAD_EYES = "sad_eyes"
