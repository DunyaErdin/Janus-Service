from enum import Enum


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CURIOUS = "curious"
    SLEEPY = "sleepy"
    EXCITED = "excited"
    THINKING = "thinking"
    SAD = "sad"
    AFFECTIONATE = "affectionate"
    PLAYFUL = "playful"
    LISTENING = "listening"
