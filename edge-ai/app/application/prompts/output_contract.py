from __future__ import annotations

from copy import deepcopy
from textwrap import dedent
from typing import Any


STRICT_OUTPUT_INSTRUCTION = dedent(
    """
    Return exactly one JSON object and nothing else.

    Use this exact structure:
    {
      "spoken_text": "...",
      "emotion": "...",
      "face_expression": "...",
      "voice_style": "...",
      "touch_interpretation": "...",
      "actions": [
        {
          "type": "...",
          "value": "..."
        }
      ]
    }

    Hard requirements:
    - all six top-level fields are required
    - use only allowed enum values
    - use double quotes for all keys and string values
    - do not output null
    - do not output extra keys
    - do not wrap the JSON in markdown
    - do not add any text before or after the JSON
    - actions must always be present
    - if no action is needed, use "actions": []

    If anything is uncertain, output the safest short valid JSON response instead of risking invalid structure.
    """
).strip()

_FEW_SHOT_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "input_summary": (
            "Interaction Mode: replying\n"
            "Recent Touch Context: petting\n"
            "Latest User Utterance: <empty>\n"
            "Conversation Summary: User is offering affectionate touch."
        ),
        "output_payload": {
            "spoken_text": "Canım, buradayım. Seni sevgiyle dinliyorum.",
            "emotion": "affectionate",
            "face_expression": "smile",
            "voice_style": "warm",
            "touch_interpretation": "petting",
            "actions": [
                {
                    "type": "face",
                    "value": "smile",
                }
            ],
        },
    },
    {
        "input_summary": (
            "Interaction Mode: listening\n"
            "Recent Touch Context: explicit_listen_request\n"
            "Latest User Utterance: <empty>\n"
            "Conversation Summary: User pressed the dedicated listen trigger."
        ),
        "output_payload": {
            "spoken_text": "Hazırım, seni dinliyorum. İstediğini söyleyebilirsin.",
            "emotion": "listening",
            "face_expression": "listening",
            "voice_style": "soft",
            "touch_interpretation": "explicit_listen_request",
            "actions": [
                {
                    "type": "sound",
                    "value": "listen_beep",
                }
            ],
        },
    },
    {
        "input_summary": (
            "Interaction Mode: replying\n"
            "Recent Touch Context: none\n"
            "Latest User Utterance: Bugün biraz üzgünüm.\n"
            "Conversation Summary: The user sounds emotionally low."
        ),
        "output_payload": {
            "spoken_text": "Bunu duyduğuma üzüldüm. İstersen biraz yanında kalayım ve seni dinleyeyim.",
            "emotion": "sad",
            "face_expression": "sad_eyes",
            "voice_style": "soft",
            "touch_interpretation": "none",
            "actions": [],
        },
    },
)


def get_strict_output_instruction() -> str:
    return STRICT_OUTPUT_INSTRUCTION


def get_few_shot_examples() -> list[dict[str, Any]]:
    return deepcopy(list(_FEW_SHOT_EXAMPLES))

