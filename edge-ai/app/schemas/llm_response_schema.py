from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from app.domain.models.ai_response_plan import AIResponsePlan

ROBOT_STRUCTURED_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RobotStructuredResponsePlan",
    "type": "object",
    "additionalProperties": False,
    "propertyOrdering": [
        "spoken_text",
        "emotion",
        "face_expression",
        "voice_style",
        "touch_interpretation",
        "actions",
    ],
    "required": [
        "spoken_text",
        "emotion",
        "face_expression",
        "voice_style",
        "touch_interpretation",
        "actions",
    ],
    "properties": {
        "spoken_text": {
            "type": "string",
            "description": "Natural spoken Turkish response text, usually one to three short sentences.",
        },
        "emotion": {
            "type": "string",
            "description": "Exactly one dominant robot emotion aligned with the spoken text.",
            "enum": [
                "neutral",
                "happy",
                "curious",
                "sleepy",
                "excited",
                "thinking",
                "sad",
                "affectionate",
                "playful",
                "listening",
            ],
        },
        "face_expression": {
            "type": "string",
            "description": "Display-safe face expression for the OLED or screen renderer.",
            "enum": [
                "idle",
                "smile",
                "blink",
                "wink",
                "listening",
                "surprised",
                "sleepy",
                "thinking",
                "happy_eyes",
                "sad_eyes",
            ],
        },
        "voice_style": {
            "type": "string",
            "description": "Voice synthesis style for downstream TTS selection.",
            "enum": [
                "calm",
                "warm",
                "soft",
                "energetic",
                "serious",
                "playful",
                "sleepy",
                "cheerful",
            ],
        },
        "touch_interpretation": {
            "type": "string",
            "description": "Semantic touch meaning preserved or inferred from runtime context.",
            "enum": [
                "none",
                "petting",
                "affection",
                "attention_request",
                "explicit_listen_request",
                "unknown",
            ],
        },
        "actions": {
            "type": "array",
            "description": "Optional high-level semantic device actions. Use an empty array when no action is needed.",
            "minItems": 0,
            "maxItems": 2,
            "items": {
                "$ref": "#/$defs/action",
            },
        },
    },
    "$defs": {
        "action": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "propertyOrdering": ["type", "value"],
                    "required": ["type", "value"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["face"],
                        },
                        "value": {
                            "type": "string",
                            "enum": [
                                "idle",
                                "smile",
                                "blink",
                                "wink",
                                "listening",
                                "surprised",
                                "sleepy",
                                "thinking",
                                "happy_eyes",
                                "sad_eyes",
                            ],
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "propertyOrdering": ["type", "value"],
                    "required": ["type", "value"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["gesture"],
                        },
                        "value": {
                            "type": "string",
                            "enum": ["nod", "wave", "lean_in", "head_tilt"],
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "propertyOrdering": ["type", "value"],
                    "required": ["type", "value"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["motion"],
                        },
                        "value": {
                            "type": "string",
                            "enum": ["small_forward", "small_back", "orient_user", "settle_idle"],
                        },
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "propertyOrdering": ["type", "value"],
                    "required": ["type", "value"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["sound"],
                        },
                        "value": {
                            "type": "string",
                            "enum": ["chime", "soft_ack", "listen_beep"],
                        },
                    },
                },
            ],
        }
    },
}


class StructuredResponseSchemaError(ValueError):
    pass


class LlmStructuredResponse(AIResponsePlan):
    """Canonical schema the LLM must satisfy before any response is dispatched."""


def get_robot_structured_response_json_schema() -> dict[str, Any]:
    return deepcopy(ROBOT_STRUCTURED_RESPONSE_JSON_SCHEMA)


def parse_llm_structured_response(
    raw_response: str | bytes | dict[str, Any],
) -> LlmStructuredResponse:
    try:
        if isinstance(raw_response, (str, bytes)):
            return LlmStructuredResponse.model_validate_json(raw_response)
        return LlmStructuredResponse.model_validate(raw_response)
    except ValidationError as exc:
        raise StructuredResponseSchemaError(
            "LLM response did not match the RobotStructuredResponsePlan schema."
        ) from exc
