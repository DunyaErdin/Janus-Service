from __future__ import annotations

import json

import pytest

from app.domain.ports.provider_errors import ProviderInvocationError
from app.domain.ports.llm_port import LlmPromptInput, LlmRuntimeContext
from app.infrastructure.adapters.llm.claude_llm_adapter import ClaudeLlmAdapter
from app.schemas.llm_response_schema import get_robot_structured_response_json_schema


def _prompt() -> LlmPromptInput:
    return LlmPromptInput(
        device_id="device-1",
        session_id="session-1",
        runtime_context=LlmRuntimeContext(
            robot_name="Janus",
            language="tr-TR",
            interaction_mode="listening",
        ),
        system_prompt="You are Janus.",
        developer_prompt="Answer as a small robot.",
        dynamic_context="The user greeted the robot.",
        output_instruction="Return a response plan.",
        response_schema=get_robot_structured_response_json_schema(),
    )


def test_claude_payload_uses_messages_api_shape() -> None:
    adapter = ClaudeLlmAdapter(
        api_key="test-key",
        model_id="claude-haiku-4-5-20251001",
        max_tokens=600,
        request_timeout_seconds=1.0,
    )

    payload = adapter._build_request_payload(_prompt())

    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["max_tokens"] == 600
    assert payload["temperature"] == 0.2
    assert payload["system"] == "You are Janus.\n\nAnswer as a small robot."
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0]["type"] == "text"
    assert "RobotStructuredResponsePlan" in payload["messages"][0]["content"][0]["text"]
    assert "Do not wrap it in markdown" in payload["messages"][0]["content"][0]["text"]


def test_claude_parser_accepts_text_json_response() -> None:
    adapter = ClaudeLlmAdapter(
        api_key="test-key",
        model_id="claude-haiku-4-5-20251001",
        max_tokens=600,
        request_timeout_seconds=1.0,
    )
    payload = {
        "stop_reason": "end_turn",
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "spoken_text": "Merhaba.",
                        "emotion": "happy",
                        "face_expression": "smile",
                        "voice_style": "warm",
                        "touch_interpretation": "attention_request",
                        "actions": [{"type": "face", "value": "smile"}],
                    }
                ),
            }
        ],
    }

    response_plan = adapter._parse_messages_response(payload)

    assert response_plan.spoken_text == "Merhaba."
    assert response_plan.emotion.value == "happy"
    assert response_plan.actions[0].value == "smile"


def test_claude_parser_rejects_truncated_response() -> None:
    adapter = ClaudeLlmAdapter(
        api_key="test-key",
        model_id="claude-haiku-4-5-20251001",
        max_tokens=600,
        request_timeout_seconds=1.0,
    )

    with pytest.raises(ProviderInvocationError, match="did not finish cleanly"):
        adapter._parse_messages_response(
            {
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "{}"}],
            }
        )
