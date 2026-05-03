import json

from app.domain.enums.emotion import Emotion
from app.domain.enums.face_expression import FaceExpression
from app.domain.enums.voice_style import VoiceStyle
from app.domain.models.device_event import (
    AudioChunkEvent,
    AudioEncoding,
    HeartbeatEvent,
    SessionStartEvent,
    WakeAudioChunkEvent,
)
from app.domain.models.touch_context import RawTouchSensor, TouchGesture
from app.domain.models.ai_response_plan import AIResponsePlan
from app.domain.models.touch_context import TouchInterpretation
from app.infrastructure.transport.websocket.protocol import (
    build_ack_message,
    build_ai_response_message,
    build_audio_output_chunk_messages,
    build_audio_output_end_message,
    build_audio_output_message,
    build_error_message,
    build_wake_detected_message,
    build_wake_rejected_message,
    parse_incoming_message,
    serialize_outgoing_message,
    to_domain_event,
)


def test_ack_message_includes_legacy_and_current_fields() -> None:
    payload = serialize_outgoing_message(
        build_ack_message(
            device_id="janus-esp-01",
            session_id="session-1",
            correlation_id="corr-1",
            ack_for="hello",
            message="hello_ack",
        )
    )

    assert '"message_type":"ack"' in payload
    assert '"ack_for":"hello"' in payload
    assert '"acked_message_type":"hello"' in payload
    assert '"message":"hello_ack"' in payload
    assert '"detail":"hello_ack"' in payload


def test_ai_response_plan_includes_flattened_fields_for_esp() -> None:
    payload = serialize_outgoing_message(
        build_ai_response_message(
            device_id="janus-esp-01",
            session_id="session-1",
            correlation_id="corr-2",
            response_plan=AIResponsePlan(
                spoken_text="Merhaba, seni duydum.",
                emotion=Emotion.CURIOUS,
                face_expression=FaceExpression.LISTENING,
                voice_style=VoiceStyle.WARM,
                touch_interpretation=TouchInterpretation.NONE,
                actions=[],
            ),
        )
    )

    assert '"message_type":"ai_response_plan"' in payload
    assert '"spoken_text":"Merhaba, seni duydum."' in payload
    assert '"emotion":"curious"' in payload
    assert '"face_expression":"listening"' in payload
    assert '"voice_style":"warm"' in payload
    assert '"response_plan":' in payload


def test_error_message_includes_detail_and_message() -> None:
    payload = serialize_outgoing_message(
        build_error_message(
            code="protocol.invalid_message",
            message="Incoming websocket message failed schema validation.",
            retryable=False,
            device_id="janus-esp-01",
            correlation_id="corr-3",
        )
    )

    assert '"detail":"Incoming websocket message failed schema validation."' in payload
    assert '"message":"Incoming websocket message failed schema validation."' in payload


def test_audio_output_message_serializes_pcm_payload() -> None:
    payload = serialize_outgoing_message(
        build_audio_output_message(
            device_id="janus-esp-01",
            session_id="session-1",
            correlation_id="corr-4",
            encoding="pcm16",
            sample_rate_hz=24000,
            channels=1,
            data_base64="AQID",
            mime_type="audio/L16;rate=24000",
        )
    )

    assert '"message_type":"audio_output"' in payload
    assert '"encoding":"pcm16"' in payload
    assert '"data_base64":"AQID"' in payload


def test_audio_output_chunk_messages_split_base64_on_safe_boundaries() -> None:
    messages = build_audio_output_chunk_messages(
        device_id="janus-esp-01",
        session_id="session-1",
        correlation_id="corr-4",
        encoding="pcm16",
        sample_rate_hz=24000,
        channels=1,
        data_base64="A" * 4096 + "B" * 8,
        mime_type="audio/L16;rate=24000",
    )
    payloads = [serialize_outgoing_message(message) for message in messages]

    assert len(messages) == 3
    assert all(len(message.data_base64) % 4 == 0 for message in messages)
    assert '"message_type":"audio_output_chunk"' in payloads[0]
    assert '"chunk_id":0' in payloads[0]
    assert '"is_final":false' in payloads[0]
    assert '"chunk_id":2' in payloads[2]
    assert '"is_final":true' in payloads[2]


def test_wake_audio_chunk_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "wake_audio_chunk",
                "device_id": "janus-esp-01",
                "interaction_id": "wake-1",
                "chunk_id": 0,
                "encoding": "pcm16",
                "sample_rate_hz": 16000,
                "channels": 1,
                "data_base64": "aGV5IGphbnVz",
                "is_final": True,
                "rms": 400,
                "peak_abs": 900,
            }
        )
    )
    event = to_domain_event(message)

    assert isinstance(event, WakeAudioChunkEvent)
    assert event.interaction_id == "wake-1"
    assert event.encoding == AudioEncoding.PCM16


def test_wake_response_messages_serialize_for_firmware() -> None:
    detected = serialize_outgoing_message(
        build_wake_detected_message(
            device_id="janus-esp-01",
            interaction_id="wake-1",
            correlation_id=None,
            transcript="hey janus",
            confidence=0.9,
        )
    )
    rejected = serialize_outgoing_message(
        build_wake_rejected_message(
            device_id="janus-esp-01",
            interaction_id="wake-2",
            correlation_id=None,
            reason="not_wake_word",
        )
    )
    ended = serialize_outgoing_message(
        build_audio_output_end_message(
            device_id="janus-esp-01",
            session_id="wake-1",
            interaction_id="wake-1",
            correlation_id=None,
        )
    )

    assert '"message_type":"wake_detected"' in detected
    assert '"message_type":"wake_rejected"' in rejected
    assert '"message_type":"audio_output_end"' in ended


def test_firmware_session_start_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "session_start",
                "device_id": "janus-esp-01",
                "session_id": "session-1",
                "trigger": "record_touch",
                "encoding": "pcm16",
                "sample_rate_hz": 16000,
                "channels": 1,
            }
        )
    )
    event = to_domain_event(message)

    assert isinstance(event, SessionStartEvent)
    assert event.requested_session_id == "session-1"
    assert event.encoding == "pcm16"
    assert event.sample_rate_hz == 16000
    assert event.channels == 1


def test_firmware_audio_chunk_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "audio_chunk",
                "device_id": "janus-esp-01",
                "session_id": "session-1",
                "chunk_id": 7,
                "encoding": "pcm16",
                "sample_rate_hz": 16000,
                "channels": 1,
                "data_base64": "AAAA",
                "is_final": True,
            }
        )
    )
    event = to_domain_event(message)

    assert isinstance(event, AudioChunkEvent)
    assert event.session_id == "session-1"
    assert event.chunk_id == 7
    assert event.encoding == AudioEncoding.PCM16
    assert event.is_final is True


def test_firmware_38khz_audio_chunk_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "audio_chunk",
                "device_id": "janus-esp-01",
                "session_id": "session-38k",
                "chunk_id": 1,
                "encoding": "pcm16",
                "sample_rate_hz": 38000,
                "channels": 1,
                "data_base64": "AAAA",
                "is_final": False,
            }
        )
    )
    event = to_domain_event(message)

    assert isinstance(event, AudioChunkEvent)
    assert event.sample_rate_hz == 38000
    assert event.encoding == AudioEncoding.PCM16


def test_legacy_firmware_pcm_s16le_audio_chunk_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "audio_chunk",
                "device_id": "janus-esp-01",
                "session_id": "session-1",
                "chunk_id": 8,
                "encoding": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "data_base64": "AAAA",
                "is_final": False,
            }
        )
    )
    event = to_domain_event(message)

    assert isinstance(event, AudioChunkEvent)
    assert event.encoding == AudioEncoding.PCM16


def test_empty_final_audio_chunk_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "audio_chunk",
                "device_id": "janus-esp-01",
                "session_id": "session-1",
                "chunk_id": 9,
                "encoding": "pcm16",
                "sample_rate_hz": 16000,
                "channels": 1,
                "data_base64": "",
                "is_final": True,
            }
        )
    )
    event = to_domain_event(message)

    assert isinstance(event, AudioChunkEvent)
    assert event.data_base64 == ""
    assert event.is_final is True


def test_firmware_heartbeat_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "heartbeat",
                "device_id": "janus-esp-01",
                "uptime_ms": 12345,
                "wifi_connected": True,
                "transport_connected": True,
                "active_session": False,
            }
        )
    )
    event = to_domain_event(message)

    assert isinstance(event, HeartbeatEvent)
    assert event.uptime_ms == 12345
    assert event.wifi_connected is True
    assert event.transport_connected is True
    assert event.active_session is False


def test_firmware_touch_event_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "touch_event",
                "device_id": "janus-esp-01",
                "sensor": "record",
                "pressed": True,
                "source": "gpio",
            }
        )
    )
    event = to_domain_event(message)

    assert event.touch.sensor == RawTouchSensor.RECORD_BUTTON
    assert event.touch.gesture == TouchGesture.PRESS


def test_firmware_status_shape_is_accepted() -> None:
    message = parse_incoming_message(
        json.dumps(
            {
                "message_type": "status",
                "device_id": "janus-esp-01",
                "level": "info",
                "component": "audio",
                "detail": "stream_ready",
            }
        )
    )
    event = to_domain_event(message)

    assert event.status.note == "info/audio: stream_ready"
