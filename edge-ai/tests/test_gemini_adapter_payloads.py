import base64

from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.tts_port import TtsSynthesisRequest
from app.infrastructure.audio.wav_codec import pcm16le_to_wav_bytes
from app.infrastructure.adapters.stt.gemini_stt_adapter import GeminiSttAdapter
from app.infrastructure.adapters.tts.gemini_tts_adapter import GeminiTtsAdapter


def test_gemini_stt_uses_rest_inline_audio_shape() -> None:
    adapter = GeminiSttAdapter(
        api_key="test-key",
        model_id="gemini-3-flash-preview",
        request_timeout_seconds=1.0,
    )

    payload = adapter._build_request_payload(b"RIFF....WAVE")
    audio_part = payload["contents"][0]["parts"][0]

    assert "inlineData" in audio_part
    assert "inline_data" not in audio_part
    assert audio_part["inlineData"]["mimeType"] == "audio/wav"


def test_gemini_tts_uses_audio_response_modality() -> None:
    adapter = GeminiTtsAdapter(
        api_key="test-key",
        model_id="gemini-3.1-flash-tts-preview",
        voice_name="Kore",
        request_timeout_seconds=1.0,
    )

    payload = adapter._build_request_payload(
        TtsSynthesisRequest(
            device_id="device-1",
            session_id="session-1",
            text="Merhaba.",
            voice_style=VoiceStyle.WARM,
        )
    )

    generation_config = payload["generationConfig"]
    assert generation_config["responseModalities"] == ["AUDIO"]
    assert (
        generation_config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"][
            "voiceName"
        ]
        == "Kore"
    )


def test_gemini_tts_normalizes_wav_response_to_pcm16() -> None:
    adapter = GeminiTtsAdapter(
        api_key="test-key",
        model_id="gemini-3.1-flash-tts-preview",
        voice_name="Kore",
        request_timeout_seconds=1.0,
    )
    pcm_bytes = b"\x01\x00\x02\x00"
    wav_bytes = pcm16le_to_wav_bytes(
        pcm_bytes,
        sample_rate_hz=24000,
        channels=1,
    )

    plan = adapter._parse_generate_content_response(
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/wav",
                                    "data": base64.b64encode(wav_bytes).decode("ascii"),
                                }
                            }
                        ]
                    },
                }
            ]
        }
    )

    assert plan.encoding == "pcm16"
    assert plan.sample_rate_hz == 24000
    assert plan.channels == 1
    assert plan.mime_type == "audio/L16;rate=24000"
    assert base64.b64decode(plan.data_base64 or "") == pcm_bytes


def test_gemini_tts_accepts_provider_pcm_with_explicit_adapter_metadata() -> None:
    adapter = GeminiTtsAdapter(
        api_key="test-key",
        model_id="gemini-3.1-flash-tts-preview",
        voice_name="Kore",
        request_timeout_seconds=1.0,
    )
    pcm_bytes = b"\x01\x00\x02\x00"

    plan = adapter._parse_generate_content_response(
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/L16;rate=24000",
                                    "data": base64.b64encode(pcm_bytes).decode("ascii"),
                                }
                            }
                        ]
                    },
                }
            ]
        }
    )

    assert plan.encoding == "pcm16"
    assert plan.sample_rate_hz == 24000
    assert plan.channels == 1
    assert base64.b64decode(plan.data_base64 or "") == pcm_bytes
