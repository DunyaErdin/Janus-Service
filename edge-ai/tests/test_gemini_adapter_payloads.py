from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.tts_port import TtsSynthesisRequest
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
