from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.dependencies import get_tts_adapter
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest
from app.main import create_app


class FakeTtsAdapter(TtsPort):
    provider_name = "fake_tts"

    def __init__(self) -> None:
        self.requests: list[TtsSynthesisRequest] = []

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        self.requests.append(request)
        pcm = b"\x00\x00\x00\x10\x00\xf0\x00\x00"
        return TtsSynthesisPlan(
            provider=self.provider_name,
            status="generated",
            encoding="pcm16",
            sample_rate_hz=24_000,
            channels=1,
            data_base64=base64.b64encode(pcm).decode("ascii"),
            mime_type="audio/L16;rate=24000",
        )


def _client(*, token: str | None = "secret") -> tuple[TestClient, FakeTtsAdapter]:
    fake_tts = FakeTtsAdapter()
    settings = Settings(debug_audio_token=token)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_tts_adapter] = lambda: fake_tts
    return TestClient(app), fake_tts


def test_debug_tts_requires_configured_token() -> None:
    client, _ = _client(token=None)

    response = client.post(
        "/debug/tts",
        json={"text": "Merhaba Janus."},
        headers={"x-janus-debug-token": "secret"},
    )

    assert response.status_code == 503


def test_debug_tts_rejects_wrong_token() -> None:
    client, _ = _client()

    response = client.post(
        "/debug/tts",
        json={"text": "Merhaba Janus."},
        headers={"x-janus-debug-token": "wrong"},
    )

    assert response.status_code == 403


def test_debug_tts_returns_wav_from_pcm_tts_plan() -> None:
    client, fake_tts = _client()

    response = client.post(
        "/debug/tts",
        json={"text": "Merhaba Janus.", "voice_style": "warm"},
        headers={"x-janus-debug-token": "secret"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")
    assert response.content[8:12] == b"WAVE"
    assert fake_tts.requests[0].text == "Merhaba Janus."


def test_debug_response_audio_uses_mock_response_then_tts() -> None:
    client, fake_tts = _client()

    response = client.post(
        "/debug/response-audio",
        json={"transcript": "Selam, beni duyuyor musun?"},
        headers={"x-janus-debug-token": "secret"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["spoken_text"] == "Seni duydum. Yardimci olayim."
    assert payload["encoding"] == "pcm16"
    assert payload["sample_rate_hz"] == 24000
    assert payload["channels"] == 1
    assert fake_tts.requests[0].text == "Seni duydum. Yardimci olayim."
