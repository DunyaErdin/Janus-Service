from __future__ import annotations

import base64
import json
import struct
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.provider_errors import ProviderInvocationError, ProviderUnavailableError
from app.domain.ports.tts_port import TtsSynthesisPlan, TtsSynthesisRequest
from app.infrastructure.adapters.tts.fallback_text_tts_adapter import FallbackTextTtsAdapter
from app.infrastructure.adapters.tts.openrouter_tts_adapter import OpenRouterTtsAdapter

_SAMPLE_RATE = 24_000


def _make_pcm(num_samples: int = 512) -> bytes:
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def _make_request(text: str = "Merhaba") -> TtsSynthesisRequest:
    return TtsSynthesisRequest(
        device_id="test-device",
        session_id="test-session",
        text=text,
        voice_style=VoiceStyle.CALM,
    )


def _adapter(api_key: str | None = "sk-test") -> OpenRouterTtsAdapter:
    return OpenRouterTtsAdapter(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o-mini-tts-2025-12-15",
        voice="coral",
        speed=1.0,
        request_timeout_seconds=5.0,
        http_referer="https://janus.local",
        app_title="Janus Edge AI",
    )


def _mock_http_response(
    status: int,
    content: bytes | None = None,
    json_body: dict | None = None,
) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.content = content or b""
    if json_body is not None:
        resp.json = MagicMock(return_value=json_body)
        resp.text = json.dumps(json_body)
    else:
        resp.json = MagicMock(side_effect=Exception("not json"))
        resp.text = (content or b"").decode("utf-8", errors="replace")
    return resp


def _patch_client(response: MagicMock):
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=mock_client), mock_client


# ---------------------------------------------------------------------------
# Basic config checks
# ---------------------------------------------------------------------------


async def test_no_api_key_raises_unavailable() -> None:
    with pytest.raises(ProviderUnavailableError, match="API key"):
        await _adapter(api_key=None).plan_synthesis(_make_request())


# ---------------------------------------------------------------------------
# HTTP 200 — valid PCM
# ---------------------------------------------------------------------------


async def test_200_valid_pcm_returns_generated() -> None:
    pcm = _make_pcm(1024)
    resp = _mock_http_response(200, content=pcm)
    patcher, _ = _patch_client(resp)
    with patcher:
        plan = await _adapter().plan_synthesis(_make_request())
    assert plan.status == "generated"
    assert plan.encoding == "pcm16"
    assert plan.sample_rate_hz == _SAMPLE_RATE
    assert plan.channels == 1
    assert plan.data_base64 is not None
    assert len(base64.b64decode(plan.data_base64)) == len(pcm)


async def test_200_request_body_is_correct() -> None:
    pcm = _make_pcm(512)
    resp = _mock_http_response(200, content=pcm)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        await _adapter().plan_synthesis(_make_request("Test text"))
    sent_json: dict = mock_client.post.call_args.kwargs["json"]
    assert sent_json["model"] == "openai/gpt-4o-mini-tts-2025-12-15"
    assert sent_json["input"] == "Test text"
    assert sent_json["voice"] == "coral"
    assert sent_json["response_format"] == "pcm"
    assert sent_json["speed"] == 1.0


async def test_200_optional_headers_sent() -> None:
    pcm = _make_pcm(512)
    resp = _mock_http_response(200, content=pcm)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        await _adapter().plan_synthesis(_make_request())
    sent_headers: dict = mock_client.post.call_args.kwargs["headers"]
    assert sent_headers.get("HTTP-Referer") == "https://janus.local"
    assert sent_headers.get("X-Title") == "Janus Edge AI"
    assert "Bearer sk-test" in sent_headers.get("Authorization", "")


async def test_200_no_optional_headers_when_not_configured() -> None:
    pcm = _make_pcm(512)
    resp = _mock_http_response(200, content=pcm)
    a = OpenRouterTtsAdapter(
        api_key="key",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        voice="v",
        speed=1.0,
        request_timeout_seconds=5.0,
    )
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client):
        await a.plan_synthesis(_make_request())
    sent_headers: dict = mock_client.post.call_args.kwargs["headers"]
    assert "HTTP-Referer" not in sent_headers
    assert "X-Title" not in sent_headers


# ---------------------------------------------------------------------------
# HTTP 200 — invalid PCM — must NEVER reach firmware
# ---------------------------------------------------------------------------


async def test_200_empty_body_raises_invocation_error() -> None:
    resp = _mock_http_response(200, content=b"")
    patcher, _ = _patch_client(resp)
    with patcher, pytest.raises(ProviderInvocationError, match="empty"):
        await _adapter().plan_synthesis(_make_request())


async def test_200_odd_length_raises_invocation_error() -> None:
    resp = _mock_http_response(200, content=b"\x00\x01\x02")  # 3 bytes — odd
    patcher, _ = _patch_client(resp)
    with patcher, pytest.raises(ProviderInvocationError, match="odd"):
        await _adapter().plan_synthesis(_make_request())


# ---------------------------------------------------------------------------
# Non-200 errors — non-retryable (no cooldown)
# ---------------------------------------------------------------------------


async def test_401_raises_auth_error_no_cooldown() -> None:
    resp = _mock_http_response(401, json_body={"error": {"message": "Invalid credentials"}})
    patcher, _ = _patch_client(resp)
    a = _adapter()
    with patcher, pytest.raises(ProviderInvocationError, match="auth_error"):
        await a.plan_synthesis(_make_request())
    assert a._cooldown_until is None


async def test_402_payment_required() -> None:
    resp = _mock_http_response(402, json_body={"error": {"message": "Insufficient credits"}})
    patcher, _ = _patch_client(resp)
    with patcher, pytest.raises(ProviderInvocationError, match="payment_required"):
        await _adapter().plan_synthesis(_make_request())


async def test_404_model_not_found() -> None:
    resp = _mock_http_response(404, json_body={"error": {"message": "Model not found"}})
    patcher, _ = _patch_client(resp)
    with patcher, pytest.raises(ProviderInvocationError, match="model_not_found"):
        await _adapter().plan_synthesis(_make_request())


async def test_non_200_json_error_body_not_streamed_to_firmware() -> None:
    """Verify the error is raised as an exception, not returned as a plan with data."""
    resp = _mock_http_response(402, json_body={"error": {"message": "pay up"}})
    patcher, _ = _patch_client(resp)
    with patcher:
        try:
            plan = await _adapter().plan_synthesis(_make_request())
            # Should not reach here
            assert plan.data_base64 is None, "Error JSON must not be in data_base64"
        except ProviderInvocationError:
            pass  # expected


# ---------------------------------------------------------------------------
# Retryable errors — cooldown set
# ---------------------------------------------------------------------------


async def test_429_sets_cooldown_after_max_retries() -> None:
    resp = _mock_http_response(429, json_body={"error": {"message": "rate limit"}})
    patcher, _ = _patch_client(resp)
    a = _adapter()
    with patcher, patch("asyncio.sleep", new=AsyncMock()), pytest.raises(ProviderInvocationError):
        await a.plan_synthesis(_make_request())
    assert a._cooldown_until is not None
    assert a._failure_count == 1
    assert a._last_error_code == "rate_limited"


async def test_500_sets_cooldown() -> None:
    resp = _mock_http_response(500, json_body={"error": {"message": "internal error"}})
    patcher, _ = _patch_client(resp)
    a = _adapter()
    with patcher, patch("asyncio.sleep", new=AsyncMock()), pytest.raises(ProviderInvocationError):
        await a.plan_synthesis(_make_request())
    assert a._cooldown_until is not None
    assert a._last_error_code == "provider_unavailable"


async def test_repeated_failures_escalate_cooldown() -> None:
    resp = _mock_http_response(429, json_body={"error": {"message": "rate limit"}})
    a = _adapter()

    patcher1, _ = _patch_client(resp)
    with patcher1, patch("asyncio.sleep", new=AsyncMock()), pytest.raises(ProviderInvocationError):
        await a.plan_synthesis(_make_request())
    first_cooldown = a._cooldown_until
    assert a._failure_count == 1

    a._cooldown_until = None  # bypass cooldown for second call
    patcher2, _ = _patch_client(resp)
    with patcher2, patch("asyncio.sleep", new=AsyncMock()), pytest.raises(ProviderInvocationError):
        await a.plan_synthesis(_make_request())
    assert a._failure_count == 2
    assert a._cooldown_until > first_cooldown


# ---------------------------------------------------------------------------
# Cooldown enforcement
# ---------------------------------------------------------------------------


async def test_in_cooldown_raises_unavailable_without_http_call() -> None:
    a = _adapter()
    a._cooldown_until = datetime(9999, 1, 1, tzinfo=timezone.utc)
    a._failure_count = 1

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client), pytest.raises(
        ProviderUnavailableError, match="cooldown"
    ):
        await a.plan_synthesis(_make_request())
    mock_client.post.assert_not_called()


async def test_success_clears_cooldown() -> None:
    a = _adapter()
    a._failure_count = 2
    a._last_error_code = "rate_limited"
    # cooldown already expired

    pcm = _make_pcm(512)
    resp = _mock_http_response(200, content=pcm)
    patcher, _ = _patch_client(resp)
    with patcher:
        plan = await a.plan_synthesis(_make_request())
    assert plan.status == "generated"
    assert a._failure_count == 0
    assert a._cooldown_until is None
    assert a._last_error_code is None


# ---------------------------------------------------------------------------
# Retry succeeds on second attempt
# ---------------------------------------------------------------------------


async def test_retries_on_429_succeeds_second_attempt() -> None:
    pcm = _make_pcm(512)
    fail_resp = _mock_http_response(429, json_body={"error": {"message": "rate limit"}})
    ok_resp = _mock_http_response(200, content=pcm)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    a = _adapter()
    with patch("httpx.AsyncClient", return_value=mock_client), patch(
        "asyncio.sleep", new=AsyncMock()
    ):
        plan = await a.plan_synthesis(_make_request())

    assert plan.status == "generated"
    assert a._failure_count == 0
    assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


async def test_timeout_raises_with_cooldown() -> None:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    a = _adapter()
    with patch("httpx.AsyncClient", return_value=mock_client), patch(
        "asyncio.sleep", new=AsyncMock()
    ), pytest.raises(ProviderInvocationError, match="timed out"):
        await a.plan_synthesis(_make_request())
    assert a._cooldown_until is not None


async def test_connect_error_raises_invocation_error() -> None:
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    with patch("httpx.AsyncClient", return_value=mock_client), patch(
        "asyncio.sleep", new=AsyncMock()
    ), pytest.raises(ProviderInvocationError):
        await _adapter().plan_synthesis(_make_request())


# ---------------------------------------------------------------------------
# health_dict
# ---------------------------------------------------------------------------


def test_health_dict_no_key() -> None:
    h = _adapter(api_key=None).health_dict()
    assert h["status"] == "no_key"
    assert h["model"] == "openai/gpt-4o-mini-tts-2025-12-15"
    assert h["voice"] == "coral"
    assert h["cooldown_until"] is None


def test_health_dict_ok() -> None:
    h = _adapter().health_dict()
    assert h["status"] == "ok"
    assert h["cooldown_until"] is None
    assert h["last_error"] is None


def test_health_dict_in_cooldown() -> None:
    a = _adapter()
    a._cooldown_until = datetime(9999, 1, 1, tzinfo=timezone.utc)
    a._failure_count = 1
    h = a.health_dict()
    assert h["status"] == "cooldown"
    assert h["cooldown_until"] is not None


def test_health_dict_last_error_code_shown() -> None:
    a = _adapter()
    a._last_error_code = "payment_required"
    h = a.health_dict()
    assert h["status"] == "payment_required"


# ---------------------------------------------------------------------------
# FallbackTextTtsAdapter
# ---------------------------------------------------------------------------


class TestFallbackTextTtsAdapter:
    def _inner(self, status: str = "generated") -> MagicMock:
        inner = MagicMock()
        inner.provider_name = "cached_pcm"
        if status == "generated":
            plan = TtsSynthesisPlan(
                provider="cached_pcm",
                status="generated",
                encoding="pcm16",
                sample_rate_hz=_SAMPLE_RATE,
                channels=1,
                data_base64=base64.b64encode(_make_pcm()).decode(),
                mime_type="audio/L16;rate=24000",
            )
        else:
            plan = TtsSynthesisPlan(provider="cached_pcm", status="unavailable")
        inner.plan_synthesis = AsyncMock(return_value=plan)
        return inner

    def test_empty_fallback_text_raises(self) -> None:
        with pytest.raises(ValueError):
            FallbackTextTtsAdapter(MagicMock(), "   ")

    async def test_substitutes_text_with_fallback_phrase(self) -> None:
        inner = self._inner()
        adapter = FallbackTextTtsAdapter(inner, "Selam adaş, buradayım.")
        await adapter.plan_synthesis(_make_request("Dynamic AI text not in cache"))
        call_req: TtsSynthesisRequest = inner.plan_synthesis.call_args.args[0]
        assert call_req.text == "Selam adaş, buradayım."

    async def test_other_request_fields_preserved(self) -> None:
        inner = self._inner()
        adapter = FallbackTextTtsAdapter(inner, "fallback")
        req = _make_request("original")
        await adapter.plan_synthesis(req)
        call_req: TtsSynthesisRequest = inner.plan_synthesis.call_args.args[0]
        assert call_req.device_id == req.device_id
        assert call_req.session_id == req.session_id
        assert call_req.voice_style == req.voice_style

    async def test_returns_generated_plan_when_inner_hits(self) -> None:
        plan = await FallbackTextTtsAdapter(
            self._inner("generated"), "x"
        ).plan_synthesis(_make_request())
        assert plan.status == "generated"

    async def test_returns_unavailable_when_inner_misses(self) -> None:
        plan = await FallbackTextTtsAdapter(
            self._inner("unavailable"), "x"
        ).plan_synthesis(_make_request())
        assert plan.status == "unavailable"

    def test_provider_name(self) -> None:
        assert FallbackTextTtsAdapter(MagicMock(), "x").provider_name == "cached_pcm_fallback"
