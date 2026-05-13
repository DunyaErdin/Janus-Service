from __future__ import annotations

import base64
import json
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.provider_errors import ProviderInvocationError, ProviderUnavailableError
from app.domain.ports.tts_port import TtsSynthesisPlan, TtsSynthesisRequest
from app.infrastructure.adapters.tts.cached_pcm_tts_adapter import CachedPcmTtsAdapter
from app.infrastructure.adapters.tts.chained_tts_adapter import ChainedTtsAdapter
from app.infrastructure.adapters.tts.openai_tts_adapter import OpenAiTtsAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 24_000
_CHANNELS = 1


def _make_pcm_bytes(num_samples: int = 512) -> bytes:
    """Return raw PCM16 LE silence."""
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def _make_wav_bytes(num_samples: int = 512) -> bytes:
    """Return a minimal valid WAV file with PCM16 LE 24kHz mono content."""
    pcm = _make_pcm_bytes(num_samples)
    buf = tempfile.SpooledTemporaryFile()
    with wave.open(buf, "wb") as wf:  # type: ignore[arg-type]
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm)
    buf.seek(0)
    return buf.read()


def _make_request(text: str = "Merhaba") -> TtsSynthesisRequest:
    return TtsSynthesisRequest(
        device_id="test-device",
        session_id="test-session",
        text=text,
        voice_style=VoiceStyle.CALM,
    )


def _make_plan(status: str = "generated", provider: str = "mock") -> TtsSynthesisPlan:
    if status == "generated":
        return TtsSynthesisPlan(
            provider=provider,
            status="generated",
            encoding="pcm16",
            sample_rate_hz=_SAMPLE_RATE,
            channels=_CHANNELS,
            data_base64=base64.b64encode(_make_pcm_bytes()).decode(),
            mime_type="audio/L16;rate=24000",
        )
    return TtsSynthesisPlan(provider=provider, status=status)


# ---------------------------------------------------------------------------
# CachedPcmTtsAdapter
# ---------------------------------------------------------------------------


class TestCachedPcmTtsAdapter:
    def _adapter_with_wav(self, phrases: dict[str, str]) -> tuple[CachedPcmTtsAdapter, Path]:
        tmp = Path(tempfile.mkdtemp())
        manifest: dict[str, str] = {}
        for phrase, filename in phrases.items():
            wav_path = tmp / filename
            wav_path.write_bytes(_make_wav_bytes())
            manifest[phrase] = filename
        (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return CachedPcmTtsAdapter(tmp), tmp

    @pytest.mark.asyncio
    async def test_hit_returns_generated(self) -> None:
        adapter, _ = self._adapter_with_wav({"Merhaba": "merhaba.wav"})
        plan = await adapter.plan_synthesis(_make_request("Merhaba"))
        assert plan.status == "generated"
        assert plan.data_base64 is not None
        assert plan.sample_rate_hz == _SAMPLE_RATE
        assert plan.channels == _CHANNELS

    @pytest.mark.asyncio
    async def test_miss_returns_unavailable(self) -> None:
        adapter, _ = self._adapter_with_wav({"Merhaba": "merhaba.wav"})
        plan = await adapter.plan_synthesis(_make_request("Bilinmeyen cümle"))
        assert plan.status == "unavailable"
        assert plan.data_base64 is None

    def test_missing_manifest_loads_empty_cache(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        adapter = CachedPcmTtsAdapter(tmp)
        assert adapter.size == 0

    def test_size_reflects_loaded_phrases(self) -> None:
        adapter, _ = self._adapter_with_wav({"A": "a.wav", "B": "b.wav"})
        assert adapter.size == 2

    def test_missing_wav_skipped_gracefully(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        manifest = {"Ghost phrase": "nonexistent.wav"}
        (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        adapter = CachedPcmTtsAdapter(tmp)
        assert adapter.size == 0

    def test_raw_pcm_file_loaded(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        pcm_path = tmp / "tone.pcm"
        pcm_path.write_bytes(_make_pcm_bytes(1024))
        manifest = {"Tone": "tone.pcm"}
        (tmp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        adapter = CachedPcmTtsAdapter(tmp)
        assert adapter.size == 1


# ---------------------------------------------------------------------------
# OpenAiTtsAdapter
# ---------------------------------------------------------------------------


class TestOpenAiTtsAdapter:
    def _adapter(self, api_key: str | None = "test-key") -> OpenAiTtsAdapter:
        return OpenAiTtsAdapter(
            api_key=api_key,
            model="tts-1",
            voice="alloy",
            request_timeout_seconds=5.0,
        )

    @pytest.mark.asyncio
    async def test_no_api_key_raises_unavailable(self) -> None:
        adapter = self._adapter(api_key=None)
        with pytest.raises(ProviderUnavailableError):
            await adapter.plan_synthesis(_make_request())

    @pytest.mark.asyncio
    async def test_successful_response_returns_generated(self) -> None:
        pcm = _make_pcm_bytes(1024)
        mock_resp = MagicMock()
        mock_resp.content = pcm
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            plan = await self._adapter().plan_synthesis(_make_request("Merhaba"))

        assert plan.status == "generated"
        assert plan.encoding == "pcm16"
        assert plan.sample_rate_hz == _SAMPLE_RATE
        assert plan.data_base64 is not None

    @pytest.mark.asyncio
    async def test_empty_response_raises_invocation_error(self) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b""
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderInvocationError, match="empty"):
                await self._adapter().plan_synthesis(_make_request())

    @pytest.mark.asyncio
    async def test_odd_length_response_raises_invocation_error(self) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"\x00" * 3  # odd length
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderInvocationError, match="odd"):
                await self._adapter().plan_synthesis(_make_request())

    @pytest.mark.asyncio
    async def test_timeout_raises_invocation_error(self) -> None:
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderInvocationError, match="timed out"):
                await self._adapter().plan_synthesis(_make_request())

    @pytest.mark.asyncio
    async def test_non_retryable_http_error_raises_immediately(self) -> None:
        error_resp = MagicMock()
        error_resp.status_code = 401
        error_resp.text = "Unauthorized"
        exc = httpx.HTTPStatusError("401", request=MagicMock(), response=error_resp)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=exc)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderInvocationError, match="401"):
                await self._adapter().plan_synthesis(_make_request())


# ---------------------------------------------------------------------------
# ChainedTtsAdapter
# ---------------------------------------------------------------------------


class TestChainedTtsAdapter:
    def _mock_provider(self, name: str, plan: TtsSynthesisPlan) -> MagicMock:
        p = MagicMock()
        p.provider_name = name
        p.plan_synthesis = AsyncMock(return_value=plan)
        return p

    def _failing_provider(self, name: str, exc: Exception) -> MagicMock:
        p = MagicMock()
        p.provider_name = name
        p.plan_synthesis = AsyncMock(side_effect=exc)
        return p

    def test_requires_at_least_one_provider(self) -> None:
        with pytest.raises(ValueError):
            ChainedTtsAdapter()

    @pytest.mark.asyncio
    async def test_first_generated_plan_returned(self) -> None:
        p1 = self._mock_provider("p1", _make_plan("generated", "p1"))
        p2 = self._mock_provider("p2", _make_plan("generated", "p2"))
        adapter = ChainedTtsAdapter(p1, p2)
        plan = await adapter.plan_synthesis(_make_request())
        assert plan.provider == "p1"
        p2.plan_synthesis.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_unavailable_falls_through(self) -> None:
        p1 = self._mock_provider("p1", _make_plan("unavailable", "p1"))
        p2 = self._mock_provider("p2", _make_plan("generated", "p2"))
        adapter = ChainedTtsAdapter(p1, p2)
        plan = await adapter.plan_synthesis(_make_request())
        assert plan.provider == "p2"

    @pytest.mark.asyncio
    async def test_skips_on_provider_error(self) -> None:
        p1 = self._failing_provider("p1", ProviderUnavailableError("down"))
        p2 = self._mock_provider("p2", _make_plan("generated", "p2"))
        adapter = ChainedTtsAdapter(p1, p2)
        plan = await adapter.plan_synthesis(_make_request())
        assert plan.provider == "p2"

    @pytest.mark.asyncio
    async def test_skips_on_invocation_error(self) -> None:
        p1 = self._failing_provider("p1", ProviderInvocationError("bad"))
        p2 = self._mock_provider("p2", _make_plan("generated", "p2"))
        adapter = ChainedTtsAdapter(p1, p2)
        plan = await adapter.plan_synthesis(_make_request())
        assert plan.provider == "p2"

    @pytest.mark.asyncio
    async def test_all_exhausted_returns_unavailable(self) -> None:
        p1 = self._mock_provider("p1", _make_plan("unavailable", "p1"))
        p2 = self._failing_provider("p2", ProviderInvocationError("fail"))
        adapter = ChainedTtsAdapter(p1, p2)
        plan = await adapter.plan_synthesis(_make_request())
        assert plan.status == "unavailable"
        assert plan.provider == "chained_tts"

    @pytest.mark.asyncio
    async def test_provider_order_is_respected(self) -> None:
        plans = []
        for name in ("cache", "openai", "fallback"):
            p = self._mock_provider(name, _make_plan("unavailable", name))
            plans.append(p)
        plans[1].plan_synthesis = AsyncMock(return_value=_make_plan("generated", "openai"))

        adapter = ChainedTtsAdapter(*plans)
        plan = await adapter.plan_synthesis(_make_request())
        assert plan.provider == "openai"
        plans[2].plan_synthesis.assert_not_called()
