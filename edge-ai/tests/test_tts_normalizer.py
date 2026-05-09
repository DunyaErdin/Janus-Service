import wave

import pytest

import app.infrastructure.audio.tts_normalizer as tts_normalizer
from app.infrastructure.audio.tts_normalizer import (
    PcmProviderMetadata,
    normalize_tts_provider_audio,
)
from app.infrastructure.audio.wav_codec import pcm16le_to_wav_bytes


def test_normalizer_extracts_wav_frames_and_writes_debug_files(tmp_path, monkeypatch) -> None:
    provider_path = tmp_path / "last_tts_provider_audio.bin"
    normalized_wav_path = tmp_path / "last_tts_normalized.wav"
    normalized_pcm_path = tmp_path / "last_tts_normalized.pcm"
    monkeypatch.setattr(tts_normalizer, "DEBUG_PROVIDER_AUDIO_PATH", provider_path)
    monkeypatch.setattr(tts_normalizer, "DEBUG_NORMALIZED_WAV_PATH", normalized_wav_path)
    monkeypatch.setattr(tts_normalizer, "DEBUG_NORMALIZED_PCM_PATH", normalized_pcm_path)
    pcm = b"\x01\x00\x02\x00"
    wav = pcm16le_to_wav_bytes(pcm, sample_rate_hz=24_000, channels=1)

    normalized = normalize_tts_provider_audio(
        provider_audio=wav,
        content_type="audio/wav",
        provider="test",
    )

    assert normalized.pcm_s16le == pcm
    assert provider_path.read_bytes().startswith(b"RIFF")
    assert normalized_pcm_path.read_bytes() == pcm
    with wave.open(str(normalized_wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnchannels() == 1


def test_normalizer_accepts_headerless_pcm_only_with_metadata() -> None:
    pcm = b"\x01\x00\x02\x00"

    normalized = normalize_tts_provider_audio(
        provider_audio=pcm,
        content_type="audio/L16;rate=24000;channels=1",
        provider="test",
        pcm_metadata=PcmProviderMetadata(
            sample_rate_hz=24_000,
            channels=1,
            sample_format="s16le",
        ),
    )

    assert normalized.pcm_s16le == pcm
    assert normalized.sample_rate_hz == 24_000
    assert normalized.channels == 1


def test_normalizer_rejects_headerless_pcm_without_metadata() -> None:
    with pytest.raises(ValueError, match="explicit PCM provider metadata"):
        normalize_tts_provider_audio(
            provider_audio=b"\x01\x00\x02\x00",
            content_type=None,
            provider="test",
        )


def test_normalizer_rejects_mp3_bytes_as_pcm() -> None:
    with pytest.raises(ValueError):
        normalize_tts_provider_audio(
            provider_audio=b"ID3\x04\x00\x00\x00\x00\x00\x21not pcm",
            content_type="audio/L16;rate=24000;channels=1",
            provider="test",
            pcm_metadata=PcmProviderMetadata(
                sample_rate_hz=24_000,
                channels=1,
                sample_format="s16le",
            ),
        )
