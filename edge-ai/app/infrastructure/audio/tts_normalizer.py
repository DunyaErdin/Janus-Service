from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from app.infrastructure.audio.wav_codec import pcm16le_to_wav_bytes, wav_bytes_to_pcm16le

TARGET_SAMPLE_RATE_HZ = 24_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_FORMAT = "s16le"
DEBUG_PROVIDER_AUDIO_PATH = Path("/tmp/last_tts_provider_audio.bin")
DEBUG_NORMALIZED_WAV_PATH = Path("/tmp/last_tts_normalized.wav")
DEBUG_NORMALIZED_PCM_PATH = Path("/tmp/last_tts_normalized.pcm")

logger = logging.getLogger("edge_ai.audio.tts_normalizer")


@dataclass(frozen=True)
class NormalizedTtsAudio:
    pcm_s16le: bytes
    sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ
    channels: int = TARGET_CHANNELS
    sample_format: str = TARGET_SAMPLE_FORMAT
    mime_type: str = "audio/L16;rate=24000"


@dataclass(frozen=True)
class PcmProviderMetadata:
    sample_rate_hz: int
    channels: int
    sample_format: str


def normalize_tts_provider_audio(
    *,
    provider_audio: bytes,
    content_type: str | None,
    provider: str,
    pcm_metadata: PcmProviderMetadata | None = None,
) -> NormalizedTtsAudio:
    if not provider_audio:
        raise ValueError("TTS provider audio payload was empty.")

    normalized_content_type = (content_type or "").strip()
    logger.info(
        "tts_provider_audio_received",
        extra={
            "structured": {
                "provider": provider,
                "content_type": normalized_content_type or "unknown",
                "first_16_bytes_hex": provider_audio[:16].hex(),
                "byte_length": len(provider_audio),
            }
        },
    )
    _write_debug_file(DEBUG_PROVIDER_AUDIO_PATH, provider_audio)

    lower_content_type = normalized_content_type.lower()
    if _is_wave_audio(provider_audio, lower_content_type):
        pcm, sample_rate_hz, channels = wav_bytes_to_pcm16le(provider_audio)
    elif _is_compressed_audio(provider_audio, lower_content_type):
        pcm, sample_rate_hz, channels = _decode_compressed_audio(provider_audio)
    elif _is_mulaw_audio(lower_content_type):
        metadata = _require_pcm_metadata(pcm_metadata, lower_content_type)
        pcm = _decode_mulaw(provider_audio)
        sample_rate_hz = metadata.sample_rate_hz
        channels = metadata.channels
    elif _is_alaw_audio(lower_content_type):
        metadata = _require_pcm_metadata(pcm_metadata, lower_content_type)
        pcm = _decode_alaw(provider_audio)
        sample_rate_hz = metadata.sample_rate_hz
        channels = metadata.channels
    else:
        metadata = _require_pcm_metadata(pcm_metadata, lower_content_type)
        if metadata.sample_format != TARGET_SAMPLE_FORMAT:
            raise ValueError(
                f"Headerless PCM TTS payload must be {TARGET_SAMPLE_FORMAT}, got {metadata.sample_format!r}."
            )
        pcm = provider_audio
        sample_rate_hz = metadata.sample_rate_hz
        channels = metadata.channels

    normalized_pcm = normalize_pcm16le(
        pcm,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )
    _write_debug_file(DEBUG_NORMALIZED_PCM_PATH, normalized_pcm)
    _write_debug_file(
        DEBUG_NORMALIZED_WAV_PATH,
        pcm16le_to_wav_bytes(
            normalized_pcm,
            sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
            channels=TARGET_CHANNELS,
        ),
    )
    return NormalizedTtsAudio(pcm_s16le=normalized_pcm)


def normalize_pcm16le(
    pcm_s16le: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
) -> bytes:
    if len(pcm_s16le) % 2 != 0:
        raise ValueError("PCM16 payload must contain an even number of bytes.")
    if channels not in {1, 2}:
        raise ValueError("Only mono and stereo PCM16 can be normalized.")
    if sample_rate_hz < 8_000 or sample_rate_hz > 96_000:
        raise ValueError("PCM sample rate is outside the supported range.")

    mono = _pcm16le_to_mono_samples(pcm_s16le, channels)
    if sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        mono = _resample_linear(mono, sample_rate_hz, TARGET_SAMPLE_RATE_HZ)
    if not mono:
        raise ValueError("Normalized PCM16 payload did not include samples.")
    return b"".join(sample.to_bytes(2, "little", signed=True) for sample in mono)


def parse_pcm_metadata_from_content_type(content_type: str | None) -> PcmProviderMetadata | None:
    if not content_type:
        return None
    lower = content_type.lower()
    if not any(token in lower for token in ("audio/l16", "audio/pcm", "pcm", "s16le")):
        return None

    params = _parse_content_type_params(lower)
    rate = params.get("rate") or params.get("sample_rate") or params.get("sample-rate")
    channels = params.get("channels") or params.get("channel")
    if not rate or not channels:
        return None
    try:
        sample_rate_hz = int(rate)
        channel_count = int(channels)
    except ValueError:
        return None
    sample_format = params.get("format") or params.get("sample_format") or TARGET_SAMPLE_FORMAT
    sample_format = sample_format.replace("-", "").replace("_", "").lower()
    if sample_format in {"s16le", "pcm16", "linear16", "l16"}:
        normalized_format = TARGET_SAMPLE_FORMAT
    else:
        normalized_format = sample_format
    return PcmProviderMetadata(
        sample_rate_hz=sample_rate_hz,
        channels=channel_count,
        sample_format=normalized_format,
    )


def _is_wave_audio(audio: bytes, content_type: str) -> bool:
    return audio.startswith(b"RIFF") and audio[8:12] == b"WAVE" or "audio/wav" in content_type or "audio/x-wav" in content_type


def _is_compressed_audio(audio: bytes, content_type: str) -> bool:
    return (
        audio.startswith(b"ID3")
        or audio[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}
        or audio.startswith(b"OggS")
        or "audio/mpeg" in content_type
        or "audio/mp3" in content_type
        or "audio/ogg" in content_type
        or "audio/opus" in content_type
    )


def _is_mulaw_audio(content_type: str) -> bool:
    return "mulaw" in content_type or "mu-law" in content_type or "audio/basic" in content_type


def _is_alaw_audio(content_type: str) -> bool:
    return "alaw" in content_type or "a-law" in content_type


def _require_pcm_metadata(
    metadata: PcmProviderMetadata | None,
    content_type: str,
) -> PcmProviderMetadata:
    if metadata is not None:
        return metadata
    parsed = parse_pcm_metadata_from_content_type(content_type)
    if parsed is None:
        raise ValueError("Headerless TTS audio requires explicit PCM provider metadata.")
    return parsed


def _decode_compressed_audio(audio: bytes) -> tuple[bytes, int, int]:
    try:
        import av
    except ImportError as exc:
        raise ValueError("Compressed TTS audio requires the PyAV decoder dependency.") from exc

    try:
        container = av.open(io.BytesIO(audio))
        stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if stream is None:
            raise ValueError("Compressed TTS audio did not include an audio stream.")
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SAMPLE_RATE_HZ,
        )
        pcm = bytearray()
        for frame in container.decode(stream):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for out_frame in frames:
                pcm.extend(bytes(out_frame.planes[0]))
        if not pcm:
            raise ValueError("Compressed TTS audio decoded to no PCM samples.")
        return bytes(pcm), TARGET_SAMPLE_RATE_HZ, TARGET_CHANNELS
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Compressed TTS audio could not be decoded to PCM16.") from exc


def _pcm16le_to_mono_samples(pcm_s16le: bytes, channels: int) -> list[int]:
    frame_bytes = channels * 2
    usable_len = len(pcm_s16le) - (len(pcm_s16le) % frame_bytes)
    samples: list[int] = []
    for offset in range(0, usable_len, frame_bytes):
        left = int.from_bytes(pcm_s16le[offset : offset + 2], "little", signed=True)
        if channels == 1:
            samples.append(left)
        else:
            right = int.from_bytes(pcm_s16le[offset + 2 : offset + 4], "little", signed=True)
            samples.append(int((left + right) / 2))
    return samples


def _resample_linear(samples: list[int], source_rate_hz: int, target_rate_hz: int) -> list[int]:
    if source_rate_hz == target_rate_hz or len(samples) <= 1:
        return samples
    output_len = max(1, round(len(samples) * target_rate_hz / source_rate_hz))
    ratio = source_rate_hz / target_rate_hz
    output: list[int] = []
    for index in range(output_len):
        source_pos = index * ratio
        base = int(source_pos)
        frac = source_pos - base
        if base >= len(samples) - 1:
            value = samples[-1]
        else:
            value = round(samples[base] * (1.0 - frac) + samples[base + 1] * frac)
        output.append(max(-32768, min(32767, value)))
    return output


def _decode_mulaw(data: bytes) -> bytes:
    samples = [_mulaw_byte_to_i16(byte) for byte in data]
    return b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)


def _decode_alaw(data: bytes) -> bytes:
    samples = [_alaw_byte_to_i16(byte) for byte in data]
    return b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)


def _mulaw_byte_to_i16(byte: int) -> int:
    byte = ~byte & 0xFF
    sign = byte & 0x80
    exponent = (byte >> 4) & 0x07
    mantissa = byte & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if sign else sample


def _alaw_byte_to_i16(byte: int) -> int:
    byte ^= 0x55
    sign = byte & 0x80
    exponent = (byte & 0x70) >> 4
    mantissa = byte & 0x0F
    if exponent == 0:
        sample = (mantissa << 4) + 8
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)
    return sample if sign else -sample


def _parse_content_type_params(content_type: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in content_type.split(";")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip()] = value.strip().strip('"')
    return params


def _write_debug_file(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    except OSError:
        logger.exception("tts_debug_file_write_failed", extra={"structured": {"path": str(path)}})


def encode_normalized_audio(audio: NormalizedTtsAudio) -> str:
    return base64.b64encode(audio.pcm_s16le).decode("ascii")
