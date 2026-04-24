from __future__ import annotations

import base64
import binascii
import io
import wave
from collections.abc import Iterable


def decode_base64_audio_chunks(audio_chunks: Iterable[str]) -> bytes:
    pcm_buffer = bytearray()
    for index, chunk in enumerate(audio_chunks):
        try:
            pcm_buffer.extend(base64.b64decode(chunk, validate=True))
        except binascii.Error as exc:
            raise ValueError(f"Audio chunk at index {index} was not valid base64.") from exc

    if not pcm_buffer:
        raise ValueError("At least one non-empty audio chunk is required.")

    if len(pcm_buffer) % 2 != 0:
        raise ValueError("PCM16 audio payload must contain an even number of bytes.")

    return bytes(pcm_buffer)


def pcm16le_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
) -> bytes:
    if not pcm_bytes:
        raise ValueError("PCM16 audio payload must not be empty.")
    if sample_rate_hz < 8_000 or sample_rate_hz > 96_000:
        raise ValueError("Sample rate is outside the supported WAV range.")
    if channels not in {1, 2}:
        raise ValueError("Only mono and stereo WAV encoding are supported.")

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)
    return wav_buffer.getvalue()
