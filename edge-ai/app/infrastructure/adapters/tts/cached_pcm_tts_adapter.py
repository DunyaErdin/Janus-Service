from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest
from app.infrastructure.audio.tts_normalizer import (
    PcmProviderMetadata,
    encode_normalized_audio,
    normalize_tts_provider_audio,
)

_TARGET_SAMPLE_RATE_HZ = 24_000
_TARGET_CHANNELS = 1

logger = logging.getLogger("edge_ai.tts.cached_pcm")


class CachedPcmTtsAdapter(TtsPort):
    """Serves pre-encoded PCM for a fixed set of phrases. Zero API cost on cache hit.

    Cache directory must contain:
    - manifest.json  — { "phrase text": "filename.wav" }
    - WAV or raw PCM16 LE 24 kHz mono files referenced by the manifest

    Returns status="unavailable" on cache miss so a chained provider can continue.
    """

    provider_name = "cached_pcm"

    def __init__(self, cache_dir: str | Path) -> None:
        self._cache: dict[str, str] = {}  # phrase → base64-encoded PCM16 mono 24 kHz
        self._load_cache(Path(cache_dir))

    def _load_cache(self, cache_dir: Path) -> None:
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning(
                "tts_cache_manifest_missing",
                extra={"structured": {"path": str(manifest_path)}},
            )
            return

        try:
            manifest: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(
                "tts_cache_manifest_load_failed",
                extra={"structured": {"error": str(exc)}},
            )
            return

        loaded = 0
        for phrase, filename in manifest.items():
            file_path = cache_dir / filename
            if not file_path.exists():
                logger.warning(
                    "tts_cache_file_missing",
                    extra={"structured": {"phrase": phrase[:50], "file": filename}},
                )
                continue

            try:
                raw = file_path.read_bytes()
                suffix = file_path.suffix.lower()

                if suffix == ".wav":
                    audio = normalize_tts_provider_audio(
                        provider_audio=raw,
                        content_type="audio/wav",
                        provider=self.provider_name,
                    )
                else:
                    # Assume raw PCM16 LE 24 kHz mono (.pcm or any other extension)
                    audio = normalize_tts_provider_audio(
                        provider_audio=raw,
                        content_type=None,
                        provider=self.provider_name,
                        pcm_metadata=PcmProviderMetadata(
                            sample_rate_hz=_TARGET_SAMPLE_RATE_HZ,
                            channels=_TARGET_CHANNELS,
                            sample_format="s16le",
                        ),
                    )

                self._cache[phrase] = encode_normalized_audio(audio)
                loaded += 1
                logger.debug(
                    "tts_cache_phrase_loaded",
                    extra={"structured": {"phrase": phrase[:50], "file": filename}},
                )
            except Exception as exc:
                logger.error(
                    "tts_cache_file_load_failed",
                    extra={"structured": {"phrase": phrase[:50], "file": filename, "error": str(exc)}},
                )

        logger.info(
            "tts_cache_ready",
            extra={"structured": {"loaded": loaded, "total": len(manifest)}},
        )

    @property
    def size(self) -> int:
        return len(self._cache)

    async def plan_synthesis(self, request: TtsSynthesisRequest) -> TtsSynthesisPlan:
        data_b64 = self._cache.get(request.text)
        if data_b64 is None:
            return TtsSynthesisPlan(
                provider=self.provider_name,
                status="unavailable",
            )
        return TtsSynthesisPlan(
            provider=self.provider_name,
            status="generated",
            encoding="pcm16",
            sample_rate_hz=_TARGET_SAMPLE_RATE_HZ,
            channels=_TARGET_CHANNELS,
            data_base64=data_b64,
            mime_type="audio/L16;rate=24000",
        )
