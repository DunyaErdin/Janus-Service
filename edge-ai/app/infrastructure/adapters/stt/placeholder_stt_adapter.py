from __future__ import annotations

from app.domain.ports.provider_errors import ProviderUnavailableError
from app.domain.ports.stt_port import SttPort, TranscriptionRequest, TranscriptionResult


class PlaceholderSttAdapter(SttPort):
    provider_name = "placeholder_stt"

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        raise ProviderUnavailableError(
            "STT adapter is a placeholder. Wire a real transcription provider before expecting final audio chunks to produce user transcripts."
        )

