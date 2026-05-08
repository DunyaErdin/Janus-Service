from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

from app.application.services.prompt_builder import PromptBuilder
from app.application.services.response_validator import ResponseValidator
from app.config import Settings, get_settings
from app.dependencies import get_prompt_builder, get_response_validator, get_tts_adapter
from app.domain.enums.voice_style import VoiceStyle
from app.domain.ports.provider_errors import (
    ProviderInvocationError,
    ProviderUnavailableError,
)
from app.domain.ports.tts_port import TtsPort, TtsSynthesisPlan, TtsSynthesisRequest
from app.infrastructure.adapters.llm.mock_llm_adapter import MockLlmAdapter
from app.infrastructure.audio.wav_codec import decode_base64_audio_chunks, pcm16le_to_wav_bytes

router = APIRouter(prefix="/debug", tags=["debug-audio"])


class DebugTtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=240)
    voice_style: VoiceStyle = VoiceStyle.WARM
    device_id: str = Field(default="debug-device", min_length=1, max_length=128)
    session_id: str = Field(default="debug-tts", min_length=1, max_length=128)


class DebugResponseAudioRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=500)
    device_id: str = Field(default="debug-device", min_length=1, max_length=128)
    session_id: str = Field(default="debug-response-audio", min_length=1, max_length=128)


class DebugResponseAudioResponse(BaseModel):
    spoken_text: str
    voice_style: VoiceStyle
    emotion: str
    face_expression: str
    encoding: str
    sample_rate_hz: int
    channels: int
    mime_type: str | None
    data_base64: str


async def require_debug_audio_token(
    settings: Annotated[Settings, Depends(get_settings)],
    supplied_token: Annotated[str | None, Header(alias="x-janus-debug-token")] = None,
) -> Settings:
    configured_token = settings.debug_audio_token
    if not configured_token:
        raise HTTPException(
            status_code=503,
            detail="Debug audio endpoints are disabled until EDGE_AI_DEBUG_AUDIO_TOKEN is configured.",
        )
    if supplied_token is None or not hmac.compare_digest(supplied_token, configured_token):
        raise HTTPException(status_code=403, detail="Debug audio token is invalid.")
    return settings


@router.post("/tts")
async def debug_tts(
    request: DebugTtsRequest,
    _: Annotated[Settings, Depends(require_debug_audio_token)],
    tts: Annotated[TtsPort, Depends(get_tts_adapter)],
) -> Response:
    plan = await _synthesize_or_http_error(
        tts,
        TtsSynthesisRequest(
            device_id=request.device_id,
            session_id=request.session_id,
            text=request.text,
            voice_style=request.voice_style,
        ),
    )
    wav_bytes = _tts_plan_to_wav(plan)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="janus-debug-tts.wav"',
            "X-Janus-Audio-Encoding": plan.encoding or "",
            "X-Janus-Sample-Rate-Hz": str(plan.sample_rate_hz or ""),
            "X-Janus-Channels": str(plan.channels or ""),
        },
    )


@router.post("/response-audio")
async def debug_response_audio(
    request: DebugResponseAudioRequest,
    _: Annotated[Settings, Depends(require_debug_audio_token)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    response_validator: Annotated[ResponseValidator, Depends(get_response_validator)],
    tts: Annotated[TtsPort, Depends(get_tts_adapter)],
) -> DebugResponseAudioResponse:
    prompt = prompt_builder.build(
        device_id=request.device_id,
        session_id=request.session_id,
        language=None,
        interaction_mode="replying",
        device_state=None,
        touch_context=None,
        conversation_summary="",
        latest_user_utterance=request.transcript,
    )
    response_plan = response_validator.validate(
        await MockLlmAdapter().generate_response(prompt)
    )
    tts_plan = await _synthesize_or_http_error(
        tts,
        TtsSynthesisRequest(
            device_id=request.device_id,
            session_id=request.session_id,
            text=response_plan.spoken_text,
            voice_style=response_plan.voice_style,
        ),
    )
    _assert_pcm16_audio_plan(tts_plan)

    return DebugResponseAudioResponse(
        spoken_text=response_plan.spoken_text,
        voice_style=response_plan.voice_style,
        emotion=response_plan.emotion.value,
        face_expression=response_plan.face_expression.value,
        encoding=tts_plan.encoding or "",
        sample_rate_hz=tts_plan.sample_rate_hz or 0,
        channels=tts_plan.channels or 0,
        mime_type=tts_plan.mime_type,
        data_base64=tts_plan.data_base64 or "",
    )


async def _synthesize_or_http_error(
    tts: TtsPort,
    request: TtsSynthesisRequest,
) -> TtsSynthesisPlan:
    try:
        return await tts.plan_synthesis(request)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderInvocationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _tts_plan_to_wav(plan: TtsSynthesisPlan) -> bytes:
    _assert_pcm16_audio_plan(plan)
    pcm_bytes = decode_base64_audio_chunks([plan.data_base64 or ""])
    return pcm16le_to_wav_bytes(
        pcm_bytes,
        sample_rate_hz=plan.sample_rate_hz or 24_000,
        channels=plan.channels or 1,
    )


def _assert_pcm16_audio_plan(plan: TtsSynthesisPlan) -> None:
    if plan.data_base64 is None:
        raise HTTPException(status_code=502, detail="TTS provider returned no audio bytes.")
    if plan.encoding != "pcm16":
        raise HTTPException(
            status_code=502,
            detail=f"TTS provider returned unsupported encoding {plan.encoding!r}.",
        )
    if plan.sample_rate_hz is None or plan.channels is None:
        raise HTTPException(
            status_code=502,
            detail="TTS provider did not include sample rate and channel metadata.",
        )
