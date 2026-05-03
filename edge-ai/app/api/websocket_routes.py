from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.application.orchestrators.conversation_orchestrator import (
    ConversationOrchestrator,
)
from app.config import Settings, get_settings
from app.dependencies import get_connection_manager, get_conversation_orchestrator
from app.infrastructure.transport.websocket.connection_manager import (
    ConnectionManager,
    ConnectionUnavailableError,
)
from app.infrastructure.transport.websocket.protocol import (
    ProtocolDecodeError,
    build_ack_message,
    build_audio_output_end_message,
    build_audio_output_chunk_messages,
    build_ai_response_message,
    build_error_message,
    build_wake_detected_message,
    build_wake_rejected_message,
    parse_incoming_message,
    to_domain_event,
)

router = APIRouter()
WEBSOCKET_PATH = get_settings().websocket_path
logger = logging.getLogger("edge_ai.websocket.route")


@router.websocket(WEBSOCKET_PATH)
async def device_websocket(
    websocket: WebSocket,
    orchestrator: Annotated[
        ConversationOrchestrator, Depends(get_conversation_orchestrator)
    ],
    connection_manager: Annotated[ConnectionManager, Depends(get_connection_manager)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    await connection_manager.accept(websocket)
    if settings.device_auth_token:
        supplied_token = websocket.headers.get(
            "x-janus-device-token"
        ) or websocket.query_params.get("token")
        if supplied_token != settings.device_auth_token:
            await connection_manager.send_to_socket(
                websocket,
                build_error_message(
                    code="protocol.auth_failed",
                    message="Device authentication failed.",
                    retryable=False,
                ),
            )
            await connection_manager.close_socket(
                websocket,
                code=4403,
                reason="auth_failed",
            )
            return

    registered_device_id: str | None = None
    protocol_error_count = 0

    try:
        while True:
            timeout_seconds = (
                settings.websocket_hello_timeout_seconds
                if registered_device_id is None
                else settings.websocket_receive_timeout_seconds
            )

            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                await connection_manager.close_socket(
                    websocket,
                    code=4408,
                    reason="connection_stale",
                )
                break

            await connection_manager.prune_stale_connections()

            try:
                incoming_message = parse_incoming_message(raw_message)
                if (
                    registered_device_id is None
                    and incoming_message.message_type != "hello"
                ):
                    await connection_manager.send_to_socket(
                        websocket,
                        build_error_message(
                            code="protocol.hello_required",
                            message="The first websocket message must be a hello message with a valid device_id.",
                            retryable=True,
                        ),
                    )
                    await connection_manager.close_socket(
                        websocket,
                        code=4401,
                        reason="hello_required",
                    )
                    break

                if not settings.is_device_allowed(incoming_message.device_id):
                    await connection_manager.send_to_socket(
                        websocket,
                        build_error_message(
                            code="protocol.device_not_allowed",
                            message="This device_id is not allowed by the edge service configuration.",
                            retryable=False,
                            device_id=incoming_message.device_id,
                            correlation_id=incoming_message.correlation_id,
                        ),
                    )
                    await connection_manager.close_socket(
                        websocket,
                        code=4403,
                        reason="device_not_allowed",
                    )
                    break

                if (
                    registered_device_id is not None
                    and incoming_message.device_id != registered_device_id
                ):
                    await connection_manager.send_to_socket(
                        websocket,
                        build_error_message(
                            code="protocol.device_identity_mismatch",
                            message="All websocket messages on a connection must use the same device_id after hello.",
                            retryable=False,
                            device_id=registered_device_id,
                            correlation_id=incoming_message.correlation_id,
                        ),
                    )
                    await connection_manager.close_socket(
                        websocket,
                        code=4408,
                        reason="device_identity_mismatch",
                    )
                    break

                domain_event = to_domain_event(incoming_message)
                protocol_error_count = 0

                if incoming_message.message_type == "hello":
                    await connection_manager.register(
                        domain_event.device_id,
                        websocket,
                        protocol_version=incoming_message.protocol_version,
                    )
                    registered_device_id = domain_event.device_id

                await connection_manager.mark_activity(
                    websocket=websocket,
                    is_heartbeat=incoming_message.message_type == "heartbeat",
                )

                result = await orchestrator.handle_event(domain_event)
                if registered_device_id is not None:
                    await connection_manager.bind_session(
                        registered_device_id,
                        result.session_id,
                    )

                await connection_manager.send_to_socket(
                    websocket,
                    build_ack_message(
                        device_id=domain_event.device_id,
                        session_id=result.session_id,
                        correlation_id=domain_event.correlation_id,
                        ack_for=incoming_message.message_type,
                        message=result.ack_message,
                    ),
                )

                if (
                    result.wake_detected is True
                    and result.wake_interaction_id is not None
                ):
                    await connection_manager.send_to_socket(
                        websocket,
                        build_wake_detected_message(
                            device_id=domain_event.device_id,
                            interaction_id=result.wake_interaction_id,
                            correlation_id=domain_event.correlation_id,
                            transcript=result.wake_transcript,
                            confidence=result.wake_confidence,
                        ),
                    )
                elif (
                    result.wake_detected is False
                    and result.wake_interaction_id is not None
                ):
                    await connection_manager.send_to_socket(
                        websocket,
                        build_wake_rejected_message(
                            device_id=domain_event.device_id,
                            interaction_id=result.wake_interaction_id,
                            correlation_id=domain_event.correlation_id,
                            reason=result.wake_reject_reason or "not_wake_word",
                        ),
                    )

                if result.response_plan is not None and result.session_id is not None:
                    await connection_manager.send_to_socket(
                        websocket,
                        build_ai_response_message(
                            device_id=domain_event.device_id,
                            session_id=result.session_id,
                            correlation_id=domain_event.correlation_id,
                            response_plan=result.response_plan,
                        ),
                    )

                audio_session_id = result.wake_interaction_id or result.session_id
                if (
                    audio_session_id is not None
                    and result.tts_plan is not None
                    and result.tts_plan.data_base64 is not None
                    and result.tts_plan.encoding is not None
                    and result.tts_plan.sample_rate_hz is not None
                    and result.tts_plan.channels is not None
                ):
                    for audio_message in build_audio_output_chunk_messages(
                        device_id=domain_event.device_id,
                        session_id=audio_session_id,
                        correlation_id=domain_event.correlation_id,
                        encoding=result.tts_plan.encoding,
                        sample_rate_hz=result.tts_plan.sample_rate_hz,
                        channels=result.tts_plan.channels,
                        data_base64=result.tts_plan.data_base64,
                        mime_type=result.tts_plan.mime_type,
                    ):
                        await connection_manager.send_to_socket(
                            websocket, audio_message
                        )
                    if result.audio_output_end:
                        await connection_manager.send_to_socket(
                            websocket,
                            build_audio_output_end_message(
                                device_id=domain_event.device_id,
                                session_id=audio_session_id,
                                interaction_id=result.wake_interaction_id,
                                correlation_id=domain_event.correlation_id,
                            ),
                        )

                if (
                    incoming_message.message_type == "session_end"
                    and registered_device_id is not None
                ):
                    await connection_manager.bind_session(registered_device_id, None)
            except ProtocolDecodeError as exc:
                protocol_error_count += 1
                try:
                    await connection_manager.send_to_socket(
                        websocket,
                        build_error_message(
                            code="protocol.invalid_message",
                            message=str(exc),
                            retryable=False,
                            device_id=registered_device_id,
                        ),
                    )
                except ConnectionUnavailableError:
                    break
                if protocol_error_count >= settings.websocket_max_protocol_errors:
                    await connection_manager.close_socket(
                        websocket,
                        code=4400,
                        reason="too_many_protocol_errors",
                    )
                    break
            except ConnectionUnavailableError:
                break
            except Exception:
                logger.exception(
                    "websocket_message_processing_failed",
                    extra={
                        "structured": {
                            **connection_manager.describe(websocket),
                            "device_id": registered_device_id,
                        }
                    },
                )
                try:
                    await connection_manager.send_to_socket(
                        websocket,
                        build_error_message(
                            code="transport.unhandled_error",
                            message="The edge service could not process the websocket message.",
                            retryable=True,
                            device_id=registered_device_id,
                        ),
                    )
                except ConnectionUnavailableError:
                    break
    except WebSocketDisconnect:
        logger.info(
            "websocket_disconnected",
            extra={
                "structured": {
                    **connection_manager.describe(websocket),
                    "device_id": registered_device_id,
                }
            },
        )
    finally:
        await connection_manager.unregister(
            websocket=websocket,
            reason="route_closed",
        )
