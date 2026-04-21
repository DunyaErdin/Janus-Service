from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.application.orchestrators.conversation_orchestrator import ConversationOrchestrator
from app.config import get_settings
from app.dependencies import get_connection_manager, get_conversation_orchestrator
from app.infrastructure.transport.websocket.connection_manager import ConnectionManager
from app.infrastructure.transport.websocket.protocol import (
    ProtocolDecodeError,
    build_ack_message,
    build_ai_response_message,
    build_error_message,
    parse_incoming_message,
    to_domain_event,
)

router = APIRouter()
WEBSOCKET_PATH = get_settings().websocket_path


@router.websocket(WEBSOCKET_PATH)
async def device_websocket(
    websocket: WebSocket,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_conversation_orchestrator)],
    connection_manager: Annotated[ConnectionManager, Depends(get_connection_manager)],
) -> None:
    await connection_manager.accept(websocket)
    registered_device_id: str | None = None

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                incoming_message = parse_incoming_message(raw_message)
                domain_event = to_domain_event(incoming_message)

                if incoming_message.message_type == "hello":
                    await connection_manager.register(domain_event.device_id, websocket)
                    registered_device_id = domain_event.device_id
                elif registered_device_id is None:
                    registered_device_id = domain_event.device_id

                result = await orchestrator.handle_event(domain_event)
                await connection_manager.send_to_socket(
                    websocket,
                    build_ack_message(
                        device_id=domain_event.device_id,
                        session_id=result.session_id,
                        correlation_id=domain_event.correlation_id,
                        message=result.ack_message,
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
            except ProtocolDecodeError as exc:
                await connection_manager.send_to_socket(
                    websocket,
                    build_error_message(
                        code="protocol.invalid_message",
                        message=str(exc),
                        retryable=False,
                        device_id=registered_device_id,
                    ),
                )
            except Exception:
                await connection_manager.send_to_socket(
                    websocket,
                    build_error_message(
                        code="transport.unhandled_error",
                        message="The edge service could not process the websocket message.",
                        retryable=True,
                        device_id=registered_device_id,
                    ),
                )
    except WebSocketDisconnect:
        if registered_device_id is not None:
            await connection_manager.unregister(registered_device_id, websocket)

