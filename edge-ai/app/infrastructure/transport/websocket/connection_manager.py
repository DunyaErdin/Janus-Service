from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.infrastructure.transport.websocket.protocol import serialize_outgoing_message
from app.schemas.websocket_messages import OutgoingDeviceMessage


class ConnectionUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class ManagedConnection:
    websocket: WebSocket
    connection_id: str = field(default_factory=lambda: str(uuid4()))
    device_id: str | None = None
    session_id: str | None = None
    protocol_version: str | None = None
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat_at: datetime | None = None
    remote_host: str | None = None


class ConnectionManager:
    def __init__(
        self,
        *,
        stale_after_seconds: float,
        close_timeout_seconds: float,
    ) -> None:
        self._connections_by_device: dict[str, ManagedConnection] = {}
        self._connections_by_socket_id: dict[int, ManagedConnection] = {}
        self._lock = asyncio.Lock()
        self._stale_after = timedelta(seconds=stale_after_seconds)
        self._close_timeout_seconds = close_timeout_seconds
        self._logger = logging.getLogger("edge_ai.websocket.connection_manager")

    async def accept(self, websocket: WebSocket) -> ManagedConnection:
        await websocket.accept()
        connection = ManagedConnection(
            websocket=websocket,
            remote_host=self._get_remote_host(websocket),
        )
        async with self._lock:
            self._connections_by_socket_id[id(websocket)] = connection
        self._logger.info(
            "websocket_accepted",
            extra={
                "structured": {
                    "connection_id": connection.connection_id,
                    "remote_host": connection.remote_host,
                }
            },
        )
        return connection

    async def register(
        self,
        device_id: str,
        websocket: WebSocket,
        *,
        protocol_version: str | None = None,
    ) -> ManagedConnection:
        duplicate_connection: ManagedConnection | None = None
        async with self._lock:
            connection = self._connections_by_socket_id.get(id(websocket))
            if connection is None:
                connection = ManagedConnection(
                    websocket=websocket,
                    remote_host=self._get_remote_host(websocket),
                )
                self._connections_by_socket_id[id(websocket)] = connection

            duplicate_connection = self._connections_by_device.get(device_id)
            if duplicate_connection is not None and duplicate_connection.websocket is not websocket:
                self._detach_connection(duplicate_connection)

            connection.device_id = device_id
            connection.protocol_version = protocol_version
            connection.last_seen_at = datetime.now(timezone.utc)
            self._connections_by_device[device_id] = connection

        if duplicate_connection is not None and duplicate_connection.websocket is not websocket:
            await self._safe_close(
                duplicate_connection.websocket,
                code=4009,
                reason="duplicate_device_connection",
            )

        self._logger.info(
            "websocket_registered",
            extra={
                "structured": {
                    "connection_id": connection.connection_id,
                    "device_id": device_id,
                    "protocol_version": protocol_version,
                    "remote_host": connection.remote_host,
                }
            },
        )
        return connection

    async def bind_session(self, device_id: str, session_id: str | None) -> None:
        async with self._lock:
            connection = self._connections_by_device.get(device_id)
            if connection is not None:
                connection.session_id = session_id

    async def mark_activity(
        self,
        *,
        websocket: WebSocket,
        is_heartbeat: bool = False,
    ) -> None:
        async with self._lock:
            connection = self._connections_by_socket_id.get(id(websocket))
            if connection is None:
                return

            now = datetime.now(timezone.utc)
            connection.last_seen_at = now
            if is_heartbeat:
                connection.last_heartbeat_at = now

    async def unregister(
        self,
        *,
        websocket: WebSocket,
        reason: str,
    ) -> None:
        async with self._lock:
            connection = self._connections_by_socket_id.get(id(websocket))
            if connection is None:
                return
            self._detach_connection(connection)

        self._logger.info(
            "websocket_unregistered",
            extra={
                "structured": {
                    "connection_id": connection.connection_id,
                    "device_id": connection.device_id,
                    "session_id": connection.session_id,
                    "reason": reason,
                }
            },
        )

    async def close_socket(
        self,
        websocket: WebSocket,
        *,
        code: int,
        reason: str,
    ) -> None:
        await self.unregister(websocket=websocket, reason=reason)
        await self._safe_close(websocket, code=code, reason=reason)

    async def close_all(
        self,
        *,
        code: int,
        reason: str,
    ) -> None:
        async with self._lock:
            connections = list(self._connections_by_socket_id.values())
            self._connections_by_socket_id.clear()
            self._connections_by_device.clear()

        await asyncio.gather(
            *[
                self._safe_close(connection.websocket, code=code, reason=reason)
                for connection in connections
            ],
            return_exceptions=True,
        )

    async def prune_stale_connections(self) -> int:
        now = datetime.now(timezone.utc)
        stale_connections: list[ManagedConnection] = []

        async with self._lock:
            for connection in list(self._connections_by_socket_id.values()):
                if now - connection.last_seen_at > self._stale_after:
                    stale_connections.append(connection)
                    self._detach_connection(connection)

        for connection in stale_connections:
            self._logger.warning(
                "websocket_connection_stale",
                extra={
                    "structured": {
                        "connection_id": connection.connection_id,
                        "device_id": connection.device_id,
                        "session_id": connection.session_id,
                    }
                },
            )
            await self._safe_close(
                connection.websocket,
                code=4408,
                reason="connection_stale",
            )

        return len(stale_connections)

    async def send_to_device(self, device_id: str, message: OutgoingDeviceMessage) -> None:
        async with self._lock:
            connection = self._connections_by_device.get(device_id)

        if connection is None:
            raise ConnectionUnavailableError(
                f"No active websocket connection for device_id='{device_id}'."
            )

        await self.send_to_socket(connection.websocket, message)

    async def send_to_socket(self, websocket: WebSocket, message: OutgoingDeviceMessage) -> None:
        try:
            if websocket.application_state == WebSocketState.DISCONNECTED:
                raise ConnectionUnavailableError("Websocket is already disconnected.")
            await websocket.send_text(serialize_outgoing_message(message))
        except (WebSocketDisconnect, RuntimeError) as exc:
            await self.unregister(websocket=websocket, reason="send_failed")
            raise ConnectionUnavailableError("Unable to send websocket message.") from exc

    def describe(self, websocket: WebSocket) -> dict[str, str | None]:
        connection = self._connections_by_socket_id.get(id(websocket))
        if connection is None:
            return {
                "connection_id": None,
                "device_id": None,
                "session_id": None,
                "protocol_version": None,
                "remote_host": None,
            }
        return {
            "connection_id": connection.connection_id,
            "device_id": connection.device_id,
            "session_id": connection.session_id,
            "protocol_version": connection.protocol_version,
            "remote_host": connection.remote_host,
        }

    def _detach_connection(self, connection: ManagedConnection) -> None:
        self._connections_by_socket_id.pop(id(connection.websocket), None)
        if connection.device_id is not None:
            current = self._connections_by_device.get(connection.device_id)
            if current is connection:
                self._connections_by_device.pop(connection.device_id, None)

    async def _safe_close(self, websocket: WebSocket, *, code: int, reason: str) -> None:
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await asyncio.wait_for(
                    websocket.close(code=code, reason=reason),
                    timeout=self._close_timeout_seconds,
                )
        except Exception:
            self._logger.debug(
                "websocket_close_failed",
                extra={
                    "structured": {
                        "code": code,
                        "reason": reason,
                    }
                },
            )

    def _get_remote_host(self, websocket: WebSocket) -> str | None:
        if websocket.client is None:
            return None
        return websocket.client.host
