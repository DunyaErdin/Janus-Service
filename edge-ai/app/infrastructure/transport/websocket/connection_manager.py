from __future__ import annotations

import asyncio

from fastapi import WebSocket

from app.infrastructure.transport.websocket.protocol import serialize_outgoing_message
from app.schemas.websocket_messages import OutgoingDeviceMessage


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def accept(self, websocket: WebSocket) -> None:
        await websocket.accept()

    async def register(self, device_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections[device_id] = websocket

    async def unregister(self, device_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            current = self._connections.get(device_id)
            if current is websocket:
                self._connections.pop(device_id, None)

    async def send_to_device(self, device_id: str, message: OutgoingDeviceMessage) -> None:
        async with self._lock:
            websocket = self._connections.get(device_id)

        if websocket is None:
            raise RuntimeError(f"No active websocket connection for device_id='{device_id}'.")

        await self.send_to_socket(websocket, message)

    async def send_to_socket(self, websocket: WebSocket, message: OutgoingDeviceMessage) -> None:
        await websocket.send_text(serialize_outgoing_message(message))

