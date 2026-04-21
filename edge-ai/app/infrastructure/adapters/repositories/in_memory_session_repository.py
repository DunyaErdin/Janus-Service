from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from app.domain.models.session_context import DeviceSessionContext
from app.domain.ports.session_repository_port import DeviceSessionRepositoryPort


class InMemorySessionRepository(DeviceSessionRepositoryPort):
    def __init__(self) -> None:
        self._active_sessions: dict[str, DeviceSessionContext] = {}
        self._lock = asyncio.Lock()

    async def get_active(self, device_id: str) -> DeviceSessionContext | None:
        async with self._lock:
            session = self._active_sessions.get(device_id)
            if session is None or not session.active:
                return None
            return session.model_copy(deep=True)

    async def get_or_create(self, device_id: str) -> DeviceSessionContext:
        session = await self.get_active(device_id)
        if session is not None:
            return session
        return await self.start_new_session(device_id=device_id)

    async def start_new_session(
        self,
        device_id: str,
        requested_session_id: str | None = None,
    ) -> DeviceSessionContext:
        new_session = DeviceSessionContext(
            device_id=device_id,
            session_id=requested_session_id or str(uuid4()),
        )
        async with self._lock:
            self._active_sessions[device_id] = new_session.model_copy(deep=True)
        return new_session

    async def save(self, session: DeviceSessionContext) -> DeviceSessionContext:
        stored = session.model_copy(deep=True)
        async with self._lock:
            self._active_sessions[session.device_id] = stored
        return session

    async def end_session(
        self,
        device_id: str,
        reason: str | None = None,
    ) -> DeviceSessionContext | None:
        async with self._lock:
            session = self._active_sessions.get(device_id)
            if session is None:
                return None

            ended_session = session.model_copy(deep=True)
            ended_session.active = False
            ended_session.last_event_at = datetime.now(timezone.utc)
            if reason is not None:
                ended_session.metadata["session_end_reason"] = reason

            self._active_sessions[device_id] = ended_session
            return ended_session.model_copy(deep=True)

