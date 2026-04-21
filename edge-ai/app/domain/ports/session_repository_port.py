from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.session_context import DeviceSessionContext


class DeviceSessionRepositoryPort(ABC):
    @abstractmethod
    async def get_active(self, device_id: str) -> DeviceSessionContext | None:
        raise NotImplementedError

    @abstractmethod
    async def get_or_create(self, device_id: str) -> DeviceSessionContext:
        raise NotImplementedError

    @abstractmethod
    async def start_new_session(
        self,
        device_id: str,
        requested_session_id: str | None = None,
    ) -> DeviceSessionContext:
        raise NotImplementedError

    @abstractmethod
    async def save(self, session: DeviceSessionContext) -> DeviceSessionContext:
        raise NotImplementedError

    @abstractmethod
    async def end_session(
        self,
        device_id: str,
        reason: str | None = None,
    ) -> DeviceSessionContext | None:
        raise NotImplementedError

