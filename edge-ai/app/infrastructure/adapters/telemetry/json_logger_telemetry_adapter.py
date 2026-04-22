from __future__ import annotations

import logging
from typing import Any

from app.domain.ports.telemetry_port import TelemetryEvent, TelemetryPort


class JsonLoggerTelemetryAdapter(TelemetryPort):
    def __init__(self, logger_name: str = "edge_ai.telemetry") -> None:
        self._logger = logging.getLogger(logger_name)

    async def publish(self, event: TelemetryEvent) -> None:
        payload = self._sanitize(event.model_dump(mode="json"))
        self._logger.info(
            "telemetry_event",
            extra={"structured": payload},
        )

    def _sanitize(self, value: Any, key: str | None = None) -> Any:
        if key is not None and any(
            sensitive_key in key.lower()
            for sensitive_key in ("api_key", "token", "secret", "authorization")
        ):
            return "***redacted***"

        if isinstance(value, dict):
            return {
                item_key: self._sanitize(item_value, item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        return value
