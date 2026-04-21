from __future__ import annotations

import json
import logging

from app.domain.ports.telemetry_port import TelemetryEvent, TelemetryPort


class JsonLoggerTelemetryAdapter(TelemetryPort):
    def __init__(self, logger_name: str = "edge_ai.telemetry") -> None:
        self._logger = logging.getLogger(logger_name)

    async def publish(self, event: TelemetryEvent) -> None:
        payload = event.model_dump(mode="json")
        self._logger.info(json.dumps(payload, ensure_ascii=False))

