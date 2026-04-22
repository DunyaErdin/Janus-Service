from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)

        if record.exc_info:
            exception = record.exc_info[1]
            payload["exception_type"] = type(exception).__name__ if exception else "Exception"
            payload["exception_message"] = str(exception) if exception else ""

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, level_name: str, json_logs: bool) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
