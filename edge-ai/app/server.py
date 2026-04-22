from __future__ import annotations

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
        log_config=None,
        proxy_headers=settings.proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        ws_max_size=settings.websocket_max_message_bytes,
        ws_ping_interval=settings.websocket_ping_interval_seconds,
        ws_ping_timeout=settings.websocket_ping_timeout_seconds,
        server_header=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
