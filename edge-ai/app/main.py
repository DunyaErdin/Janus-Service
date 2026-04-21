from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.websocket_routes import router as websocket_router
from app.config import get_settings


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Edge AI orchestration service for Janus home assistant robot.",
    )
    app.include_router(websocket_router)

    @app.get("/health", tags=["system"])
    async def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.environment,
        }

    return app


app = create_app()

