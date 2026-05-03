from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.websocket_routes import router as websocket_router
from app.config import get_settings
from app.dependencies import get_connection_manager
from app.logging_config import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level_name=settings.log_level, json_logs=settings.log_json)
    logger = logging.getLogger("edge_ai.app")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info(
            "edge_ai_starting",
            extra={
                "structured": {
                    "environment": settings.environment,
                    "llm_provider": settings.llm_provider,
                    "stt_provider": settings.stt_provider,
                    "tts_provider": settings.tts_provider,
                    "wake_detector_provider": settings.wake_detector_provider,
                    "websocket_path": settings.websocket_path,
                    "docs_enabled": settings.docs_enabled,
                }
            },
        )
        yield
        await get_connection_manager().close_all(
            code=1012,
            reason="server_shutdown",
        )
        logger.info(
            "edge_ai_stopped",
            extra={
                "structured": {
                    "environment": settings.environment,
                }
            },
        )

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Edge AI orchestration service for Janus home assistant robot.",
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )
    app.include_router(websocket_router)

    @app.get("/health", tags=["system"])
    async def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.environment,
        }

    @app.get("/ready", tags=["system"])
    async def readycheck() -> dict[str, str | bool]:
        providers_configured = (
            settings.wake_detector_provider == "dev_fake"
            or settings.wake_detector_provider == "disabled"
            or bool(settings.gemini_api_key)
        )
        return {
            "status": "ready" if providers_configured else "degraded",
            "service": settings.app_name,
            "wake_detector_provider": settings.wake_detector_provider,
            "provider_credentials_present": providers_configured,
        }

    @app.get("/version", tags=["system"])
    async def version() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": "0.1.0",
            "device_protocol_version": "1.1",
        }

    return app


app = create_app()
