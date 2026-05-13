from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.debug_audio_routes import router as debug_audio_router
from app.api.websocket_routes import router as websocket_router
from app.config import get_settings
from app.dependencies import (
    build_provider_matrix,
    get_connection_manager,
    get_providers_health,
    get_tts_health_status,
)
from app.logging_config import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level_name=settings.log_level, json_logs=settings.log_json)
    logger = logging.getLogger("edge_ai.app")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        matrix = build_provider_matrix(settings)

        logger.info(
            "edge_ai_starting",
            extra={
                "structured": {
                    "environment": settings.environment,
                    "websocket_path": settings.websocket_path,
                    "docs_enabled": settings.docs_enabled,
                    **matrix,
                }
            },
        )

        # Log provider_matrix as a dedicated event for easy grepping.
        logger.info(
            "provider_matrix",
            extra={"structured": matrix},
        )

        # Warn on missing required credentials so ops sees it at startup.
        if settings.llm_provider == "claude" and not settings.anthropic_api_key:
            logger.error(
                "provider_config_error",
                extra={
                    "structured": {
                        "provider": "claude",
                        "error": "ANTHROPIC_API_KEY missing — LLM will fail at runtime",
                    }
                },
            )
        if settings.tts_provider == "openrouter" and not settings.openrouter_api_key:
            logger.error(
                "provider_config_error",
                extra={
                    "structured": {
                        "provider": "openrouter_tts",
                        "error": "OPENROUTER_API_KEY missing — TTS will fall back to cache/error",
                    }
                },
            )

        yield

        await get_connection_manager().close_all(code=1012, reason="server_shutdown")
        logger.info(
            "edge_ai_stopped",
            extra={"structured": {"environment": settings.environment}},
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
    app.include_router(debug_audio_router)
    app.include_router(websocket_router)

    @app.get("/health", tags=["system"])
    async def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.environment,
        }

    @app.get("/ready", tags=["system"])
    async def readycheck() -> dict:
        llm_ok = settings.llm_provider == "mock" or (
            settings.llm_provider == "claude" and bool(settings.anthropic_api_key)
        )
        stt_ok = settings.stt_provider in {"mock", "placeholder"}
        tts_ok = settings.tts_provider == "placeholder" or (
            settings.tts_provider == "openrouter" and bool(settings.openrouter_api_key)
        )
        all_ok = llm_ok and stt_ok and tts_ok
        return {
            "status": "ready" if all_ok else "degraded",
            "service": settings.app_name,
            **build_provider_matrix(settings),
            "llm_ready": llm_ok,
            "stt_ready": stt_ok,
            "tts_ready": tts_ok,
        }

    @app.get("/health/tts", tags=["system"])
    async def tts_health() -> dict:
        return get_tts_health_status()

    @app.get("/health/providers", tags=["system"])
    async def providers_health() -> dict:
        return get_providers_health()

    @app.get("/version", tags=["system"])
    async def version() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": "0.1.0",
            "device_protocol_version": "1.1",
        }

    return app


app = create_app()
