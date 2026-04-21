from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EDGE_AI_",
        extra="ignore",
    )

    app_name: str = "Janus Edge AI"
    environment: str = "development"
    log_level: str = "INFO"
    robot_name: str = "Janus"
    default_language: str = "tr-TR"
    websocket_path: str = "/ws/device"
    llm_provider: Literal["mock", "gemini"] = "mock"
    gemini_api_key: str | None = None
    gemini_model_id: str = "configure-me"
    request_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    max_audio_chunks_per_session: int = Field(default=256, ge=16, le=4096)
    session_history_limit: int = Field(default=20, ge=4, le=100)


@lru_cache
def get_settings() -> Settings:
    return Settings()

