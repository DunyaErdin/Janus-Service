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
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True
    server_host: str = "0.0.0.0"
    server_port: int = Field(default=8080, ge=1, le=65535)
    proxy_headers: bool = True
    forwarded_allow_ips: str = "*"
    robot_name: str = "Janus"
    default_language: str = "tr-TR"
    websocket_path: str = "/ws/device"
    allowed_device_ids: str | None = None
    llm_provider: Literal["mock", "gemini"] = "mock"
    stt_provider: Literal["placeholder", "gemini"] = "gemini"
    tts_provider: Literal["placeholder", "gemini"] = "gemini"
    gemini_api_key: str | None = None
    gemini_model_id: str = "configure-me"
    gemini_stt_model_id: str = "gemini-2.5-flash"
    gemini_tts_model_id: str = "gemini-3.1-flash-tts-preview"
    gemini_tts_voice_name: str = "Kore"
    request_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    max_audio_chunks_per_session: int = Field(default=256, ge=16, le=4096)
    session_history_limit: int = Field(default=20, ge=4, le=100)
    websocket_hello_timeout_seconds: float = Field(default=10.0, gt=1.0, le=120.0)
    websocket_receive_timeout_seconds: float = Field(default=90.0, gt=5.0, le=3600.0)
    websocket_close_timeout_seconds: float = Field(default=2.0, gt=0.1, le=10.0)
    websocket_max_protocol_errors: int = Field(default=3, ge=1, le=10)
    websocket_max_message_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216)
    websocket_ping_interval_seconds: float = Field(default=20.0, gt=0.0, le=300.0)
    websocket_ping_timeout_seconds: float = Field(default=20.0, gt=0.0, le=300.0)

    @property
    def allowed_device_id_set(self) -> set[str]:
        if not self.allowed_device_ids:
            return set()
        return {
            item.strip()
            for item in self.allowed_device_ids.split(",")
            if item.strip()
        }

    def is_device_allowed(self, device_id: str) -> bool:
        allowed_device_ids = self.allowed_device_id_set
        return not allowed_device_ids or device_id in allowed_device_ids

    @property
    def docs_enabled(self) -> bool:
        return self.environment != "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
