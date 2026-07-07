"""Centralised settings — loaded once at import time, shared across the engine."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="ANTHROPIC_API_KEY env var")
    model_id: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=4096, ge=256, le=16384)

    # ── Sandbox ───────────────────────────────────────────────────────────────
    execution_timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_code_length: int = Field(default=20_000, ge=500)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    max_retries: int = Field(default=3, ge=1, le=5)
    enable_jax: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # ── Router backend ───────────────────────────────────────────────────────
    # "anthropic" (default, NeuralSymbolicRouter) or "local" (LocalTransformerRouter,
    # the from-scratch model trained under model_training/ — see its README for
    # the training roadmap). Only meaningful once a Phase B checkpoint exists.
    model_backend: str = Field(default="anthropic")
    local_model_checkpoint: str = Field(default="", description="Path to a Phase B ckpt_best.pt")
    local_model_tokenizer: str = Field(default="", description="Path to model_training/tokenizer/tokenizer.json")
    local_model_max_new_tokens: int = Field(default=1024, ge=64, le=8192)


settings = Settings()
