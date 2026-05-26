"""Central configuration loaded from environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with .env support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default=Path("."))
    data_dir: Path = Field(default=Path("data"))
    deepgram_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    assemblyai_api_key: str = Field(default="")
    google_application_credentials: str = Field(default="")

    whisper_model: str = Field(default="base")
    faster_whisper_model: str = Field(default="base")
    faster_whisper_device: str = Field(default="auto")
    faster_whisper_compute_type: str = Field(default="int8")
    vosk_model_path: str = Field(default="data/models/vosk-model-small-en-in-0.4")

    indic_asr_model: str = Field(default="ai4bharat/indicwav2vec-hindi")
    enable_indic_asr: bool = Field(default=False)

    deepgram_model: str = Field(default="nova-2")
    deepgram_language: str = Field(default="hi")
    google_stt_language_code: str = Field(default="hi-IN")

    max_workers: int = Field(default=2)
    enable_cache: bool = Field(default=True)
    cache_dir: Path = Field(default=Path("data/outputs/.cache"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(requested: str = "auto") -> Literal["cuda", "cpu"]:
    if requested != "auto":
        return "cuda" if requested == "cuda" else "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
