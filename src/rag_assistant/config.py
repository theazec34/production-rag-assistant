"""Environment-backed application settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    knowledge_path: Path = PROJECT_ROOT / "data" / "knowledge_base.json"
    evaluation_path: Path = PROJECT_ROOT / "evaluation" / "golden_v1.json"
    chunk_size_words: int = Field(default=100, ge=40, le=300)
    chunk_overlap_words: int = Field(default=20, ge=0, le=100)
    minimum_retrieval_score: float = Field(default=0.25, ge=0.0, le=1.0)
    provider: str = "extractive"
    provider_base_url: str | None = None
    provider_api_key: str | None = None
    provider_model: str = "local-or-provider-model"
    provider_timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0)


def get_settings() -> Settings:
    return Settings()
