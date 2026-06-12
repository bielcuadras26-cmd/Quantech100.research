"""Central configuration for QuantTech100."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = "QuantTech100"
    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("backend/data")
    reports_dir: Path = Path("backend/reports")
    research_dir: Path = Path("research")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
