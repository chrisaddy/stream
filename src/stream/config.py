"""Application configuration loaded from environment variables.

Uses pydantic-settings to validate and type-cast env vars with sensible
defaults for local development.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the Stream application.

    Fields are populated from environment variables or a ``.env`` file.
    """

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/stream"

    # Prefect
    PREFECT_API_URL: str = ""
    PREFECT_API_KEY: str = ""

    # Cloudflare R2 (S3-compatible)
    R2_ENDPOINT_URL: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "stream-models"

    # Anthropic
    ANTHROPIC_API_KEY: str = ""

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 5000
    DEBUG: bool = False


settings = Settings()
