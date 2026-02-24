"""
Configuration settings for the RAG AI Engineer project.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or .env file."""

    # Model configuration
    hf_token: str = Field(..., description="Hugging Face API authentication key")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for embeddings",
    )
    llm_model: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model ID for LLM",
    )

    # Database configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant service URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis service URL",
    )

    # Optional configuration
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily web search API key (if unset, DuckDuckGo is used)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity level",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global settings instance
settings = Settings()  # type: ignore[call-arg]
