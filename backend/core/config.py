"""
Central configuration management using Pydantic Settings.
All secrets come from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # ── Environment ──────────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")

    # ── Supabase ─────────────────────────────────────────────────
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL")
    SUPABASE_SERVICE_KEY: str = Field(..., env="SUPABASE_SERVICE_KEY")
    SUPABASE_DB_URL: str = Field(..., env="SUPABASE_DB_URL")

    # ── LLM Provider ─────────────────────────────────────────────
    # Use one of: "anthropic" | "groq" | "openai"
    LLM_PROVIDER: str = Field(default="anthropic", env="LLM_PROVIDER")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # LLM model names
    ANTHROPIC_MODEL: str = Field(default="claude-3-haiku-20240307", env="ANTHROPIC_MODEL")
    GROQ_MODEL: str = Field(default="llama3-8b-8192", env="GROQ_MODEL")
    OPENAI_MODEL: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")

    # ── Embeddings ───────────────────────────────────────────────
    # Free, high-quality: "BAAI/bge-small-en-v1.5" (384-dim)
    # Or larger: "BAAI/bge-base-en-v1.5" (768-dim)
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5", env="EMBEDDING_MODEL"
    )
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")

    # ── Chunking ─────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=400, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=80, env="CHUNK_OVERLAP")

    # ── Retrieval ────────────────────────────────────────────────
    TOP_K: int = Field(default=5, env="TOP_K")
    SIMILARITY_THRESHOLD: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")

    # Hybrid search weights (must sum to 1.0)
    DENSE_WEIGHT: float = Field(default=0.7, env="DENSE_WEIGHT")
    SPARSE_WEIGHT: float = Field(default=0.3, env="SPARSE_WEIGHT")

    # ── File Upload ──────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = Field(default=20, env="MAX_FILE_SIZE_MB")
    UPLOAD_DIR: str = Field(default="./uploads", env="UPLOAD_DIR")

    # ── Supabase Table Names ─────────────────────────────────────
    DOCUMENTS_TABLE: str = "documents"
    CHUNKS_TABLE: str = "document_chunks"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
