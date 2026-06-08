"""
Centralized configuration for the RAG Assistant.

All settings are loaded from environment variables with sensible defaults.
Uses pydantic-settings for validation and type coercion.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = Field(
        default="Intelligent Multi-Document RAG Assistant",
        description="Application display name",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # ── LLM Provider ────────────────────────────────────────────────────
    llm_provider: Literal["groq"] = Field(
        default="groq",
        description="LLM provider to use. Currently supports: groq",
    )
    groq_api_key: str = Field(
        default="",
        description="Groq API key for LLM access",
    )
    default_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default LLM model identifier",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation (0 = deterministic)",
    )
    llm_max_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum tokens in LLM response",
    )

    # ── Embedding Model ─────────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model name",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for embedding model (cpu or cuda)",
    )

    # ── Document Processing ─────────────────────────────────────────────
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum characters per text chunk",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Character overlap between consecutive chunks",
    )
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum upload file size in megabytes",
    )

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of chunks to retrieve per query",
    )
    retrieval_mode: Literal["vector", "hybrid"] = Field(
        default="vector",
        description="Retrieval mode: 'vector' (default) or 'hybrid' (vector + BM25)",
    )
    hybrid_bm25_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 in hybrid mode (vector weight = 1 - this)",
    )
    enable_reranking: bool = Field(
        default=False,
        description="Enable cross-encoder re-ranking (increases latency)",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking",
    )

    # ── Vector Store ────────────────────────────────────────────────────
    vector_store_dir: str = Field(
        default="vector_db",
        description="Directory for ChromaDB persistent storage",
    )
    collection_name: str = Field(
        default="rag_documents",
        description="ChromaDB collection name",
    )

    # ── Available Models ────────────────────────────────────────────────
    @property
    def available_models(self) -> dict[str, str]:
        """Map of display names to model identifiers."""
        return {
            "Llama 3.3 70B (Versatile)": "llama-3.3-70b-versatile",
            "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Uses lru_cache so the .env file is read only once per process.
    """
    return Settings()
