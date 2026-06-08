"""
Custom exception hierarchy for the RAG Assistant.

Provides specific exception types for each pipeline stage so that
errors can be caught, logged, and displayed to the user with
meaningful messages.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for all RAG Assistant errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} — {self.details}"
        return self.message


class ConfigurationError(RAGError):
    """Raised when application configuration is invalid or missing."""
    pass


class DocumentLoadError(RAGError):
    """Raised when a document cannot be loaded or parsed."""
    pass


class PreprocessingError(RAGError):
    """Raised when text preprocessing fails."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""
    pass


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""
    pass


class LLMError(RAGError):
    """Raised when LLM invocation fails."""
    pass
