"""Embedding model management."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings
from utils.logger import get_logger
from utils.exceptions import EmbeddingError

logger = get_logger(__name__)


class EmbeddingManager:
    """Manages the initialization and access to embedding models."""

    def __init__(self) -> None:
        """Initialize the embedding manager."""
        self.settings = get_settings()
        self.model_name = self.settings.embedding_model
        self.device = self.settings.embedding_device
        self._embeddings: HuggingFaceEmbeddings | None = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the initialized embedding model.
        Initializes lazily on first access.
        """
        if self._embeddings is None:
            self._initialize_embeddings()
        return self._embeddings

    def _initialize_embeddings(self) -> None:
        """Initialize the HuggingFace embeddings model."""
        try:
            logger.info(f"Initializing embedding model: {self.model_name} on {self.device}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise EmbeddingError(f"Failed to initialize embedding model: {self.model_name}", str(e))
