"""Retriever manager for fetching documents."""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from rag.vector_store.chroma_store import ChromaVectorStore
from config.settings import get_settings
from utils.logger import get_logger
from utils.exceptions import RetrievalError

logger = get_logger(__name__)


class RetrieverManager:
    """Manages the creation and configuration of document retrievers."""

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        """
        Initialize the retriever manager.

        Args:
            vector_store: The configured vector store instance.
        """
        self.vector_store = vector_store
        self.settings = get_settings()

    def get_retriever(self) -> BaseRetriever:
        """
        Get the configured retriever based on application settings.
        Currently supports standard vector search. (Hybrid to be added).
        
        Returns:
            A LangChain BaseRetriever instance.
        """
        try:
            mode = self.settings.retrieval_mode
            
            if mode == "vector":
                logger.info(f"Creating vector retriever (top_k={self.settings.top_k})")
                return self.vector_store.get_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.settings.top_k}
                )
            else:
                # Fallback to vector search for MVP if hybrid is selected but not implemented
                logger.warning(f"Retrieval mode '{mode}' not fully implemented yet. Falling back to 'vector'.")
                return self.vector_store.get_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.settings.top_k}
                )
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise RetrievalError("Failed to configure document retriever", str(e))
