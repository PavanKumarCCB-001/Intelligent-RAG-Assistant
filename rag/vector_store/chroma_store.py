"""Vector store operations with ChromaDB."""

from __future__ import annotations

import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import get_settings
from utils.logger import get_logger
from utils.exceptions import VectorStoreError

logger = get_logger(__name__)


class ChromaVectorStore:
    """Wrapper for ChromaDB operations."""

    def __init__(self, embedding_function) -> None:
        """
        Initialize the Chroma vector store.

        Args:
            embedding_function: The embedding function to use.
        """
        self.settings = get_settings()
        self.embedding_function = embedding_function
        self.persist_directory = self.settings.vector_store_dir
        self.collection_name = self.settings.collection_name
        self._store: Chroma | None = None

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

    @property
    def store(self) -> Chroma:
        """
        Get the initialized Chroma vector store.
        Initializes lazily on first access.
        """
        if self._store is None:
            self._initialize_store()
        return self._store

    def _initialize_store(self) -> None:
        """Initialize the Chroma vector store."""
        try:
            logger.info(
                f"Initializing Chroma store in {self.persist_directory}"
            )

            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )

            logger.info("Chroma store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Chroma store: {e}")

            raise VectorStoreError(
                "Failed to initialize Chroma vector store",
                str(e),
            )

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.

        Useful when uploading a completely new set
        of documents and avoiding mixing old documents
        with newly uploaded ones.
        """
        try:
            logger.info("Clearing Chroma collection")

            if self._store is not None:
                self._store.delete_collection()

            self._store = None

            self._initialize_store()

            logger.info("Collection cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

            raise VectorStoreError(
                "Failed to clear Chroma collection",
                str(e),
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of document IDs.
        """
        if not documents:
            logger.warning(
                "No documents provided to add to vector store"
            )
            return []

        try:
            logger.info(
                f"Adding {len(documents)} documents to vector store"
            )

            ids = self.store.add_documents(documents)

            logger.info(
                f"Successfully added {len(ids)} chunks to vector store"
            )

            return ids

        except Exception as e:
            logger.error(
                f"Failed to add documents to vector store: {e}"
            )

            raise VectorStoreError(
                "Failed to add documents to vector store",
                str(e),
            )

    def get_retriever(self, **kwargs):
        """
        Get a retriever interface for the vector store.

        Args:
            kwargs: Additional retriever arguments.

        Returns:
            A LangChain BaseRetriever instance.
        """
        try:
            search_kwargs = kwargs.get("search_kwargs", {})

            if "k" not in search_kwargs:
                search_kwargs["k"] = self.settings.top_k

            kwargs["search_kwargs"] = search_kwargs

            return self.store.as_retriever(**kwargs)

        except Exception as e:
            logger.error(
                f"Failed to create retriever: {e}"
            )

            raise VectorStoreError(
                "Failed to create retriever from vector store",
                str(e),
            )