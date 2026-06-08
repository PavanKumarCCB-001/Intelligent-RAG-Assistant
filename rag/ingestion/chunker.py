"""Document chunking into smaller segments."""

from __future__ import annotations

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Handles splitting documents into smaller chunks for vector storage."""

    def __init__(self) -> None:
        """Initialize the document chunker."""
        self.settings = get_settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain Document objects to split.

        Returns:
            List of chunked Document objects.
        """
        if not documents:
            logger.warning("No documents provided to chunker")
            return []

        logger.info(f"Splitting {len(documents)} documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} total chunks")
        return chunks
