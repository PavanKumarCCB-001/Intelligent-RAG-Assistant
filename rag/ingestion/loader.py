"""Document loading from various file formats."""

from __future__ import annotations

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

from utils.logger import get_logger
from utils.exceptions import DocumentLoadError

logger = get_logger(__name__)


class DocumentLoader:
    """Handles loading documents from different file types."""

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    @staticmethod
    def load_file(file_path: str) -> List[Document]:
        """
        Load a document from a file path based on its extension.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of LangChain Document objects.
        """
        if not os.path.exists(file_path):
            raise DocumentLoadError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file format: {ext}. Supported formats are: {', '.join(DocumentLoader.SUPPORTED_EXTENSIONS.keys())}"
            )

        try:
            logger.info(f"Loading document: {file_path}")
            loader_class = DocumentLoader.SUPPORTED_EXTENSIONS[ext]
            loader = loader_class(file_path)
            documents = loader.load()
            
            # Ensure metadata has source
            for doc in documents:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path
                    
            logger.info(f"Successfully loaded {len(documents)} pages/sections from {os.path.basename(file_path)}")
            return documents
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise DocumentLoadError(f"Failed to load document: {file_path}", str(e))
