"""Basic text preprocessing before chunking."""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger
from utils.exceptions import PreprocessingError

logger = get_logger(__name__)


class TextPreprocessor:
    """Handles text cleaning and normalization for loaded documents."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize raw text.
        - Removes excessive whitespace
        - Cleans up common artifacts
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        # Remove trailing/leading whitespace
        text = text.strip()
        return text

    @staticmethod
    def process_documents(documents: List[Document]) -> List[Document]:
        """
        Process a list of documents in place.
        """
        try:
            logger.info(f"Preprocessing {len(documents)} documents")
            for doc in documents:
                doc.page_content = TextPreprocessor.clean_text(doc.page_content)
            return documents
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise PreprocessingError("Failed to preprocess documents", str(e))
