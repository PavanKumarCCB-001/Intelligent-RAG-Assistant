"""RAG Chain combining retrieval and generation."""

from __future__ import annotations

from typing import Dict, Any

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from utils.logger import get_logger
from utils.exceptions import LLMError

logger = get_logger(__name__)


class RAGChain:
    """Manages the creation and execution of the RAG pipeline."""

    # Default QA prompt template
    QA_SYSTEM_PROMPT = """
You are a Retrieval-Augmented Generation (RAG) assistant.

Answer questions ONLY using the information provided in the retrieved context.

Rules:
1. Use ONLY the retrieved context.
2. Do NOT use your own knowledge.
3. Do NOT infer information that is not explicitly stated.
4. Do NOT speculate.
5. If the answer cannot be found in the retrieved context, respond exactly:

"I could not find this information in the uploaded documents."

6. When answering, cite the relevant document sources if available.

Retrieved Context:
{context}
"""

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever) -> None:
        """
        Initialize the RAG chain.

        Args:
            llm: The initialized language model.
            retriever: The document retriever.
        """
        self.llm = llm
        self.retriever = retriever
        self._chain = None

    @property
    def chain(self):
        """Get the compiled RAG chain."""
        if self._chain is None:
            self._build_chain()
        return self._chain

    def _build_chain(self) -> None:
        """Build the retrieval and document combination chain."""
        try:
            logger.info("Building RAG chain")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.QA_SYSTEM_PROMPT),
                ("human", "{input}"),
            ])

            # Chain to combine retrieved documents and pass them to the LLM
            combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # The final retrieval chain that first retrieves, then passes to combine_docs_chain
            self._chain = create_retrieval_chain(self.retriever, combine_docs_chain)
            logger.info("RAG chain built successfully")
        except Exception as e:
            logger.error(f"Failed to build RAG chain: {e}")
            raise LLMError("Failed to build RAG chain", str(e))

    def invoke(self, query: str) -> Dict[str, Any]:
        """
        Invoke the RAG chain with a user query.

        Args:
            query: The user's question.

        Returns:
            Dictionary containing the 'answer' and 'context' (retrieved documents).
        """
        try:
            logger.info(f"Invoking RAG chain for query: '{query}'")
            response = self.chain.invoke({"input": query})
            return response
        except Exception as e:
            logger.error(f"Error during chain invocation: {e}")
            raise LLMError("Failed to generate response", str(e))
