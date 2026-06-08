"""LLM Factory for initializing the language model."""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import get_settings
from utils.logger import get_logger
from utils.exceptions import LLMError

logger = get_logger(__name__)


class LLMFactory:
    """Factory for creating and configuring LLM instances."""

    @staticmethod
    def create_llm(api_key: str | None = None) -> BaseChatModel:
        """
        Create and return an initialized ChatGroq instance based on settings.
        
        Args:
            api_key: Optional Groq API key to override settings.
            
        Returns:
            A LangChain BaseChatModel instance.
        """
        settings = get_settings()
        
        final_api_key = api_key or settings.groq_api_key
        
        if not final_api_key:
            logger.error("Groq API key is missing")
            raise LLMError("Groq API key is required. Please provide it in the UI or set GROQ_API_KEY environment variable.")
            
        try:
            logger.info(f"Initializing ChatGroq with model: {settings.default_model}")
            llm = ChatGroq(
                model=settings.default_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=final_api_key,
            )
            logger.info("LLM initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise LLMError("Failed to initialize Groq LLM", str(e))
