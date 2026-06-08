"""
Structured logging setup for the RAG Assistant.

Provides a configured logger with console output and consistent formatting.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _create_console_handler(level: int) -> logging.StreamHandler:
    """Create a console handler with formatted output."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    return handler


@lru_cache(maxsize=32)
def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Return a configured logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Optional override for log level. If ``None``, reads from
               the ``LOG_LEVEL`` environment variable (default ``INFO``).

    Returns:
        A configured ``logging.Logger`` instance.
    """
    if level is None:
        import os
        level = os.getenv("LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers when called multiple times
    if not logger.handlers:
        logger.addHandler(_create_console_handler(numeric_level))

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    return logger
