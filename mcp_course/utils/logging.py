"""Logging utilities for the MCP course."""

import logging


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if level is None:
        level = logging.INFO

    logger.setLevel(level)

    # Only add handler if logger doesn't have one already
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
