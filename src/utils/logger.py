"""Logging utility for Hiver AI Support Agent."""
import logging
import sys
from typing import Optional


def get_logger(name: str = "hiver_agent", level: int = logging.INFO) -> logging.Logger:
    """Creates a configured logger with standard stream handling."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
