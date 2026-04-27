"""
utils/logger.py
===============
Structured, timestamped logger factory for the CIFAR-10 project.

All modules obtain a logger via ``get_logger(__name__)`` so that log
records carry the originating module name, timestamp, severity, and
message — making it easy to trace issues in long training runs.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_dir: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a named logger with console and optional file output.

    The function is idempotent: calling it twice with the same ``name``
    returns the same logger without adding duplicate handlers.

    Args:
        name:    Logger name — typically ``__name__`` of the calling module.
        log_dir: If provided, a timestamped ``.log`` file is written there
                 in addition to stdout.
        level:   Logging level (default: ``logging.INFO``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"cifar10_{timestamp}.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate console output)
    logger.propagate = False

    return logger
