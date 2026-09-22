"""
Centralized structured logger for AirSpark.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from app.utils.paths import resolve_path, ensure_dir


def get_logger(name: str = "AirSpark", log_file: Optional[str] = "logs/airspark.log", level: int = logging.INFO) -> logging.Logger:
    """
    Get a structured, pre-configured logger that logs to stdout and optionally to a file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        resolved_log = resolve_path(log_file)
        ensure_dir(resolved_log.parent)
        file_handler = logging.FileHandler(str(resolved_log), encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
