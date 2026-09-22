"""
Logging configuration setup.
"""
import logging
from app.config.settings import get_settings
from app.utils.logging import get_logger


def configure_logging() -> logging.Logger:
    """Configure and return root AirSpark logger based on settings."""
    settings = get_settings()
    log_file = settings.paths.get("logs_dir", "logs") + "/airspark.log"
    return get_logger("AirSpark", log_file=log_file, level=logging.INFO)
