"""
Structured JSON logging configuration.
"""
import json
import logging
from datetime import datetime
from typing import Any

from src.config import settings


class JSONFormatter(logging.Formatter):
    """Formatter to output logs in JSON format."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
        }

        # Include request_id if it exists in the record's extra context or kwargs
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance with JSON formatting.
    
    Args:
        name: The name of the logger (typically __name__)
        
    Returns:
        A configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Default to INFO if invalid level provided
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Ensure no duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Don't propagate to the root logger to avoid duplicate log lines
        logger.propagate = False

    return logger
