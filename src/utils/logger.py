"""
src.utils.logger
================
Centralized logging setup for the AI Resume Analyzer.

Configures the Python logging system from the YAML logging config file.
All modules should retrieve loggers using:

    import logging
    logger = logging.getLogger(__name__)

Usage:
    from src.utils.logger import setup_logging
    setup_logging()  # Call once at application startup
"""

import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: str = "configs/logging.yaml",
    default_level: int = logging.INFO,
    log_dir: str = "logs",
) -> None:
    """Configure the logging system from a YAML configuration file.

    Creates the logs directory if it does not exist, then applies the
    logging configuration. Falls back to a basic configuration if the
    YAML file is missing.

    Args:
        config_path: Path to the YAML logging configuration file.
        default_level: Fallback log level if YAML config is unavailable.
        log_dir: Directory where log files will be written.
    """
    # Ensure the log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    yaml_path = Path(config_path)
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            log_config = yaml.safe_load(f)
        logging.config.dictConfig(log_config)
        logging.getLogger(__name__).info(
            "Logging configured from: %s", config_path
        )
    else:
        # Fallback: basic configuration with sensible defaults
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.warning(
            "Logging config file not found at '%s'. Using default config.",
            config_path,
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retrieve a named logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module).
              If None, returns the root logger.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)


# TODO: Add support for structured JSON logging (for cloud log aggregators)
# TODO: Add Sentry / OpenTelemetry integration hooks for production monitoring
