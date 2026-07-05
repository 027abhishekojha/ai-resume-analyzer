"""
src.utils.config_loader
=======================
Loads and validates application configuration from YAML files and
environment variables (.env).

Usage:
    from src.utils.config_loader import load_config, get_config_value

    config = load_config("configs/config.yaml")
    threshold = get_config_value(config, "similarity.thresholds.good")
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file at module import time
load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load application configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
                     Relative to the project root.

    Returns:
        A dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Please ensure the file exists at the project root."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from: %s", config_path)
    return config


def get_config_value(
    config: dict[str, Any],
    key_path: str,
    default: Optional[Any] = None,
) -> Any:
    """Retrieve a nested configuration value using dot-notation path.

    Args:
        config: Configuration dictionary (from load_config).
        key_path: Dot-separated key path, e.g. "similarity.thresholds.good".
        default: Default value if the key is not found.

    Returns:
        The configuration value at the specified path, or `default`.

    Example:
        >>> cfg = {"similarity": {"thresholds": {"good": 0.65}}}
        >>> get_config_value(cfg, "similarity.thresholds.good")
        0.65
    """
    keys = key_path.split(".")
    value: Any = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            logger.debug(
                "Config key '%s' not found; returning default: %s",
                key_path,
                default,
            )
            return default

    return value


def get_env_variable(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve a value from environment variables with a fallback default.

    Args:
        key: The environment variable name.
        default: Default value if the variable is not set.

    Returns:
        The environment variable value as a string, or `default`.
    """
    value = os.environ.get(key, default)
    if value is None:
        logger.warning("Environment variable '%s' is not set.", key)
    return value


# TODO: Add config validation using Pydantic models for type-safe config access
# TODO: Support config overlays (e.g., test config overrides base config)
