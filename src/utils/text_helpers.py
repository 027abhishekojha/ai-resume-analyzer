"""
src.utils.text_helpers
======================
General-purpose text utility functions shared across modules.

These functions handle common string operations that don't belong to the
NLP preprocessing pipeline but are reused in multiple places.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate a text string to a maximum length with an optional suffix.

    Args:
        text: The input text string.
        max_length: Maximum number of characters to retain.
        suffix: String to append when truncation occurs.

    Returns:
        Truncated string if input exceeds max_length, otherwise the original.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """Normalize whitespace in a text string.

    Collapses multiple consecutive whitespace characters (spaces, tabs,
    newlines) into a single space and strips leading/trailing whitespace.

    Args:
        text: Input text with potentially irregular whitespace.

    Returns:
        Text with normalized whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    """Count the number of words in a text string.

    Args:
        text: Input text.

    Returns:
        Word count (split on whitespace).
    """
    if not text or not text.strip():
        return 0
    return len(text.split())


def extract_email(text: str) -> Optional[str]:
    """Extract the first email address found in a text string.

    Args:
        text: Input text potentially containing an email address.

    Returns:
        The first email address found, or None if no email is present.
    """
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    if match:
        logger.debug("Email extracted from text.")
        return match.group(0)
    return None


def extract_phone_number(text: str) -> Optional[str]:
    """Extract the first phone number found in a text string.

    Supports common formats: +1-234-567-8901, (234) 567-8901, 234.567.8901

    Args:
        text: Input text potentially containing a phone number.

    Returns:
        The first phone number found as a string, or None.
    """
    # Pattern covers international and US-style numbers
    pattern = r"(\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
    match = re.search(pattern, text)
    if match:
        logger.debug("Phone number extracted from text.")
        return match.group(0).strip()
    return None


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format a float score (0.0–1.0) as a percentage string.

    Args:
        value: A numeric score between 0.0 and 1.0.
        decimal_places: Number of decimal places to display.

    Returns:
        Formatted percentage string, e.g. "73.5%".
    """
    return f"{value * 100:.{decimal_places}f}%"


# TODO: Add extract_linkedin_url() and extract_github_url() helpers
# TODO: Add language detection utility for multilingual resume support
