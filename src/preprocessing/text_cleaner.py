"""
src.preprocessing.text_cleaner
================================
Cleans and normalizes raw text extracted from PDF resumes and job descriptions.

Cleaning steps applied in order:
1. Convert to lowercase
2. Remove URLs and email addresses
3. Remove special characters and punctuation
4. Normalize Unicode characters
5. Collapse whitespace

Usage:
    from src.preprocessing.text_cleaner import clean_text

    cleaned = clean_text(raw_text)
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for performance
_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|"
    r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_DIGITS_PATTERN = re.compile(r"\b\d+\b")


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_special_chars: bool = True,
    remove_digits: bool = False,
    lowercase: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """Clean and normalize raw text for NLP processing.

    Args:
        text: Raw input text from PDF extraction.
        remove_urls: Whether to strip URLs.
        remove_emails: Whether to strip email addresses.
        remove_special_chars: Whether to remove non-alphanumeric characters.
        remove_digits: Whether to remove standalone digit tokens.
        lowercase: Whether to convert text to lowercase.
        normalize_unicode: Whether to normalize Unicode to ASCII-compatible form.

    Returns:
        Cleaned and normalized text string.

    Example:
        >>> clean_text("Visit https://example.com or email me@test.com!")
        'visit or'
    """
    if not text:
        logger.debug("clean_text received empty input; returning empty string.")
        return ""

    original_length = len(text)

    # Step 1: Normalize Unicode characters (handles accented chars, etc.)
    if normalize_unicode:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")

    # Step 2: Convert to lowercase
    if lowercase:
        text = text.lower()

    # Step 3: Remove URLs
    if remove_urls:
        text = _URL_PATTERN.sub(" ", text)

    # Step 4: Remove email addresses
    if remove_emails:
        text = _EMAIL_PATTERN.sub(" ", text)

    # Step 5: Remove special characters (keep alphanumeric and spaces)
    if remove_special_chars:
        text = _SPECIAL_CHARS_PATTERN.sub(" ", text)

    # Step 6: Remove standalone digit tokens (optional)
    if remove_digits:
        text = _DIGITS_PATTERN.sub(" ", text)

    # Step 7: Normalize whitespace (collapse multiple spaces/newlines)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    logger.debug(
        "Text cleaned: %d → %d characters (%.1f%% reduction).",
        original_length,
        len(text),
        (1 - len(text) / max(original_length, 1)) * 100,
    )
    return text


def remove_resume_headers(text: str) -> str:
    """Remove common resume section headers that don't add semantic value.

    Section headers like "EDUCATION", "WORK EXPERIENCE", "SKILLS" are
    structural labels rather than meaningful content. Removing them prevents
    them from inflating TF-IDF weights.

    Args:
        text: Cleaned resume text.

    Returns:
        Text with common resume headers removed.
    """
    # Common resume section headers to strip
    headers = [
        r"\beducation\b",
        r"\bwork experience\b",
        r"\bprofessional experience\b",
        r"\bskills\b",
        r"\btechnical skills\b",
        r"\bsummary\b",
        r"\bobjective\b",
        r"\bcertifications\b",
        r"\bprojects\b",
        r"\bpublications\b",
        r"\breferences\b",
        r"\bhobbies\b",
        r"\binterests\b",
        r"\bachievements\b",
        r"\bawards\b",
        r"\bvolunteer\b",
        r"\blanguages\b",
    ]
    pattern = re.compile("|".join(headers), re.IGNORECASE)
    cleaned = pattern.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


# TODO: Add domain-specific stop phrase removal (e.g., "references available upon request")
# TODO: Benchmark cleaning performance on large resume batches
