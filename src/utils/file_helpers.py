"""
src.utils.file_helpers
======================
File I/O utility functions for the AI Resume Analyzer.

Provides helpers for reading, writing, and validating files — primarily
used by the PDF extraction and model persistence modules.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported file types for upload
ALLOWED_EXTENSIONS: set[str] = {".pdf"}
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB


def ensure_directory_exists(dir_path: str) -> Path:
    """Create a directory and all intermediate parents if they don't exist.

    Args:
        dir_path: Path to the directory to create.

    Returns:
        A Path object pointing to the created or existing directory.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)
    return path


def validate_pdf_file(
    file_path: str,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> tuple[bool, Optional[str]]:
    """Validate a PDF file for existence, extension, and size.

    Args:
        file_path: Path to the file to validate.
        max_size_bytes: Maximum allowed file size in bytes.

    Returns:
        A tuple of (is_valid: bool, error_message: Optional[str]).
        error_message is None if the file is valid.
    """
    path = Path(file_path)

    # Check existence
    if not path.exists():
        return False, f"File not found: {file_path}"

    # Check extension
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{path.suffix}'. "
            f"Only {ALLOWED_EXTENSIONS} files are allowed."
        )

    # Check file size
    file_size = path.stat().st_size
    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        return False, (
            f"File size {size_mb:.1f}MB exceeds maximum allowed "
            f"size of {max_mb:.0f}MB."
        )

    logger.debug("File validation passed: %s (%d bytes)", file_path, file_size)
    return True, None


def get_file_size_mb(file_path: str) -> float:
    """Return the size of a file in megabytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in MB, or 0.0 if the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning("File not found when checking size: %s", file_path)
        return 0.0
    return path.stat().st_size / (1024 * 1024)


# TODO: Add file hash computation (MD5/SHA-256) for deduplication
# TODO: Add support for validating DOCX files in future iterations
