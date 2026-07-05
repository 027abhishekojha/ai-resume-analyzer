"""
tests/test_utils.py
====================
Unit tests for shared utility functions:
    - config_loader
    - text_helpers
    - file_helpers

Run with:
    pytest tests/test_utils.py -v
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.utils.text_helpers import (
    clean_whitespace,
    count_words,
    extract_email,
    extract_phone_number,
    format_percentage,
    truncate_text,
)
from src.utils.file_helpers import (
    validate_pdf_file,
    get_file_size_mb,
    ALLOWED_EXTENSIONS,
)


# ─── Text Helpers Tests ────────────────────────────────────────────────────────


class TestTruncateText:
    """Tests for src.utils.text_helpers.truncate_text"""

    def test_short_text_unchanged(self):
        """Text shorter than max_length should not be truncated."""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_long_text_is_truncated(self):
        """Text longer than max_length should be truncated with suffix."""
        text = "A" * 600
        result = truncate_text(text, max_length=500)
        assert len(result) <= 500
        assert result.endswith("...")

    def test_exact_length_not_truncated(self):
        """Text exactly at max_length should not be truncated."""
        text = "A" * 500
        result = truncate_text(text, max_length=500)
        assert result == text

    def test_custom_suffix(self):
        """Truncation should use the specified custom suffix."""
        text = "A" * 200
        result = truncate_text(text, max_length=100, suffix=" [more]")
        assert result.endswith(" [more]")


class TestCleanWhitespace:
    """Tests for src.utils.text_helpers.clean_whitespace"""

    def test_collapses_multiple_spaces(self):
        """Multiple spaces should be collapsed to a single space."""
        result = clean_whitespace("hello   world")
        assert result == "hello world"

    def test_collapses_newlines(self):
        """Newlines should be replaced with spaces."""
        result = clean_whitespace("hello\nworld\n\n!")
        assert "\n" not in result

    def test_strips_leading_trailing(self):
        """Leading and trailing whitespace should be stripped."""
        result = clean_whitespace("  hello  ")
        assert result == "hello"

    def test_empty_string(self):
        """Empty string input should return empty string."""
        assert clean_whitespace("") == ""


class TestCountWords:
    """Tests for src.utils.text_helpers.count_words"""

    def test_basic_word_count(self):
        """Should count words split by spaces."""
        assert count_words("machine learning engineer") == 3

    def test_empty_string_returns_zero(self):
        """Empty string should return 0."""
        assert count_words("") == 0

    def test_whitespace_only_returns_zero(self):
        """Whitespace-only string should return 0."""
        assert count_words("   ") == 0

    def test_single_word(self):
        """Single word should return 1."""
        assert count_words("Python") == 1


class TestExtractEmail:
    """Tests for src.utils.text_helpers.extract_email"""

    def test_extracts_valid_email(self):
        """Should extract a standard email address."""
        result = extract_email("Contact: john.doe@example.com today.")
        assert result == "john.doe@example.com"

    def test_returns_none_when_no_email(self):
        """Should return None when no email is present."""
        result = extract_email("No email here.")
        assert result is None

    def test_extracts_first_email(self):
        """Should extract the first email when multiple are present."""
        text = "a@b.com and c@d.com"
        result = extract_email(text)
        assert result == "a@b.com"


class TestExtractPhoneNumber:
    """Tests for src.utils.text_helpers.extract_phone_number"""

    def test_extracts_us_format(self):
        """Should extract a US-style phone number."""
        result = extract_phone_number("Call me at 234-567-8901")
        assert result is not None
        assert "234" in result

    def test_returns_none_when_no_phone(self):
        """Should return None when no phone number is present."""
        result = extract_phone_number("No phone here.")
        assert result is None


class TestFormatPercentage:
    """Tests for src.utils.text_helpers.format_percentage"""

    def test_formats_correctly(self):
        """Should format a 0-1 float as a percentage string."""
        assert format_percentage(0.735) == "73.5%"

    def test_zero(self):
        """Should format 0.0 as '0.0%'."""
        assert format_percentage(0.0) == "0.0%"

    def test_one(self):
        """Should format 1.0 as '100.0%'."""
        assert format_percentage(1.0) == "100.0%"

    def test_custom_decimal_places(self):
        """Should respect the decimal_places argument."""
        result = format_percentage(0.735, decimal_places=2)
        assert result == "73.50%"


# ─── File Helpers Tests ────────────────────────────────────────────────────────


class TestValidatePdfFile:
    """Tests for src.utils.file_helpers.validate_pdf_file"""

    def test_valid_pdf_file(self, tmp_path):
        """A small, correctly-named .pdf file should pass validation."""
        pdf_file = tmp_path / "resume.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")
        is_valid, error = validate_pdf_file(str(pdf_file))
        assert is_valid is True
        assert error is None

    def test_nonexistent_file_fails(self):
        """A non-existent file path should fail validation."""
        is_valid, error = validate_pdf_file("/nonexistent/path/resume.pdf")
        assert is_valid is False
        assert error is not None

    def test_wrong_extension_fails(self, tmp_path):
        """A file with a non-PDF extension should fail validation."""
        txt_file = tmp_path / "resume.txt"
        txt_file.write_text("not a pdf")
        is_valid, error = validate_pdf_file(str(txt_file))
        assert is_valid is False
        assert error is not None

    def test_file_too_large_fails(self, tmp_path):
        """A file exceeding max_size_bytes should fail validation."""
        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b"0" * 1000)
        is_valid, error = validate_pdf_file(str(large_file), max_size_bytes=500)
        assert is_valid is False
        assert "size" in error.lower()


class TestGetFileSizeMb:
    """Tests for src.utils.file_helpers.get_file_size_mb"""

    def test_returns_correct_size(self, tmp_path):
        """Should return file size in MB accurately."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"0" * 1024 * 1024)  # 1 MB
        size = get_file_size_mb(str(test_file))
        assert abs(size - 1.0) < 0.01

    def test_nonexistent_file_returns_zero(self):
        """Non-existent file should return 0.0 without raising."""
        size = get_file_size_mb("/no/such/file.pdf")
        assert size == 0.0
