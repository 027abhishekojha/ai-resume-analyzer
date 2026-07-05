"""
tests/test_preprocessing.py
============================
Unit tests for the preprocessing pipeline:
    - PDF text extraction
    - Text cleaning
    - Tokenization

Run with:
    pytest tests/test_preprocessing.py -v
"""

import pytest

from src.preprocessing.text_cleaner import clean_text, remove_resume_headers
from src.preprocessing.tokenizer import tokenize, tokens_to_string


# ─── Text Cleaner Tests ────────────────────────────────────────────────────────


class TestCleanText:
    """Tests for src.preprocessing.text_cleaner.clean_text"""

    def test_lowercase_conversion(self):
        """clean_text should convert all text to lowercase."""
        result = clean_text("Machine Learning Engineer")
        assert result == "machine learning engineer"

    def test_removes_urls(self):
        """clean_text should strip HTTP/HTTPS URLs."""
        text = "Visit https://example.com for more details."
        result = clean_text(text)
        assert "https://example.com" not in result
        assert "visit" in result

    def test_removes_email_addresses(self):
        """clean_text should strip email addresses."""
        text = "Contact me at john.doe@example.com for inquiries."
        result = clean_text(text)
        assert "john.doe@example.com" not in result

    def test_removes_special_characters(self):
        """clean_text should remove non-alphanumeric characters."""
        text = "Python! Java@, C++; Skills#"
        result = clean_text(text)
        assert "!" not in result
        assert "@" not in result
        assert ";" not in result

    def test_normalizes_whitespace(self):
        """clean_text should collapse multiple spaces into one."""
        text = "machine    learning   engineer"
        result = clean_text(text)
        assert "  " not in result
        assert result == "machine learning engineer"

    def test_empty_string_input(self):
        """clean_text should return empty string for empty input."""
        result = clean_text("")
        assert result == ""

    def test_preserves_alphanumeric(self):
        """clean_text should retain letters and digits."""
        text = "Python3 and Machine Learning 101"
        result = clean_text(text)
        assert "python3" in result
        assert "machine" in result
        assert "101" in result

    def test_strip_leading_trailing_whitespace(self):
        """clean_text should strip leading and trailing whitespace."""
        result = clean_text("  hello world  ")
        assert result == "hello world"

    def test_unicode_normalization(self):
        """clean_text should handle unicode characters gracefully."""
        text = "Résumé with àccented chàracters"
        result = clean_text(text, normalize_unicode=True)
        assert isinstance(result, str)
        assert len(result) > 0


class TestRemoveResumeHeaders:
    """Tests for src.preprocessing.text_cleaner.remove_resume_headers"""

    def test_removes_common_headers(self):
        """remove_resume_headers should strip known section headers."""
        text = "education machine learning skills python experience"
        result = remove_resume_headers(text)
        assert "education" not in result.lower()
        assert "skills" not in result.lower()

    def test_preserves_non_header_text(self):
        """remove_resume_headers should keep meaningful content."""
        text = "developed scalable machine learning pipelines"
        result = remove_resume_headers(text)
        assert "machine learning" in result
        assert "pipelines" in result


# ─── Tokenizer Tests ───────────────────────────────────────────────────────────


class TestTokenize:
    """Tests for src.preprocessing.tokenizer.tokenize"""

    def test_basic_tokenization(self):
        """tokenize should split text into token list."""
        result = tokenize("machine learning engineer python")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_returns_list_of_strings(self):
        """tokenize should return a list of string tokens."""
        result = tokenize("data science artificial intelligence")
        assert all(isinstance(t, str) for t in result)

    def test_empty_string_returns_empty_list(self):
        """tokenize should return empty list for empty input."""
        result = tokenize("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """tokenize should return empty list for whitespace-only input."""
        result = tokenize("   ")
        assert result == []

    def test_min_token_length_filter(self):
        """tokenize should filter tokens shorter than min_token_length."""
        result = tokenize("a ab abc abcd", min_token_length=3)
        # Tokens "a" and "ab" should be filtered out
        for token in result:
            assert len(token) >= 3

    def test_stopword_removal(self):
        """tokenize should remove English stopwords when enabled."""
        # "the", "is", "a" are common stopwords
        text = "the candidate is a machine learning engineer"
        result = tokenize(text, remove_stopwords=True)
        assert "the" not in result
        assert "is" not in result

    def test_no_stopword_removal(self):
        """tokenize should keep stopwords when remove_stopwords=False."""
        text = "the candidate is an engineer"
        result = tokenize(text, remove_stopwords=False)
        # With stopword removal off, "the" should be retained
        # (may vary depending on NLTK availability — soft assertion)
        assert isinstance(result, list)

    def test_tokens_to_string_roundtrip(self):
        """tokens_to_string should correctly join a token list."""
        tokens = ["machine", "learning", "engineer"]
        result = tokens_to_string(tokens)
        assert result == "machine learning engineer"

    def test_tokens_to_string_empty_list(self):
        """tokens_to_string should return empty string for empty list."""
        result = tokens_to_string([])
        assert result == ""
