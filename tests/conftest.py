"""
tests/conftest.py
==================
Shared pytest fixtures for the AI Resume Analyzer test suite.

Fixtures defined here are automatically available to all test modules
without explicit imports.
"""

import pytest


@pytest.fixture
def sample_resume_text() -> str:
    """Return a minimal pre-processed resume text sample for testing."""
    return (
        "experienced software engineer python machine learning "
        "scikit-learn tensorflow deep learning data pipeline "
        "deployment aws docker five years experience backend api"
    )


@pytest.fixture
def sample_jd_text() -> str:
    """Return a minimal pre-processed job description text sample for testing."""
    return (
        "looking for machine learning engineer python scikit-learn "
        "tensorflow experience required deployment cloud aws docker "
        "strong communication skills team collaboration"
    )


@pytest.fixture
def mismatched_resume_text() -> str:
    """Return a resume text with low similarity to the sample JD."""
    return (
        "marketing manager creative advertising social media "
        "branding campaign management brand strategy communications "
        "public relations consumer research"
    )


@pytest.fixture
def empty_text() -> str:
    """Return an empty string for edge case testing."""
    return ""


@pytest.fixture
def whitespace_text() -> str:
    """Return a whitespace-only string for edge case testing."""
    return "   \n\t  "
