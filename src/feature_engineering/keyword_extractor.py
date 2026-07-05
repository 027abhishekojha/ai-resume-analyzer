"""
src.feature_engineering.keyword_extractor
==========================================
Extracts top keywords from TF-IDF vectorized documents and
identifies keyword gaps between a resume and a job description.

Usage:
    from src.feature_engineering.keyword_extractor import (
        get_top_keywords,
        find_keyword_gaps,
    )
"""

import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]

logger = logging.getLogger(__name__)


def get_top_keywords(
    text: str,
    vectorizer: TfidfVectorizer,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Extract the top N keywords from a document by TF-IDF weight.

    Args:
        text: A single pre-processed text document.
        vectorizer: A **fitted** TfidfVectorizer.
        top_n: Number of top keywords to return.

    Returns:
        List of (keyword, tfidf_score) tuples, sorted descending by score.
    """
    if not text.strip():
        logger.warning("get_top_keywords called with empty text.")
        return []

    feature_names: np.ndarray = np.array(vectorizer.get_feature_names_out())
    tfidf_matrix = vectorizer.transform([text]).toarray()[0]

    # Get indices of top N scores
    top_indices = np.argsort(tfidf_matrix)[::-1][:top_n]
    top_keywords = [
        (feature_names[i], float(tfidf_matrix[i]))
        for i in top_indices
        if tfidf_matrix[i] > 0
    ]

    logger.debug("Extracted %d top keywords from document.", len(top_keywords))
    return top_keywords


def find_keyword_gaps(
    resume_text: str,
    jd_text: str,
    vectorizer: TfidfVectorizer,
    top_n: int = 20,
    min_jd_score: float = 0.01,
) -> list[str]:
    """Identify keywords present in the JD but absent or weak in the resume.

    These "gap keywords" represent terms the candidate should consider
    adding to their resume to improve ATS compatibility.

    Args:
        resume_text: Pre-processed resume text.
        jd_text: Pre-processed job description text.
        vectorizer: A fitted TfidfVectorizer.
        top_n: Number of top JD keywords to consider.
        min_jd_score: Minimum TF-IDF score in JD for a term to be considered.

    Returns:
        List of keyword strings that are in the JD but missing from the resume.
    """
    jd_keywords = get_top_keywords(jd_text, vectorizer, top_n=top_n)
    resume_keywords_set = {kw for kw, _ in get_top_keywords(
        resume_text, vectorizer, top_n=top_n * 3
    )}

    gaps: list[str] = []
    for keyword, score in jd_keywords:
        if score >= min_jd_score and keyword not in resume_keywords_set:
            gaps.append(keyword)

    logger.info(
        "Keyword gap analysis: %d/%d JD keywords missing from resume.",
        len(gaps),
        len(jd_keywords),
    )
    return gaps


def get_keyword_overlap(
    resume_text: str,
    jd_text: str,
    vectorizer: TfidfVectorizer,
    top_n: int = 20,
) -> list[str]:
    """Find keywords that appear in both the resume and the job description.

    Args:
        resume_text: Pre-processed resume text.
        jd_text: Pre-processed job description text.
        vectorizer: A fitted TfidfVectorizer.
        top_n: Number of top keywords to compare from each document.

    Returns:
        List of keyword strings present in both the resume and JD.
    """
    resume_kw_set = {kw for kw, _ in get_top_keywords(resume_text, vectorizer, top_n)}
    jd_kw_set = {kw for kw, _ in get_top_keywords(jd_text, vectorizer, top_n)}
    overlap = list(resume_kw_set & jd_kw_set)

    logger.debug("Keyword overlap: %d matching terms.", len(overlap))
    return sorted(overlap)


# TODO: Add skill taxonomy mapping (e.g., "ML" → "machine learning") for fuzzy matching
# TODO: Implement Named Entity Recognition to extract skills, titles, and technologies
