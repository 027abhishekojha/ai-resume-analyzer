"""
src.feature_engineering.tfidf_vectorizer
==========================================
Wraps scikit-learn's TfidfVectorizer for the resume analysis pipeline.

Provides utilities to:
- Fit a vectorizer on a corpus of documents
- Transform new documents into TF-IDF feature matrices
- Save and load fitted vectorizers via Joblib

Usage:
    from src.feature_engineering.tfidf_vectorizer import (
        build_vectorizer,
        fit_vectorizer,
        transform_documents,
        save_vectorizer,
        load_vectorizer,
    )
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]

logger = logging.getLogger(__name__)


def build_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Instantiate a TfidfVectorizer with the given hyperparameters.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: Range of n-gram sizes (min_n, max_n).
        min_df: Minimum document frequency for a term.
        max_df: Maximum document frequency (as fraction) for a term.
        sublinear_tf: Apply log(1 + tf) scaling.

    Returns:
        An un-fitted TfidfVectorizer instance.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
    )
    logger.debug(
        "TfidfVectorizer built: max_features=%d, ngram=%s",
        max_features,
        ngram_range,
    )
    return vectorizer


def fit_vectorizer(
    corpus: list[str],
    vectorizer: Optional[TfidfVectorizer] = None,
    **kwargs,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a corpus of pre-processed text documents.

    Args:
        corpus: List of pre-processed text documents (one string per document).
        vectorizer: An existing TfidfVectorizer to fit. If None, a new one is
                    created using `kwargs` as hyperparameter overrides.
        **kwargs: Hyperparameter overrides forwarded to `build_vectorizer`.

    Returns:
        The fitted TfidfVectorizer.

    Raises:
        ValueError: If corpus is empty or contains no valid documents.
    """
    if not corpus:
        raise ValueError("Cannot fit vectorizer on an empty corpus.")

    if vectorizer is None:
        vectorizer = build_vectorizer(**kwargs)

    vectorizer.fit(corpus)
    logger.info(
        "Vectorizer fitted on %d documents. Vocabulary size: %d.",
        len(corpus),
        len(vectorizer.vocabulary_),
    )
    return vectorizer


def transform_documents(
    documents: list[str],
    vectorizer: TfidfVectorizer,
) -> np.ndarray:
    """Transform a list of documents into a TF-IDF feature matrix.

    Args:
        documents: List of pre-processed text strings.
        vectorizer: A **fitted** TfidfVectorizer instance.

    Returns:
        Dense numpy array of shape (n_documents, n_features).

    Raises:
        RuntimeError: If the vectorizer has not been fitted.
    """
    # Check if vectorizer is fitted
    if not hasattr(vectorizer, "vocabulary_"):
        raise RuntimeError(
            "The vectorizer has not been fitted yet. "
            "Call fit_vectorizer() before transform_documents()."
        )

    sparse_matrix = vectorizer.transform(documents)
    dense_matrix = sparse_matrix.toarray()

    logger.debug(
        "Transformed %d documents to shape %s.",
        len(documents),
        dense_matrix.shape,
    )
    return dense_matrix


def save_vectorizer(vectorizer: TfidfVectorizer, save_path: str) -> None:
    """Serialize and save a fitted TF-IDF vectorizer to disk.

    Args:
        vectorizer: A fitted TfidfVectorizer instance to save.
        save_path: Destination file path (e.g., "models/vectorizers/tfidf.joblib").
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, save_path)
    logger.info("Vectorizer saved to: %s", save_path)


def load_vectorizer(load_path: str) -> TfidfVectorizer:
    """Load a previously serialized TF-IDF vectorizer from disk.

    Args:
        load_path: Path to the saved .joblib vectorizer file.

    Returns:
        The deserialized TfidfVectorizer.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {load_path}. "
            "Train the model first using: python scripts/train_model.py"
        )
    vectorizer = joblib.load(load_path)
    logger.info("Vectorizer loaded from: %s", load_path)
    return vectorizer


# TODO: Evaluate sentence-transformers (BERT) as a replacement for TF-IDF
# TODO: Add support for sparse matrix operations to reduce memory usage on large corpora
