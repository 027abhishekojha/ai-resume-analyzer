"""
src.inference.similarity_scorer
================================
Computes cosine similarity between TF-IDF vector representations of
a resume and a job description.

Usage:
    from src.inference.similarity_scorer import compute_cosine_similarity, score_to_label

    score = compute_cosine_similarity(resume_vector, jd_vector)
    label = score_to_label(score)
"""

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]

logger = logging.getLogger(__name__)

# Score band thresholds (configurable via configs/config.yaml in production)
SCORE_BANDS: dict[str, float] = {
    "Excellent": 0.80,
    "Good": 0.65,
    "Fair": 0.50,
    "Poor": 0.00,
}


def compute_cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """Compute the cosine similarity between two TF-IDF feature vectors.

    Cosine similarity measures the angle between two vectors in
    high-dimensional space, ranging from 0 (orthogonal) to 1 (identical).

    Args:
        vec_a: First feature vector of shape (n_features,) or (1, n_features).
        vec_b: Second feature vector of shape (n_features,) or (1, n_features).

    Returns:
        Cosine similarity score as a float in [0.0, 1.0].

    Raises:
        ValueError: If input vectors have mismatched feature dimensions.
    """
    # Ensure 2D for sklearn's cosine_similarity
    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Vector dimension mismatch: {a.shape[1]} vs {b.shape[1]}. "
            "Both vectors must be transformed with the same vectorizer."
        )

    score = float(cosine_similarity(a, b)[0][0])

    # Clamp to [0, 1] to handle floating point edge cases
    score = max(0.0, min(1.0, score))

    logger.debug("Cosine similarity computed: %.4f", score)
    return score


def compute_similarity_from_texts(
    resume_text: str,
    jd_text: str,
    vectorizer,  # TfidfVectorizer — avoid circular import with type hint
) -> float:
    """Convenience function: vectorize two texts and compute their similarity.

    Args:
        resume_text: Pre-processed resume text.
        jd_text: Pre-processed job description text.
        vectorizer: A fitted TfidfVectorizer instance.

    Returns:
        Cosine similarity score in [0.0, 1.0].
    """
    vectors = vectorizer.transform([resume_text, jd_text]).toarray()
    resume_vec = vectors[0]
    jd_vec = vectors[1]
    return compute_cosine_similarity(resume_vec, jd_vec)


def score_to_label(
    score: float,
    bands: dict[str, float] = SCORE_BANDS,
) -> str:
    """Convert a numeric similarity score to a human-readable label.

    Args:
        score: Cosine similarity score in [0.0, 1.0].
        bands: Threshold dictionary mapping label → minimum score.
               Expected keys: "Excellent", "Good", "Fair", "Poor".

    Returns:
        Human-readable label string (e.g., "Good", "Excellent").
    """
    for label, threshold in sorted(
        bands.items(), key=lambda x: x[1], reverse=True
    ):
        if score >= threshold:
            logger.debug("Score %.4f classified as '%s'.", score, label)
            return label
    return "Poor"


# TODO: Explore BERT-based sentence similarity as a complement to TF-IDF cosine score
# TODO: Implement weighted similarity scoring (sections: skills 40%, experience 40%, etc.)
