"""
tests/test_inference.py
========================
Unit tests for the inference pipeline:
    - Cosine similarity scoring
    - Score-to-label conversion
    - PredictionResult dataclass
    - Classifier prediction logic

Run with:
    pytest tests/test_inference.py -v
"""

import numpy as np
import pytest

from src.inference.similarity_scorer import (
    SCORE_BANDS,
    compute_cosine_similarity,
    score_to_label,
)
from src.inference.classifier import PredictionResult, LABEL_MAP


# ─── Cosine Similarity Tests ───────────────────────────────────────────────────


class TestCosineSimilarity:
    """Tests for src.inference.similarity_scorer.compute_cosine_similarity"""

    def test_identical_vectors_return_one(self):
        """Identical vectors should have cosine similarity of 1.0."""
        vec = np.array([1.0, 0.5, 0.3, 0.8])
        score = compute_cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors (no shared components) should return ~0.0."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        score = compute_cosine_similarity(vec_a, vec_b)
        assert abs(score) < 1e-6

    def test_score_is_bounded_0_to_1(self):
        """Cosine similarity score must be in [0.0, 1.0]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.random(100)
            b = rng.random(100)
            score = compute_cosine_similarity(a, b)
            assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        """compute_cosine_similarity should return a Python float."""
        vec_a = np.array([0.5, 0.3, 0.8])
        vec_b = np.array([0.1, 0.9, 0.2])
        score = compute_cosine_similarity(vec_a, vec_b)
        assert isinstance(score, float)

    def test_dimension_mismatch_raises_value_error(self):
        """Mismatched vector dimensions should raise ValueError."""
        vec_a = np.array([1.0, 0.0, 0.5])
        vec_b = np.array([1.0, 0.5])
        with pytest.raises(ValueError, match="dimension mismatch"):
            compute_cosine_similarity(vec_a, vec_b)

    def test_1d_and_2d_inputs_equivalent(self):
        """1D and 2D (reshaped) vectors should produce the same score."""
        vec_a = np.array([0.4, 0.6, 0.2])
        vec_b = np.array([0.3, 0.7, 0.1])
        score_1d = compute_cosine_similarity(vec_a, vec_b)
        score_2d = compute_cosine_similarity(
            vec_a.reshape(1, -1), vec_b.reshape(1, -1)
        )
        assert abs(score_1d - score_2d) < 1e-9

    def test_partial_overlap_between_zero_and_one(self):
        """Vectors with partial overlap should score between 0 and 1."""
        vec_a = np.array([1.0, 1.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 1.0, 0.0])
        score = compute_cosine_similarity(vec_a, vec_b)
        assert 0.0 < score < 1.0


# ─── Score-to-Label Tests ──────────────────────────────────────────────────────


class TestScoreToLabel:
    """Tests for src.inference.similarity_scorer.score_to_label"""

    def test_excellent_threshold(self):
        """Score >= 0.80 should be labeled 'Excellent'."""
        assert score_to_label(0.85) == "Excellent"
        assert score_to_label(0.80) == "Excellent"

    def test_good_threshold(self):
        """Score in [0.65, 0.80) should be labeled 'Good'."""
        assert score_to_label(0.70) == "Good"
        assert score_to_label(0.65) == "Good"

    def test_fair_threshold(self):
        """Score in [0.50, 0.65) should be labeled 'Fair'."""
        assert score_to_label(0.55) == "Fair"

    def test_poor_threshold(self):
        """Score below 0.50 should be labeled 'Poor'."""
        assert score_to_label(0.30) == "Poor"
        assert score_to_label(0.00) == "Poor"

    def test_returns_string(self):
        """score_to_label should always return a string."""
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = score_to_label(score)
            assert isinstance(result, str)

    def test_all_labels_are_valid(self):
        """All returned labels should be in the defined SCORE_BANDS keys."""
        valid_labels = set(SCORE_BANDS.keys())
        test_scores = [0.0, 0.1, 0.3, 0.5, 0.55, 0.65, 0.70, 0.80, 0.90, 1.0]
        for score in test_scores:
            label = score_to_label(score)
            assert label in valid_labels, f"Unexpected label '{label}' for score {score}"


# ─── PredictionResult Tests ────────────────────────────────────────────────────


class TestPredictionResult:
    """Tests for src.inference.classifier.PredictionResult dataclass"""

    def test_suitable_prediction(self):
        """PredictionResult with prediction=1 should have label 'Suitable'."""
        result = PredictionResult(
            label="Suitable",
            prediction=1,
            probability=0.78,
            confidence=78.0,
        )
        assert result.label == "Suitable"
        assert result.prediction == 1
        assert result.probability == pytest.approx(0.78)
        assert result.confidence == pytest.approx(78.0)

    def test_not_suitable_prediction(self):
        """PredictionResult with prediction=0 should have label 'Not Suitable'."""
        result = PredictionResult(
            label="Not Suitable",
            prediction=0,
            probability=0.32,
            confidence=32.0,
        )
        assert result.label == "Not Suitable"
        assert result.prediction == 0

    def test_label_map_contains_expected_keys(self):
        """LABEL_MAP should map 0 → 'Not Suitable' and 1 → 'Suitable'."""
        assert LABEL_MAP[0] == "Not Suitable"
        assert LABEL_MAP[1] == "Suitable"
