"""
src.inference.classifier
=========================
Loads a trained ML classification model and predicts candidate suitability.

The classifier takes a feature vector (TF-IDF representation of the
concatenated resume + JD text) and outputs a binary prediction:
    - 1 (Suitable): The candidate matches the job requirements.
    - 0 (Not Suitable): The candidate is not a strong match.

Usage:
    from src.inference.classifier import load_model, predict_suitability

    model = load_model("models/trained/classifier.joblib")
    result = predict_suitability(model, feature_vector)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Label mapping for binary classifier output
LABEL_MAP: dict[int, str] = {
    1: "Suitable",
    0: "Not Suitable",
}


@dataclass
class PredictionResult:
    """Container for classifier prediction output.

    Attributes:
        label: Human-readable prediction label ("Suitable" or "Not Suitable").
        prediction: Binary prediction (1 = Suitable, 0 = Not Suitable).
        probability: Probability score for the positive class [0.0, 1.0].
        confidence: Confidence percentage (probability * 100).
    """

    label: str
    prediction: int
    probability: float
    confidence: float


def load_model(model_path: str) -> object:
    """Load a trained scikit-learn classifier from a Joblib file.

    Args:
        model_path: Path to the serialized model file (.joblib).

    Returns:
        The deserialized scikit-learn estimator.

    Raises:
        FileNotFoundError: If the model file does not exist at model_path.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please train the model first: python scripts/train_model.py"
        )
    model = joblib.load(path)
    logger.info("Model loaded from: %s [type: %s]", model_path, type(model).__name__)
    return model


def predict_suitability(
    model: object,
    feature_vector: np.ndarray,
    threshold: float = 0.60,
) -> PredictionResult:
    """Predict whether a candidate is suitable for a job using the classifier.

    Args:
        model: A fitted scikit-learn classifier with predict_proba support.
        feature_vector: A 1D or 2D numpy array of TF-IDF features.
        threshold: Probability threshold above which a candidate is "Suitable".

    Returns:
        A PredictionResult dataclass with label, prediction, and probability.

    Raises:
        AttributeError: If the model does not support predict_proba.
    """
    # Ensure 2D input
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    # Get class probabilities
    probabilities = model.predict_proba(feature_vector)[0]  # type: ignore[union-attr]
    positive_prob = float(probabilities[1])

    # Apply threshold instead of argmax for tunable sensitivity
    prediction = int(positive_prob >= threshold)
    label = LABEL_MAP[prediction]
    confidence = positive_prob * 100

    logger.info(
        "Prediction: %s (probability=%.4f, threshold=%.2f)",
        label,
        positive_prob,
        threshold,
    )

    return PredictionResult(
        label=label,
        prediction=prediction,
        probability=positive_prob,
        confidence=confidence,
    )


def save_model(model: object, save_path: str) -> None:
    """Serialize and save a trained model to disk.

    Args:
        model: A trained scikit-learn estimator.
        save_path: Destination file path (e.g., "models/trained/classifier.joblib").
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    logger.info("Model saved to: %s", save_path)


# TODO: Add model versioning metadata (timestamp, training data hash, accuracy)
# TODO: Implement SHAP-based feature importance for explainable predictions
