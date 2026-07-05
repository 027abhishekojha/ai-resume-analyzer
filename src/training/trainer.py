"""
src.training.trainer
=====================
Trains the ML classifier for candidate suitability prediction.

Supports:
    - Logistic Regression
    - Random Forest Classifier
    - Support Vector Machine (SVM)

Usage:
    from src.training.trainer import train_classifier, cross_validate_model
"""

import logging
import time
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.linear_model import LogisticRegression  # type: ignore[import]
from sklearn.model_selection import StratifiedKFold, cross_validate  # type: ignore[import]
from sklearn.svm import SVC  # type: ignore[import]

logger = logging.getLogger(__name__)

# Supported algorithms
SUPPORTED_ALGORITHMS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
}


def build_classifier(
    algorithm: str = "logistic_regression",
    hyperparams: Optional[dict[str, Any]] = None,
) -> Any:
    """Instantiate an ML classifier with the given algorithm and hyperparameters.

    Args:
        algorithm: Name of the algorithm. One of: "logistic_regression",
                   "random_forest", "svm".
        hyperparams: Dictionary of hyperparameter overrides.

    Returns:
        An un-fitted scikit-learn estimator.

    Raises:
        ValueError: If the algorithm name is not recognized.
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: '{algorithm}'. "
            f"Choose from: {list(SUPPORTED_ALGORITHMS.keys())}"
        )

    params = hyperparams or {}

    # Default configurations per algorithm
    defaults: dict[str, dict[str, Any]] = {
        "logistic_regression": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "svm": {"kernel": "rbf", "C": 1.0, "probability": True, "random_state": 42},
    }

    merged_params = {**defaults[algorithm], **params}
    classifier = SUPPORTED_ALGORITHMS[algorithm](**merged_params)

    logger.info(
        "Classifier built: %s | Params: %s", algorithm, merged_params
    )
    return classifier


def train_classifier(
    X_train: np.ndarray,
    y_train: list[int],
    algorithm: str = "logistic_regression",
    hyperparams: Optional[dict[str, Any]] = None,
) -> Any:
    """Fit an ML classifier on TF-IDF feature vectors.

    Args:
        X_train: Training feature matrix of shape (n_samples, n_features).
        y_train: Training labels (0 or 1).
        algorithm: Classifier algorithm name.
        hyperparams: Optional hyperparameter overrides.

    Returns:
        The fitted scikit-learn estimator.
    """
    classifier = build_classifier(algorithm, hyperparams)

    logger.info(
        "Training %s on %d samples...", algorithm, len(y_train)
    )
    start_time = time.perf_counter()
    classifier.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    logger.info("Training complete in %.2f seconds.", elapsed)
    return classifier


def cross_validate_model(
    X: np.ndarray,
    y: list[int],
    algorithm: str = "logistic_regression",
    n_folds: int = 5,
    hyperparams: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Perform stratified k-fold cross-validation.

    Args:
        X: Full feature matrix.
        y: Full label list.
        algorithm: Classifier algorithm name.
        n_folds: Number of CV folds.
        hyperparams: Optional hyperparameter overrides.

    Returns:
        Dictionary of CV metric scores (accuracy, precision, recall, f1).
    """
    classifier = build_classifier(algorithm, hyperparams)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    logger.info("Running %d-fold cross-validation...", n_folds)
    cv_results = cross_validate(
        classifier, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    summary: dict[str, Any] = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        summary[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
        }
        logger.info(
            "CV %s: %.4f ± %.4f", metric, summary[metric]["mean"], summary[metric]["std"]
        )

    return summary


# TODO: Add hyperparameter search via GridSearchCV or Optuna
# TODO: Add MLflow experiment tracking integration
