"""
src.evaluation.metrics
=======================
Computes and reports classification performance metrics.

Usage:
    from src.evaluation.metrics import compute_classification_report, compute_roc_auc
"""

import logging
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_report(
    y_true: list[int],
    y_pred: list[int],
    y_prob: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Compute a comprehensive classification metrics report.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (for AUC).

    Returns:
        Dictionary of metric names to computed values.
    """

    report: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Not Suitable", "Suitable"]
        ),
    }

    if y_prob is not None:
        try:
            report["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)
            report["roc_auc"] = None

    for key, value in report.items():
        if isinstance(value, float):
            logger.info("%-20s %.4f", key + ":", value)

    return report


def compute_roc_auc(y_true: list[int], y_prob: list[float]) -> float:
    """Compute the ROC-AUC score.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        ROC-AUC score as a float in [0.0, 1.0].
    """
    score = float(roc_auc_score(y_true, y_prob))
    logger.info("ROC-AUC Score: %.4f", score)
    return score





# TODO: Add calibration analysis (reliability diagrams) via sklearn.calibration
# TODO: Add per-category metrics for multi-class extension (job category classification)
