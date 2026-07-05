"""
scripts/evaluate_model.py
==========================
CLI script to evaluate a trained ML classifier.

Loads a saved model and vectorizer, runs predictions on a test dataset,
and generates a comprehensive evaluation report including:
    - Classification metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
    - Confusion matrix
    - Per-class performance

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model models/trained/classifier.joblib
    python scripts/evaluate_model.py --output outputs/evaluation/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import compute_classification_report
from src.evaluation.visualizer import plot_confusion_matrix
from src.feature_engineering.tfidf_vectorizer import load_vectorizer, transform_documents
from src.inference.classifier import load_model
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.tokenizer import tokenize, tokens_to_string
from src.training.data_loader import load_training_data, split_dataset
from src.utils.config_loader import load_config
from src.utils.file_helpers import ensure_directory_exists
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the AI Resume Analyzer trained classifier.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/trained/classifier.joblib",
        help="Path to the trained classifier (.joblib file).",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="models/vectorizers/tfidf_vectorizer.joblib",
        help="Path to the fitted TF-IDF vectorizer (.joblib file).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the evaluation dataset CSV (default: from config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/",
        help="Directory to save evaluation report and plots.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation pipeline entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AI Resume Analyzer — Model Evaluation Pipeline")
    logger.info("=" * 60)

    # ── Load Configuration ─────────────────────────────────────────────────
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    dataset_path = args.dataset or train_cfg.get(
        "dataset_path", "datasets/processed/processed_resumes.csv"
    )

    # ── Load Artifacts ─────────────────────────────────────────────────────
    logger.info("Loading model from: %s", args.model)
    model = load_model(args.model)

    logger.info("Loading vectorizer from: %s", args.vectorizer)
    vectorizer = load_vectorizer(args.vectorizer)

    # ── Load Dataset ───────────────────────────────────────────────────────
    logger.info("Loading evaluation dataset from: %s", dataset_path)
    try:
        df = load_training_data(dataset_path)
    except FileNotFoundError:
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    # Use the full dataset as evaluation set (or split if desired)
    _, X_test_raw, _, y_test = split_dataset(df, test_size=0.20)

    # ── Preprocess ─────────────────────────────────────────────────────────
    logger.info("Preprocessing evaluation texts...")
    X_test_texts = [
        tokens_to_string(tokenize(clean_text(t))) for t in X_test_raw
    ]
    X_test = transform_documents(X_test_texts, vectorizer)

    # ── Predict ────────────────────────────────────────────────────────────
    logger.info("Running predictions on %d samples...", len(X_test))
    y_pred = model.predict(X_test).tolist()
    y_prob = model.predict_proba(X_test)[:, 1].tolist()

    # ── Compute Metrics ────────────────────────────────────────────────────
    logger.info("Computing evaluation metrics...")
    report = compute_classification_report(y_test, y_pred, y_prob)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(report.get("classification_report", "N/A"))
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        val = report.get(key)
        if val is not None:
            print(f"{key:<20}: {val:.4f}")

    # ── Save Outputs ───────────────────────────────────────────────────────
    output_dir = ensure_directory_exists(args.output)

    # Save JSON report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        # Remove non-serializable keys
        serializable = {k: v for k, v in report.items() if k != "classification_report"}
        json.dump(serializable, f, indent=2)
    logger.info("Evaluation report saved to: %s", report_path)

    # Save confusion matrix plot
    cm_fig = plot_confusion_matrix(
        report["confusion_matrix"],
        title="Confusion Matrix — AI Resume Analyzer",
    )
    cm_path = output_dir / "confusion_matrix.png"
    cm_fig.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    logger.info("Confusion matrix saved to: %s", cm_path)

    logger.info("Evaluation pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
