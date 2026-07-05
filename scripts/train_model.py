"""
scripts/train_model.py
=======================
CLI script to train the ML classifier for the AI Resume Analyzer.

This script orchestrates the full training pipeline:
    1. Load configuration
    2. Load and validate training dataset
    3. Preprocess text (clean + tokenize)
    4. Fit TF-IDF vectorizer on training corpus
    5. Transform features
    6. Train classifier
    7. Evaluate on test set
    8. Save model and vectorizer artifacts

Usage:
    python scripts/train_model.py --config configs/config.yaml
    python scripts/train_model.py --algorithm random_forest --test-size 0.2
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import compute_classification_report
from src.feature_engineering.tfidf_vectorizer import (
    fit_vectorizer,
    save_vectorizer,
    transform_documents,
)
from src.inference.classifier import save_model
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.tokenizer import tokenize, tokens_to_string
from src.training.data_loader import load_training_data, split_dataset
from src.training.trainer import cross_validate_model, train_classifier
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the AI Resume Analyzer ML classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py
  python scripts/train_model.py --algorithm random_forest
  python scripts/train_model.py --config configs/config.yaml --test-size 0.25
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration YAML file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["logistic_regression", "random_forest", "svm"],
        default=None,
        help="Classifier algorithm override (default: from config).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of data for testing (default: from config).",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run k-fold cross-validation before final training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without saving model artifacts.",
    )
    return parser.parse_args()


def preprocess_texts(texts: list[str]) -> list[str]:
    """Apply full preprocessing pipeline to a list of raw text documents.

    Args:
        texts: List of raw document strings.

    Returns:
        List of preprocessed document strings, ready for TF-IDF vectorization.
    """
    processed = []
    for text in texts:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        processed.append(tokens_to_string(tokens))
    return processed


def main() -> None:
    """Main training pipeline entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AI Resume Analyzer — Model Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load Configuration ─────────────────────────────────────────
    logger.info("Loading configuration from: %s", args.config)
    config = load_config(args.config)

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    feat_cfg = config.get("feature_engineering", {})

    algorithm = args.algorithm or model_cfg.get("algorithm", "logistic_regression")
    test_size = args.test_size or train_cfg.get("test_size", 0.20)
    dataset_path = train_cfg.get("dataset_path", "datasets/processed/processed_resumes.csv")
    classifier_path = model_cfg.get("classifier_path", "models/trained/classifier.joblib")
    vectorizer_path = model_cfg.get("vectorizer_path", "models/vectorizers/tfidf_vectorizer.joblib")

    logger.info("Algorithm : %s", algorithm)
    logger.info("Test size : %.2f", test_size)
    logger.info("Dataset   : %s", dataset_path)

    # ── Step 2: Load Dataset ───────────────────────────────────────────────
    logger.info("Loading training dataset...")
    # TODO: Replace with actual dataset path after running download_dataset.py
    try:
        df = load_training_data(dataset_path)
    except FileNotFoundError:
        logger.error(
            "Dataset not found at '%s'. "
            "Run: python scripts/download_dataset.py",
            dataset_path,
        )
        sys.exit(1)

    # ── Step 3: Split Dataset ──────────────────────────────────────────────
    logger.info("Splitting dataset...")
    X_train_raw, X_test_raw, y_train, y_test = split_dataset(
        df, test_size=test_size, random_state=train_cfg.get("random_state", 42)
    )

    # ── Step 4: Preprocess ─────────────────────────────────────────────────
    logger.info("Preprocessing text (clean + tokenize)...")
    X_train_text = preprocess_texts(X_train_raw)
    X_test_text = preprocess_texts(X_test_raw)

    # ── Step 5: Fit Vectorizer ─────────────────────────────────────────────
    logger.info("Fitting TF-IDF vectorizer on training corpus...")
    vectorizer = fit_vectorizer(
        corpus=X_train_text,
        max_features=feat_cfg.get("max_features", 5000),
        ngram_range=tuple(feat_cfg.get("ngram_range", [1, 2])),
        min_df=feat_cfg.get("min_df", 2),
        max_df=feat_cfg.get("max_df", 0.95),
        sublinear_tf=feat_cfg.get("sublinear_tf", True),
    )

    # ── Step 6: Transform Features ─────────────────────────────────────────
    logger.info("Transforming training and test features...")
    X_train = transform_documents(X_train_text, vectorizer)
    X_test = transform_documents(X_test_text, vectorizer)

    # ── Step 7: Cross-Validation (optional) ───────────────────────────────
    if args.cross_validate:
        logger.info("Running cross-validation...")
        import numpy as np

        X_all = transform_documents(X_train_text + X_test_text, vectorizer)
        y_all = y_train + y_test
        cv_results = cross_validate_model(
            X_all, y_all,
            algorithm=algorithm,
            n_folds=train_cfg.get("cross_validation_folds", 5),
        )
        logger.info("Cross-validation summary: %s", cv_results)

    # ── Step 8: Train Classifier ───────────────────────────────────────────
    logger.info("Training %s classifier...", algorithm)
    hyperparams = train_cfg.get(algorithm, {})
    model = train_classifier(X_train, y_train, algorithm=algorithm, hyperparams=hyperparams)

    # ── Step 9: Evaluate ───────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1].tolist()
    report = compute_classification_report(y_test, y_pred.tolist(), y_prob)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(report.get("classification_report", "N/A"))
    print(f"ROC-AUC: {report.get('roc_auc', 'N/A')}")

    # ── Step 10: Save Artifacts ────────────────────────────────────────────
    if not args.dry_run:
        logger.info("Saving model to: %s", classifier_path)
        save_model(model, classifier_path)

        logger.info("Saving vectorizer to: %s", vectorizer_path)
        save_vectorizer(vectorizer, vectorizer_path)

        logger.info("Training complete. Artifacts saved.")
    else:
        logger.info("Dry run mode: artifacts NOT saved.")

    logger.info("=" * 60)
    logger.info("Training pipeline finished successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
