"""
src.utils.bootstrap
====================
Auto-bootstrap utility for Streamlit Cloud and first-run environments.

When the app starts on a fresh deployment (e.g., Streamlit Community Cloud),
trained model artifacts won't exist. This module detects that state and runs
the full training pipeline automatically using the synthetic dataset.

Called once at Streamlit app startup via st.cache_resource.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default artifact paths (must match configs/config.yaml)
DEFAULT_CLASSIFIER_PATH = "models/trained/classifier.joblib"
DEFAULT_VECTORIZER_PATH = "models/vectorizers/tfidf_vectorizer.joblib"
DEFAULT_DATASET_PATH = "datasets/processed/processed_resumes.csv"


def models_exist(
    classifier_path: str = DEFAULT_CLASSIFIER_PATH,
    vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
) -> bool:
    """Check whether both trained model artifacts are present on disk.

    Args:
        classifier_path: Path to the classifier .joblib file.
        vectorizer_path: Path to the vectorizer .joblib file.

    Returns:
        True if both files exist, False otherwise.
    """
    return Path(classifier_path).exists() and Path(vectorizer_path).exists()


def bootstrap_models(
    classifier_path: str = DEFAULT_CLASSIFIER_PATH,
    vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    n_samples: int = 300,
) -> bool:
    """Generate synthetic data and train the ML pipeline from scratch.

    This is called automatically on Streamlit Cloud (or any fresh environment)
    when model artifacts are not found. The full pipeline runs in-process:
        1. Generate synthetic resume-JD dataset
        2. Preprocess all documents (clean + tokenize)
        3. Fit TF-IDF vectorizer on training corpus
        4. Train Logistic Regression classifier
        5. Save both artifacts to disk

    Args:
        classifier_path: Destination path for the trained classifier.
        vectorizer_path: Destination path for the fitted vectorizer.
        dataset_path: Destination path for the generated dataset CSV.
        n_samples: Number of synthetic training samples to generate.

    Returns:
        True if bootstrap succeeded, False on any error.
    """
    logger.info(
        "Model artifacts not found — bootstrapping training pipeline "
        "with %d synthetic samples...",
        n_samples,
    )

    try:
        # ── Step 1: Generate synthetic dataset ─────────────────────────
        from scripts.download_dataset import generate_synthetic_dataset  # noqa: PLC0415
        from src.utils.file_helpers import ensure_directory_exists  # noqa: PLC0415

        ensure_directory_exists(str(Path(dataset_path).parent))
        df = generate_synthetic_dataset(n_samples=n_samples, random_seed=42)
        df.to_csv(dataset_path, index=False)
        logger.info("Synthetic dataset saved: %d samples → %s", len(df), dataset_path)

        # ── Step 2: Load and split dataset ──────────────────────────────
        from src.training.data_loader import load_training_data, split_dataset  # noqa: PLC0415

        df = load_training_data(dataset_path)
        X_train_raw, X_test_raw, y_train, y_test = split_dataset(df, test_size=0.20)

        # ── Step 3: Preprocess text ─────────────────────────────────────
        from src.preprocessing.text_cleaner import clean_text  # noqa: PLC0415
        from src.preprocessing.tokenizer import tokenize, tokens_to_string  # noqa: PLC0415

        logger.info("Preprocessing %d training documents...", len(X_train_raw))
        X_train_text = [
            tokens_to_string(tokenize(clean_text(t))) for t in X_train_raw
        ]
        X_test_text = [
            tokens_to_string(tokenize(clean_text(t))) for t in X_test_raw
        ]

        # ── Step 4: Fit TF-IDF vectorizer ───────────────────────────────
        from src.feature_engineering.tfidf_vectorizer import (  # noqa: PLC0415
            fit_vectorizer,
            save_vectorizer,
            transform_documents,
        )

        vectorizer = fit_vectorizer(
            corpus=X_train_text,
            max_features=2000,  # Smaller vocab for fast bootstrap
            ngram_range=(1, 2),
            min_df=1,           # min_df=1 so all synthetic terms are included
            max_df=0.95,
            sublinear_tf=True,
        )

        X_train = transform_documents(X_train_text, vectorizer)

        # ── Step 5: Train classifier ────────────────────────────────────
        from src.training.trainer import train_classifier  # noqa: PLC0415

        model = train_classifier(X_train, y_train, algorithm="logistic_regression")

        # ── Step 6: Save artifacts ──────────────────────────────────────
        from src.inference.classifier import save_model  # noqa: PLC0415

        ensure_directory_exists(str(Path(classifier_path).parent))
        ensure_directory_exists(str(Path(vectorizer_path).parent))

        save_model(model, classifier_path)
        save_vectorizer(vectorizer, vectorizer_path)

        logger.info(
            "Bootstrap complete. Classifier → %s | Vectorizer → %s",
            classifier_path,
            vectorizer_path,
        )
        return True

    except Exception as exc:  # noqa: BLE001
        logger.exception("Bootstrap failed: %s", exc)
        return False


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present.

    Safe to call multiple times — NLTK skips downloads if data exists.
    """
    try:
        import nltk  # type: ignore[import]

        packages = ["punkt", "punkt_tab", "stopwords", "wordnet"]
        for pkg in packages:
            nltk.download(pkg, quiet=True)
        logger.debug("NLTK data verified.")
    except ImportError:
        logger.error("NLTK not installed. Run: pip install nltk")
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLTK data download warning: %s", exc)
