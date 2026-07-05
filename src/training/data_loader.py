"""
src.training.data_loader
=========================
Loads and validates training datasets for the ML classifier.

Expected dataset schema (CSV):
    - resume_text: str — Pre-processed resume text
    - jd_text: str     — Pre-processed job description text
    - label: int       — 1 = Suitable, 0 = Not Suitable

Usage:
    from src.training.data_loader import load_training_data, split_dataset
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore[import]

logger = logging.getLogger(__name__)

# Required columns in the training dataset
REQUIRED_COLUMNS: list[str] = ["resume_text", "jd_text", "label"]


def load_training_data(dataset_path: str) -> pd.DataFrame:
    """Load and validate a labeled training dataset from a CSV file.

    Args:
        dataset_path: Path to the CSV training dataset.

    Returns:
        A validated pandas DataFrame with the required columns.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is missing required columns or is empty.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}. "
            "Run 'python scripts/download_dataset.py' to fetch sample data."
        )

    df = pd.read_csv(path)

    # Validate columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    # Validate data
    if df.empty:
        raise ValueError("Training dataset is empty.")

    original_len = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    dropped = original_len - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with missing values.", dropped)

    # Validate labels
    valid_labels = {0, 1}
    invalid_labels = set(df["label"].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(
            f"Invalid label values found: {invalid_labels}. "
            "Labels must be 0 (Not Suitable) or 1 (Suitable)."
        )

    class_dist = df["label"].value_counts().to_dict()
    logger.info(
        "Dataset loaded: %d samples | Class distribution: %s",
        len(df),
        class_dist,
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    text_column: str = "resume_text",
    label_column: str = "label",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Split a dataset into training and test sets.

    Args:
        df: Validated training DataFrame.
        text_column: Column name containing the feature text.
        label_column: Column name containing the binary labels.
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df[text_column].tolist()
    y = df[label_column].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Dataset split: %d train / %d test (test_size=%.2f).",
        len(X_train),
        len(X_test),
        test_size,
    )
    return X_train, X_test, y_train, y_test


# TODO: Add data augmentation utilities (synonym replacement, back-translation)
# TODO: Add support for imbalanced datasets via SMOTE or class_weight
