"""
scripts/download_dataset.py
============================
Script to download or generate a sample training dataset for the
AI Resume Analyzer ML classifier.

In production, this script would:
    - Download a labeled resume dataset from a public repository
    - Validate downloaded files (checksums, schema)
    - Store raw data in datasets/raw/
    - Generate a synthetic sample dataset for development/testing

Currently this script generates a **synthetic** placeholder dataset
to allow the training pipeline to run without external data.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output datasets/processed/ --samples 200
"""

import argparse
import logging
import random
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.utils.file_helpers import ensure_directory_exists
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── Synthetic Data Templates ────────────────────────────────────────────────────

RESUME_TEMPLATES: list[str] = [
    (
        "experienced software engineer python machine learning scikit-learn "
        "tensorflow deep learning data pipeline deployment aws docker kubernetes"
    ),
    (
        "data scientist pandas numpy statistics hypothesis testing regression "
        "classification clustering feature engineering cross validation model tuning"
    ),
    (
        "machine learning engineer nlp natural language processing bert transformers "
        "huggingface pytorch gpu training inference optimization production"
    ),
    (
        "backend developer java spring boot rest api microservices sql postgresql "
        "redis kafka docker ci cd jenkins agile scrum"
    ),
    (
        "junior developer html css javascript react nodejs basic python "
        "git github internship projects portfolio"
    ),
    (
        "data analyst sql tableau power bi excel reporting dashboards business "
        "intelligence stakeholder communication presentation statistics"
    ),
]

JD_TEMPLATES: list[str] = [
    (
        "looking for machine learning engineer python scikit-learn tensorflow "
        "deep learning nlp experience required deployment cloud aws docker"
    ),
    (
        "data scientist role requires pandas numpy statistics machine learning "
        "classification regression feature engineering model evaluation python"
    ),
    (
        "senior software engineer python backend api design scalable systems "
        "microservices docker kubernetes ci cd testing agile"
    ),
    (
        "nlp engineer huggingface transformers bert large language models "
        "pytorch fine-tuning inference optimization production deployment"
    ),
    (
        "junior data analyst sql excel tableau communication stakeholders "
        "reports dashboards basic statistics python preferred"
    ),
]


def generate_synthetic_dataset(
    n_samples: int = 200,
    positive_ratio: float = 0.50,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic labeled dataset for training and testing.

    Creates resume-JD pairs with binary suitability labels. This is a
    placeholder until a real labeled dataset is available.

    Args:
        n_samples: Total number of samples to generate.
        positive_ratio: Fraction of samples labeled as "Suitable" (1).
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: resume_text, jd_text, label.
    """
    random.seed(random_seed)
    records: list[dict] = []
    n_positive = int(n_samples * positive_ratio)

    for i in range(n_samples):
        # Positive samples: use aligned resume-JD templates
        if i < n_positive:
            jd_idx = random.randrange(len(JD_TEMPLATES))
            resume_idx = jd_idx  # aligned pair → high similarity → Suitable
            label = 1
        else:
            # Negative samples: use misaligned templates
            jd_idx = random.randrange(len(JD_TEMPLATES))
            resume_idx = (jd_idx + random.randint(2, len(RESUME_TEMPLATES) - 1)) % len(
                RESUME_TEMPLATES
            )
            label = 0

        # Add slight variation with random extra terms
        extra_terms = random.choice(["", "team player", "remote work", "startup experience"])
        resume = RESUME_TEMPLATES[resume_idx] + " " + extra_terms
        jd = JD_TEMPLATES[jd_idx] + " " + extra_terms

        records.append({
            "resume_text": resume.strip(),
            "jd_text": jd.strip(),
            "label": label,
        })

    # Shuffle
    random.shuffle(records)
    df = pd.DataFrame(records)

    logger.info(
        "Generated synthetic dataset: %d samples | Positive: %d | Negative: %d",
        len(df),
        df["label"].sum(),
        (df["label"] == 0).sum(),
    )
    return df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download or generate a training dataset for AI Resume Analyzer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/processed/",
        help="Output directory for the processed dataset CSV.",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="datasets/raw/",
        help="Output directory for raw downloaded files.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of synthetic samples to generate (default: 200).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["synthetic", "kaggle"],
        default="synthetic",
        help="Data source: 'synthetic' (default) or 'kaggle' (requires API key).",
    )
    return parser.parse_args()


def main() -> None:
    """Main dataset download/generation entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AI Resume Analyzer — Dataset Download Script")
    logger.info("=" * 60)

    ensure_directory_exists(args.output)
    ensure_directory_exists(args.raw_output)

    if args.source == "synthetic":
        logger.info("Generating synthetic dataset with %d samples...", args.samples)
        df = generate_synthetic_dataset(n_samples=args.samples)

        output_path = Path(args.output) / "processed_resumes.csv"
        df.to_csv(output_path, index=False)
        logger.info("Synthetic dataset saved to: %s", output_path)

    elif args.source == "kaggle":
        # TODO: Implement Kaggle dataset download via kaggle API
        # Suggested dataset: "kaggle datasets download -d gauravduttakiit/resume-dataset"
        logger.error(
            "Kaggle download not yet implemented. "
            "Use --source synthetic for now, or manually download:\n"
            "  kaggle datasets download -d gauravduttakiit/resume-dataset\n"
            "  Place the CSV in datasets/raw/ and update the dataset_path in config.yaml"
        )
        sys.exit(1)

    logger.info("Dataset download/generation complete.")
    logger.info("Next step: python scripts/train_model.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
