"""
src.inference.analyzer
=======================
High-level orchestrator that ties together the full resume analysis pipeline:
    1. PDF text extraction
    2. Text cleaning
    3. Tokenization
    4. TF-IDF vectorization
    5. Cosine similarity scoring
    6. ML suitability classification
    7. Keyword gap analysis

This is the primary entry point for the Streamlit app and API layer.

Usage:
    from src.inference.analyzer import ResumeAnalyzer, AnalysisResult

    analyzer = ResumeAnalyzer(config)
    result = analyzer.analyze(resume_bytes, job_description_text)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.feature_engineering.keyword_extractor import find_keyword_gaps, get_keyword_overlap
from src.feature_engineering.tfidf_vectorizer import load_vectorizer, transform_documents
from src.inference.classifier import PredictionResult, load_model, predict_suitability
from src.inference.similarity_scorer import compute_cosine_similarity, score_to_label
from src.preprocessing.pdf_extractor import extract_text_from_pdf
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.tokenizer import tokenize, tokens_to_string

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Complete analysis result returned by ResumeAnalyzer.

    Attributes:
        similarity_score: Cosine similarity score in [0.0, 1.0].
        similarity_label: Human-readable score band label.
        prediction: Suitability prediction result from the classifier.
        keyword_gaps: Keywords in JD missing from the resume.
        keyword_overlaps: Keywords present in both resume and JD.
        resume_text_raw: Raw extracted text from the resume.
        resume_text_clean: Cleaned and preprocessed resume text.
        error: Error message if analysis failed, else None.
    """

    similarity_score: float = 0.0
    similarity_label: str = "Poor"
    prediction: Optional[PredictionResult] = None
    keyword_gaps: list[str] = field(default_factory=list)
    keyword_overlaps: list[str] = field(default_factory=list)
    resume_text_raw: str = ""
    resume_text_clean: str = ""
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Return True if the analysis completed without errors."""
        return self.error is None


class ResumeAnalyzer:
    """Orchestrates the end-to-end resume analysis pipeline.

    This class is designed to be instantiated once (e.g., at Streamlit
    app startup) and reused across multiple analysis requests. The
    vectorizer and model are loaded once and cached in memory.

    Args:
        vectorizer_path: Path to the saved TF-IDF vectorizer (.joblib).
        model_path: Path to the saved classifier (.joblib).
        config: Optional configuration dictionary (from load_config).
    """

    def __init__(
        self,
        vectorizer_path: str = "models/vectorizers/tfidf_vectorizer.joblib",
        model_path: str = "models/trained/classifier.joblib",
        config: Optional[dict] = None,
    ) -> None:
        self.config = config or {}
        self._vectorizer = None
        self._model = None
        self._vectorizer_path = vectorizer_path
        self._model_path = model_path
        logger.info("ResumeAnalyzer initialized.")

    def _load_artifacts(self) -> None:
        """Lazily load the vectorizer and model on first use."""
        if self._vectorizer is None:
            logger.info("Loading vectorizer...")
            self._vectorizer = load_vectorizer(self._vectorizer_path)
        if self._model is None:
            logger.info("Loading classifier model...")
            self._model = load_model(self._model_path)

    def analyze(
        self,
        resume_source,  # str path, bytes, or BytesIO
        job_description: str,
        top_keywords: int = 20,
        classification_threshold: float = 0.60,
    ) -> AnalysisResult:
        """Run the full analysis pipeline on a resume and job description.

        Args:
            resume_source: PDF resume as a file path (str), bytes, or BytesIO.
            job_description: Raw job description text (pasted by user).
            top_keywords: Number of top keywords for gap analysis.
            classification_threshold: Probability threshold for classifier.

        Returns:
            AnalysisResult dataclass containing all analysis outputs.
        """
        result = AnalysisResult()

        try:
            # ── Step 1: Extract resume text ──────────────────────────────
            logger.info("Step 1/6: Extracting text from PDF...")
            result.resume_text_raw = extract_text_from_pdf(resume_source)
            if not result.resume_text_raw.strip():
                result.error = (
                    "Could not extract text from the uploaded PDF. "
                    "Please ensure the PDF is not scanned/image-based."
                )
                return result

            # ── Step 2: Clean text ───────────────────────────────────────
            logger.info("Step 2/6: Cleaning and normalizing text...")
            resume_clean = clean_text(result.resume_text_raw)
            jd_clean = clean_text(job_description)
            result.resume_text_clean = resume_clean

            # ── Step 3: Tokenize ─────────────────────────────────────────
            logger.info("Step 3/6: Tokenizing text...")
            resume_processed = tokens_to_string(tokenize(resume_clean))
            jd_processed = tokens_to_string(tokenize(jd_clean))

            # ── Step 4: Load ML artifacts ─────────────────────────────────
            logger.info("Step 4/6: Loading vectorizer and model...")
            self._load_artifacts()

            # ── Step 5: Compute similarity ────────────────────────────────
            logger.info("Step 5/6: Computing cosine similarity...")
            vectors = self._vectorizer.transform(  # type: ignore[union-attr]
                [resume_processed, jd_processed]
            ).toarray()
            result.similarity_score = compute_cosine_similarity(
                vectors[0], vectors[1]
            )
            result.similarity_label = score_to_label(result.similarity_score)

            # ── Step 6: Classify and analyze keywords ─────────────────────
            logger.info("Step 6/6: Running classifier and keyword analysis...")

            # Classifier: use concatenated text as input feature
            combined_vec = np.mean(vectors, axis=0, keepdims=True)
            result.prediction = predict_suitability(
                self._model,  # type: ignore[arg-type]
                combined_vec,
                threshold=classification_threshold,
            )

            # Keyword analysis
            result.keyword_gaps = find_keyword_gaps(
                resume_processed,
                jd_processed,
                self._vectorizer,  # type: ignore[arg-type]
                top_n=top_keywords,
            )
            result.keyword_overlaps = get_keyword_overlap(
                resume_processed,
                jd_processed,
                self._vectorizer,  # type: ignore[arg-type]
                top_n=top_keywords,
            )

            logger.info(
                "Analysis complete. Score=%.4f (%s), Prediction=%s",
                result.similarity_score,
                result.similarity_label,
                result.prediction.label if result.prediction else "N/A",
            )

        except FileNotFoundError as exc:
            logger.error("Model artifact not found: %s", exc)
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during analysis: %s", exc)
            result.error = f"An unexpected error occurred: {exc}"

        return result


# TODO: Add async support for handling concurrent requests (FastAPI + asyncio)
# TODO: Add caching layer (LRU cache or Redis) to avoid re-processing identical inputs
