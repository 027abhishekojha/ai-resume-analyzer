"""
src.app.main
============
Streamlit application entry point for the AI Resume Analyzer.

Run with:
    streamlit run src/app/main.py

The app provides:
    - PDF resume upload
    - Job description text input
    - Analysis results dashboard:
        * Match score (gauge chart)
        * Suitability prediction
        * Keyword gap analysis
        * Keyword overlap display
"""

import logging
import sys
from pathlib import Path

# Add project root to path for clean imports when running via streamlit
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from src.app.ui_helpers import (
    render_footer,
    render_header,
    render_keyword_section,
    render_score_card,
    render_sidebar,
)
from src.evaluation.visualizer import plot_keyword_bar, plot_similarity_gauge
from src.inference.analyzer import ResumeAnalyzer
from src.utils.bootstrap import bootstrap_models, ensure_nltk_data, models_exist
from src.utils.config_loader import load_config
from src.utils.logger import setup_logging

# ── Logging Setup ──────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _startup_bootstrap() -> bool:
    """Run once at app startup: ensure NLTK data and model artifacts exist.

    Uses st.cache_resource so this executes only once per Streamlit server
    process — safe for both local dev and Streamlit Community Cloud.

    Returns:
        True if artifacts are ready (existing or freshly trained).
    """
    ensure_nltk_data()
    if not models_exist():
        logger.info("No model artifacts found — triggering auto-bootstrap.")
        return bootstrap_models()
    logger.info("Model artifacts found — skipping bootstrap.")
    return True

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/ai-resume-analyzer",
        "Report a bug": "https://github.com/yourusername/ai-resume-analyzer/issues",
        "About": "AI Resume Analyzer v0.1.0 — Built with ❤️ and Python",
    },
)


@st.cache_resource(show_spinner=False)
def load_analyzer(config: dict) -> ResumeAnalyzer:
    """Load and cache the ResumeAnalyzer (vectorizer + model) at startup.

    Cached by st.cache_resource so artifacts are loaded only once,
    even across multiple user sessions.

    Args:
        config: Application configuration dictionary.

    Returns:
        Initialized ResumeAnalyzer instance.
    """
    vectorizer_path = config.get("model", {}).get(
        "vectorizer_path", "models/vectorizers/tfidf_vectorizer.joblib"
    )
    model_path = config.get("model", {}).get(
        "classifier_path", "models/trained/classifier.joblib"
    )
    logger.info("Loading ResumeAnalyzer artifacts...")
    return ResumeAnalyzer(
        vectorizer_path=vectorizer_path,
        model_path=model_path,
        config=config,
    )


@st.cache_data(show_spinner=False)
def load_app_config() -> dict:
    """Load and cache application configuration.

    Returns:
        Configuration dictionary from configs/config.yaml.
    """
    try:
        return load_config("configs/config.yaml")
    except FileNotFoundError:
        logger.warning("Config file not found; using defaults.")
        return {}


def main() -> None:
    """Main Streamlit application entry point."""
    # ── Bootstrap: ensure NLTK data + model artifacts exist ───────────────
    # On Streamlit Community Cloud (or any fresh env), models are absent.
    # _startup_bootstrap() generates synthetic data and trains in-process.
    # st.cache_resource ensures this runs ONCE per server lifetime.
    bootstrap_ok = _startup_bootstrap()
    if not bootstrap_ok:
        st.error(
            "⚠️ **Startup Error**: Could not initialize model artifacts. "
            "Please check the application logs."
        )
        st.stop()

    # ── Load Configuration ─────────────────────────────────────────────────
    config = load_app_config()

    # ── Custom CSS Injection ───────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .stApp { background-color: #0E1117; }
        .metric-card {
            background: linear-gradient(135deg, #1E88E5, #7B1FA2);
            border-radius: 12px;
            padding: 20px;
            color: white;
            text-align: center;
        }
        .score-excellent { color: #2ECC71; font-weight: bold; }
        .score-good { color: #F39C12; font-weight: bold; }
        .score-fair { color: #E67E22; font-weight: bold; }
        .score-poor { color: #E74C3C; font-weight: bold; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ────────────────────────────────────────────────────────────
    render_sidebar(config)

    # ── Header ─────────────────────────────────────────────────────────────
    render_header()

    # ── Input Section ──────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📄 Upload Your Resume")
        uploaded_file = st.file_uploader(
            label="Upload PDF Resume",
            type=["pdf"],
            help="Upload your resume in PDF format (max 10MB).",
            label_visibility="collapsed",
        )
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.success(
                f"✅ **{uploaded_file.name}** uploaded "
                f"({file_size_mb:.2f} MB)"
            )

    with col2:
        st.markdown("### 💼 Paste Job Description")
        job_description = st.text_area(
            label="Job Description",
            placeholder=(
                "Paste the full job description here...\n\n"
                "Include responsibilities, requirements, and preferred skills."
            ),
            height=220,
            label_visibility="collapsed",
        )
        word_count = len(job_description.split()) if job_description else 0
        st.caption(f"📝 {word_count} words")

    # ── Analyze Button ─────────────────────────────────────────────────────
    st.markdown("---")
    analyze_col, _ = st.columns([1, 3])
    with analyze_col:
        analyze_btn = st.button(
            "🚀 Analyze Resume",
            type="primary",
            use_container_width=True,
            disabled=(not uploaded_file or not job_description.strip()),
        )

    # ── Analysis Pipeline ──────────────────────────────────────────────────
    if analyze_btn and uploaded_file and job_description.strip():
        with st.spinner("⚙️ Analyzing your resume against the job description..."):
            try:
                analyzer = load_analyzer(config)
                resume_bytes = uploaded_file.read()
                result = analyzer.analyze(
                    resume_source=resume_bytes,
                    job_description=job_description,
                )
            except FileNotFoundError:
                st.error(
                    "⚠️ **Model Not Found**: Please train the model first.\n\n"
                    "```bash\npython scripts/train_model.py\n```"
                )
                st.stop()
            except Exception as exc:
                st.error(f"❌ An unexpected error occurred: {exc}")
                logger.exception("Analysis failed: %s", exc)
                st.stop()

        if result.error:
            st.error(f"❌ {result.error}")
            st.stop()

        # ── Results Dashboard ──────────────────────────────────────────
        st.markdown("## 📊 Analysis Results")
        st.markdown("---")

        # Score Card
        render_score_card(result)

        # Gauge Chart
        st.markdown("### 🎯 Match Score")
        gauge_fig = plot_similarity_gauge(result.similarity_score)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Keyword Sections
        render_keyword_section(result)

        # Raw Text Expander (debug)
        with st.expander("📃 View Extracted Resume Text", expanded=False):
            st.text(result.resume_text_raw[:2000] + "..." if len(result.resume_text_raw) > 2000 else result.resume_text_raw)

    # ── Footer ─────────────────────────────────────────────────────────────
    render_footer()


if __name__ == "__main__":
    main()
