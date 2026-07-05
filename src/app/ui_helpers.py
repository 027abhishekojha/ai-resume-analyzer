"""
src.app.ui_helpers
==================
Reusable Streamlit UI component helper functions for the AI Resume Analyzer.

Separating UI components from the main app logic keeps main.py clean
and makes individual components independently testable.
"""

import logging
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.inference.analyzer import AnalysisResult

logger = logging.getLogger(__name__)

# Score → color mapping for visual feedback
SCORE_COLOR_MAP: dict[str, str] = {
    "Excellent": "#2ECC71",
    "Good": "#F39C12",
    "Fair": "#E67E22",
    "Poor": "#E74C3C",
}

SCORE_EMOJI_MAP: dict[str, str] = {
    "Excellent": "🌟",
    "Good": "✅",
    "Fair": "⚠️",
    "Poor": "❌",
}


def render_header() -> None:
    """Render the application header with title and description."""
    st.markdown(
        """
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>🎯 AI Resume Analyzer</h1>
            <p style='font-size: 1.1rem; color: #9E9E9E; margin-top: 8px;'>
                ATS-powered resume screening using NLP & Machine Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: dict) -> None:
    """Render the application sidebar with settings and info.

    Args:
        config: Application configuration dictionary.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        # Display configuration info
        threshold = config.get("similarity", {}).get("thresholds", {}).get("good", 0.65)
        st.metric("Score Threshold (Good)", f"{threshold * 100:.0f}%")

        algorithm = config.get("model", {}).get("algorithm", "logistic_regression")
        st.metric("Classifier", algorithm.replace("_", " ").title())

        st.markdown("---")
        st.markdown("### 📖 How It Works")
        st.markdown(
            """
            1. **Upload** your PDF resume
            2. **Paste** the job description
            3. **Click Analyze** to run the pipeline
            4. **Review** your match score, keyword gaps,
               and suitability prediction
            """
        )

        st.markdown("---")
        st.markdown("### 🔗 Resources")
        st.markdown(
            "[📂 GitHub Repository](https://github.com/027abhishekojha/ai-resume-analyzer)\n\n"
            "[📄 Documentation](https://github.com/027abhishekojha/ai-resume-analyzer/blob/main/docs/Architecture.md)\n\n"
            "[🐛 Report an Issue](https://github.com/027abhishekojha/ai-resume-analyzer/issues)"
        )

        st.markdown("---")
        st.caption("AI Resume Analyzer v0.1.0 | MIT License")


def render_score_card(result: "AnalysisResult") -> None:
    """Render the summary score card with key metrics.

    Args:
        result: AnalysisResult from ResumeAnalyzer.analyze().
    """
    color = SCORE_COLOR_MAP.get(result.similarity_label, "#9E9E9E")
    emoji = SCORE_EMOJI_MAP.get(result.similarity_label, "❓")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🔢 Match Score",
            value=f"{result.similarity_score * 100:.1f}%",
            delta=None,
        )
    with col2:
        st.metric(
            label="🏷️ Match Level",
            value=f"{emoji} {result.similarity_label}",
        )
    with col3:
        prediction_label = (
            result.prediction.label if result.prediction else "N/A"
        )
        prediction_emoji = "✅" if prediction_label == "Suitable" else "❌"
        st.metric(
            label="🤖 Prediction",
            value=f"{prediction_emoji} {prediction_label}",
        )
    with col4:
        confidence = (
            f"{result.prediction.confidence:.1f}%"
            if result.prediction
            else "N/A"
        )
        st.metric(label="📊 Confidence", value=confidence)


def render_keyword_section(result: "AnalysisResult") -> None:
    """Render the keyword gap and overlap analysis sections.

    Args:
        result: AnalysisResult from ResumeAnalyzer.analyze().
    """
    from src.evaluation.visualizer import plot_keyword_bar  # noqa: PLC0415

    gap_col, overlap_col = st.columns(2)

    with gap_col:
        st.markdown("### ❌ Missing Keywords")
        st.caption("These keywords appear in the JD but are absent from your resume.")
        if result.keyword_gaps:
            gap_fig = plot_keyword_bar(
                result.keyword_gaps[:15], title="", color="#E53935"
            )
            st.plotly_chart(gap_fig, use_container_width=True)
        else:
            st.success("🎉 No major keyword gaps found!")

    with overlap_col:
        st.markdown("### ✅ Matching Keywords")
        st.caption("These keywords are present in both your resume and the JD.")
        if result.keyword_overlaps:
            overlap_fig = plot_keyword_bar(
                result.keyword_overlaps[:15], title="", color="#2ECC71"
            )
            st.plotly_chart(overlap_fig, use_container_width=True)
        else:
            st.warning("⚠️ No strong keyword overlaps detected.")

    # Keyword chips as tags
    if result.keyword_gaps:
        st.markdown("#### 💡 Add These Keywords to Your Resume:")
        chips_html = " ".join(
            f'<span style="background:#E53935;color:white;padding:4px 10px;'
            f'border-radius:20px;margin:3px;font-size:0.85rem;">{kw}</span>'
            for kw in result.keyword_gaps[:20]
        )
        st.markdown(chips_html, unsafe_allow_html=True)


def render_footer() -> None:
    """Render the application footer."""
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#616161;font-size:0.85rem;'>"
        "Built with ❤️ using Python, Streamlit, and scikit-learn | "
        "<a href='https://github.com/yourusername/ai-resume-analyzer' "
        "style='color:#1E88E5;'>GitHub</a>"
        "</p>",
        unsafe_allow_html=True,
    )


# TODO: Add dark/light theme toggle
# TODO: Add multi-page navigation (st.navigation) for separate History and Settings pages
