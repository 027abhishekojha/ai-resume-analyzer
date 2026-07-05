"""
src.evaluation.visualizer
==========================
Generates evaluation charts and visualizations for model assessment.

Uses Matplotlib for static plots and Plotly for interactive charts.
Outputs can be saved to files or rendered inline in Streamlit.

Usage:
    from src.evaluation.visualizer import plot_confusion_matrix, plot_roc_curve
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str] = ("Not Suitable", "Suitable"),
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Plot a confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix as a 2D list (from sklearn.metrics.confusion_matrix).
        class_names: Labels for the matrix axes.
        title: Plot title.
        save_path: If provided, save the figure to this path.

    Returns:
        A matplotlib Figure object.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate cells with counts
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j, i, format(cm_array[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_array[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to: %s", save_path)

    return fig


def plot_similarity_gauge(
    score: float,
    title: str = "Resume-JD Match Score",
) -> "go.Figure":
    """Create an interactive gauge chart for the similarity score using Plotly.

    Args:
        score: Cosine similarity score in [0.0, 1.0].
        title: Chart title.

    Returns:
        A Plotly Figure object (can be rendered with st.plotly_chart).
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import]
    except ImportError:
        logger.error("Plotly is not installed. Run: pip install plotly")
        raise

    pct = score * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            title={"text": title, "font": {"size": 20}},
            delta={"reference": 65, "increasing": {"color": "#2ECC71"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1E88E5"},
                "steps": [
                    {"range": [0, 50], "color": "#FFCDD2"},
                    {"range": [50, 65], "color": "#FFF9C4"},
                    {"range": [65, 80], "color": "#C8E6C9"},
                    {"range": [80, 100], "color": "#A5D6A7"},
                ],
                "threshold": {
                    "line": {"color": "#E53935", "width": 4},
                    "thickness": 0.75,
                    "value": 65,
                },
            },
            number={"suffix": "%", "font": {"size": 28}},
        )
    )
    fig.update_layout(height=300, margin={"t": 80, "b": 20, "l": 20, "r": 20})
    return fig


def plot_keyword_bar(
    keywords: list[str],
    title: str = "Missing Keywords",
    color: str = "#E53935",
) -> "go.Figure":
    """Create a horizontal bar chart for keyword gap analysis using Plotly.

    Args:
        keywords: List of keyword strings to display.
        title: Chart title.
        color: Bar color (hex string).

    Returns:
        A Plotly Figure object.
    """
    try:
        import plotly.graph_objects as go  # type: ignore[import]
    except ImportError:
        logger.error("Plotly is not installed.")
        raise

    if not keywords:
        keywords = ["No gaps identified!"]

    fig = go.Figure(
        go.Bar(
            x=[1] * len(keywords),
            y=keywords,
            orientation="h",
            marker_color=color,
            text=keywords,
            textposition="inside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(200, 30 * len(keywords)),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


# TODO: Add ROC curve plot with AUC annotation
# TODO: Add SHAP summary plot for model explainability
