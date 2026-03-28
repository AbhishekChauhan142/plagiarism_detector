"""Visualization utilities for plagiarism detection research results."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend suitable for servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available – visualization functions will raise ImportError.")


def _require_matplotlib() -> None:
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    auc: float,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Plot a Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr: False positive rates (x-axis).
        tpr: True positive rates (y-axis).
        auc: Area under the ROC curve to display in the legend.
        save_path: Optional file path to save the figure (PNG/PDF/SVG).

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="lightgray", lw=1.5, linestyle="--", label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)

    return fig


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Plot a confusion matrix heatmap.

    Args:
        cm_dict: Dict with keys ``tp``, ``fp``, ``tn``, ``fn``.
        save_path: Optional file path to save the figure.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    _require_matplotlib()

    tp = cm_dict.get("tp", 0)
    fp = cm_dict.get("fp", 0)
    tn = cm_dict.get("tn", 0)
    fn = cm_dict.get("fn", 0)

    matrix = np.array([[tn, fp], [fn, tp]])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    totals = matrix.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = ["Negative (0)", "Positive (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = f"{100 * matrix[i, j] / max(totals, 1):.1f}%"
            ax.text(
                j,
                i,
                f"{labels[i, j]}\n{matrix[i, j]}\n({pct})",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_algorithm_comparison(
    comparison_results: Dict[str, Dict[str, float]],
    metric: str = "f1",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Bar chart comparing algorithms by a chosen metric.

    Args:
        comparison_results: Output of
            :meth:`~src.research_evaluator.ResearchEvaluator.compare_algorithms`,
            mapping algorithm name → metrics dict.
        metric: Metric key to visualise (e.g. ``"f1"``, ``"accuracy"``).
        save_path: Optional file path to save the figure.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    _require_matplotlib()

    algo_names = list(comparison_results.keys())
    values = [comparison_results[name].get(metric, 0.0) or 0.0 for name in algo_names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(algo_names)))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(max(6, len(algo_names) * 1.5), 5))
    bars = ax.bar(algo_names, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Algorithm Comparison – {metric.upper()}", fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Algorithm comparison chart saved to %s", save_path)

    return fig


def plot_similarity_distribution(
    similarities: List[float],
    labels: List[int],
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Histogram of similarity scores split by class label.

    Args:
        similarities: Continuous similarity scores per document pair.
        labels: Binary labels (0 = clean, 1 = plagiarised) aligned with
            *similarities*.
        save_path: Optional file path to save the figure.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    _require_matplotlib()

    sims = np.array(similarities)
    lbls = np.array(labels)

    plagiarised_sims = sims[lbls == 1]
    clean_sims = sims[lbls == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 26)

    if len(clean_sims) > 0:
        ax.hist(
            clean_sims,
            bins=bins,
            alpha=0.6,
            color="steelblue",
            label=f"Clean (n={len(clean_sims)})",
            edgecolor="white",
        )
    if len(plagiarised_sims) > 0:
        ax.hist(
            plagiarised_sims,
            bins=bins,
            alpha=0.6,
            color="tomato",
            label=f"Plagiarised (n={len(plagiarised_sims)})",
            edgecolor="white",
        )

    ax.set_xlabel("Similarity Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Similarity Score Distribution by Class", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Similarity distribution plot saved to %s", save_path)

    return fig
