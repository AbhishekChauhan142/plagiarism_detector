"""Research evaluation module for plagiarism detection benchmarking."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available – some metrics will be unavailable.")

try:
    from scipy import stats as scipy_stats  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not available – statistical significance tests will be unavailable.")


class ResearchEvaluator:
    """Evaluation utilities for plagiarism detection research.

    Provides standard classification metrics, k-fold cross-validation,
    multi-algorithm comparison, confusion matrix helpers, statistical
    significance testing, and pretty-printed results tables.
    """

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: List[int],
        ground_truth: List[int],
        scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Compute classification metrics for a single algorithm run.

        Args:
            predictions: Binary predicted labels (0 or 1).
            ground_truth: Binary ground-truth labels (0 or 1).
            scores: Optional continuous similarity scores used to compute
                ROC-AUC.

        Returns:
            Dict with precision, recall, f1, accuracy, and optionally roc_auc.

        Raises:
            ImportError: If scikit-learn is not installed.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for evaluate().")

        preds = list(predictions)
        truth = list(ground_truth)

        results: Dict[str, Any] = {
            "precision": float(precision_score(truth, preds, zero_division=0)),
            "recall": float(recall_score(truth, preds, zero_division=0)),
            "f1": float(f1_score(truth, preds, zero_division=0)),
            "accuracy": float(accuracy_score(truth, preds)),
        }

        if scores is not None:
            try:
                results["roc_auc"] = float(roc_auc_score(truth, scores))
            except ValueError as exc:
                logger.warning("Could not compute ROC-AUC: %s", exc)
                results["roc_auc"] = None

        return results

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        algorithm: Any,
        pairs: List[Tuple[str, str]],
        labels: List[int],
        k: int = 5,
    ) -> Dict[str, Any]:
        """Stratified k-fold cross-validation of a detection algorithm.

        The algorithm must expose a ``detect(text1, text2)`` method that
        returns a dict with at least an ``"is_plagiarized"`` key.

        Args:
            algorithm: Detection algorithm instance.
            pairs: List of ``(text1, text2)`` tuples.
            labels: Binary ground-truth labels aligned with *pairs*.
            k: Number of folds.

        Returns:
            Dict with mean and std of precision, recall, f1, and accuracy
            across folds.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for cross_validate().")

        pairs_arr = list(pairs)
        labels_arr = list(labels)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_metrics: Dict[str, List[float]] = {
            "precision": [],
            "recall": [],
            "f1": [],
            "accuracy": [],
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(pairs_arr, labels_arr), start=1):
            test_pairs = [pairs_arr[i] for i in test_idx]
            test_labels = [labels_arr[i] for i in test_idx]

            preds = []
            for text1, text2 in test_pairs:
                try:
                    result = algorithm.detect(text1, text2)
                    preds.append(int(result.get("is_plagiarized", False)))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Algorithm error in fold %d: %s", fold, exc)
                    preds.append(0)

            metrics = self.evaluate(preds, test_labels)
            for key in fold_metrics:
                fold_metrics[key].append(metrics[key])

        return {
            f"{key}_mean": float(np.mean(vals))
            for key, vals in fold_metrics.items()
        } | {
            f"{key}_std": float(np.std(vals))
            for key, vals in fold_metrics.items()
        }

    # ------------------------------------------------------------------
    # Algorithm comparison
    # ------------------------------------------------------------------

    def compare_algorithms(
        self,
        algorithms: List[Any],
        pairs: List[Tuple[str, str]],
        labels: List[int],
    ) -> Dict[str, Dict[str, Any]]:
        """Run each algorithm on the same corpus and compare metrics.

        Args:
            algorithms: List of algorithm instances with a ``detect()``
                method and a ``name`` attribute.
            pairs: List of ``(text1, text2)`` tuples.
            labels: Binary ground-truth labels aligned with *pairs*.

        Returns:
            Dict mapping algorithm name → metrics dict.
        """
        comparison: Dict[str, Dict[str, Any]] = {}
        for algo in algorithms:
            algo_name = getattr(algo, "name", repr(algo))
            logger.info("Evaluating algorithm: %s", algo_name)
            preds: List[int] = []
            scores: List[float] = []
            for text1, text2 in pairs:
                try:
                    result = algo.detect(text1, text2)
                    preds.append(int(result.get("is_plagiarized", False)))
                    scores.append(float(result.get("similarity", 0.0)))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Algorithm %s failed: %s", algo_name, exc)
                    preds.append(0)
                    scores.append(0.0)
            comparison[algo_name] = self.evaluate(preds, labels, scores=scores)
        return comparison

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def generate_confusion_matrix(
        self, predictions: List[int], ground_truth: List[int]
    ) -> Dict[str, int]:
        """Compute confusion matrix components.

        Args:
            predictions: Binary predicted labels.
            ground_truth: Binary ground-truth labels.

        Returns:
            Dict with tp, fp, tn, fn.
        """
        tp = fp = tn = fn = 0
        for pred, truth in zip(predictions, ground_truth):
            if pred == 1 and truth == 1:
                tp += 1
            elif pred == 1 and truth == 0:
                fp += 1
            elif pred == 0 and truth == 0:
                tn += 1
            else:
                fn += 1
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    # ------------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------------

    def statistical_significance_test(
        self,
        results1: List[float],
        results2: List[float],
    ) -> Dict[str, Any]:
        """Test whether two sets of scores differ significantly.

        Uses a Wilcoxon signed-rank test when scipy is available, otherwise
        falls back to a simple mean comparison.

        Args:
            results1: Metric values from algorithm 1 (one per sample or fold).
            results2: Metric values from algorithm 2 (one per sample or fold).

        Returns:
            Dict with test_name, statistic, p_value, is_significant.
        """
        if not _SCIPY_AVAILABLE:
            logger.warning("scipy not available; using approximate t-test fallback.")
            mean1 = float(np.mean(results1))
            mean2 = float(np.mean(results2))
            return {
                "test_name": "mean_comparison_fallback",
                "statistic": abs(mean1 - mean2),
                "p_value": None,
                "is_significant": None,
                "note": "Install scipy for a proper significance test.",
            }

        try:
            stat, p_value = scipy_stats.wilcoxon(results1, results2)
            return {
                "test_name": "wilcoxon_signed_rank",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_significant": bool(p_value < 0.05),
            }
        except ValueError as exc:
            logger.warning("Wilcoxon test failed (%s); trying paired t-test.", exc)
            try:
                stat, p_value = scipy_stats.ttest_rel(results1, results2)
                return {
                    "test_name": "paired_t_test",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_significant": bool(p_value < 0.05),
                }
            except Exception as inner_exc:  # noqa: BLE001
                logger.error("Statistical test failed: %s", inner_exc)
                return {
                    "test_name": "failed",
                    "statistic": None,
                    "p_value": None,
                    "is_significant": None,
                    "error": str(inner_exc),
                }

    # ------------------------------------------------------------------
    # Results formatting
    # ------------------------------------------------------------------

    def format_results_table(self, comparison_results: Dict[str, Dict[str, Any]]) -> str:
        """Format algorithm comparison results as a plain-text table.

        Args:
            comparison_results: Output of :meth:`compare_algorithms`.

        Returns:
            Formatted string table ready for printing.
        """
        if not comparison_results:
            return "No results to display."

        # Determine which metrics are present across all algorithms
        all_metrics = sorted(
            {metric for metrics in comparison_results.values() for metric in metrics}
        )
        # Put the most informative ones first
        preferred_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        ordered_metrics = [m for m in preferred_order if m in all_metrics] + [
            m for m in all_metrics if m not in preferred_order
        ]

        col_width = 12
        algo_col_width = max(len(name) for name in comparison_results) + 2
        algo_col_width = max(algo_col_width, 12)

        header_parts = [f"{'Algorithm':<{algo_col_width}}"] + [
            f"{m.upper():>{col_width}}" for m in ordered_metrics
        ]
        header = "  ".join(header_parts)
        separator = "-" * len(header)

        lines = [separator, header, separator]
        for algo_name, metrics in comparison_results.items():
            row_parts = [f"{algo_name:<{algo_col_width}}"]
            for metric in ordered_metrics:
                value = metrics.get(metric)
                if value is None:
                    row_parts.append(f"{'N/A':>{col_width}}")
                else:
                    row_parts.append(f"{value:>{col_width}.4f}")
            lines.append("  ".join(row_parts))
        lines.append(separator)
        return "\n".join(lines)
