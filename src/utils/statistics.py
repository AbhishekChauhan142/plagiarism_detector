"""Statistical utility functions for plagiarism detection research."""

import logging
import math
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats  # type: ignore

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not available – some statistical functions will be limited.")


def compute_summary_statistics(values: List[float]) -> dict:
    """Compute descriptive statistics for a list of numeric values.

    Args:
        values: Non-empty list of floats.

    Returns:
        Dict with mean, std, min, max, median, q25, q75.
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def bootstrap_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """Estimate a bootstrap confidence interval for the mean.

    Args:
        values: Sample values.
        confidence: Desired confidence level (e.g. 0.95 for 95 %).
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Tuple (lower_bound, upper_bound).
    """
    if not values:
        return (0.0, 0.0)

    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boot_means = np.array(
        [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses the pooled standard deviation.  Returns 0.0 if either group is
    empty or the pooled std is zero.

    Args:
        group1: Values for the first group.
        group2: Values for the second group.

    Returns:
        Cohen's d (float).  Positive values indicate group1 > group2.
    """
    if not group1 or not group2:
        return 0.0

    arr1 = np.array(group1, dtype=float)
    arr2 = np.array(group2, dtype=float)

    n1, n2 = len(arr1), len(arr2)
    var1 = np.var(arr1, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(arr2, ddof=1) if n2 > 1 else 0.0

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled_std == 0:
        return 0.0

    return float((np.mean(arr1) - np.mean(arr2)) / pooled_std)


def wilcoxon_test(values1: List[float], values2: List[float]) -> dict:
    """Perform a Wilcoxon signed-rank test on two paired samples.

    Falls back to a paired t-test when scipy is unavailable.

    Args:
        values1: Metric values from the first condition.
        values2: Metric values from the second condition.

    Returns:
        Dict with statistic, p_value, test_name, and is_significant.
    """
    if not _SCIPY_AVAILABLE:
        logger.warning("scipy not available; Wilcoxon test cannot be performed.")
        return {
            "test_name": "unavailable",
            "statistic": None,
            "p_value": None,
            "is_significant": None,
            "note": "Install scipy>=1.7.0 to enable this test.",
        }

    try:
        stat, p_value = scipy_stats.wilcoxon(values1, values2)
        return {
            "test_name": "wilcoxon_signed_rank",
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
        }
    except ValueError as exc:
        logger.warning("Wilcoxon test failed: %s – falling back to paired t-test.", exc)
        try:
            stat, p_value = scipy_stats.ttest_rel(values1, values2)
            return {
                "test_name": "paired_t_test_fallback",
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
