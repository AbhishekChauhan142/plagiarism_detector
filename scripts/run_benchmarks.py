"""Benchmark suite for comparing plagiarism detection algorithms on a corpus.

Usage::

    python scripts/run_benchmarks.py \\
        --corpus-dir data/pan_corpus/sample \\
        --output-dir data/results \\
        --algorithms tfidf sbert ngram embedding ensemble
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path when executed directly
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pan_dataset_manager import PANDatasetManager
from src.research_evaluator import ResearchEvaluator
from src.detection_algorithms import (
    TFIDFAlgorithm,
    NGramAlgorithm,
    EmbeddingAlgorithm,
)

logger = logging.getLogger(__name__)

_ALGORITHM_REGISTRY = {
    "tfidf": TFIDFAlgorithm,
    "ngram": NGramAlgorithm,
    "embedding": EmbeddingAlgorithm,
}

# Optional heavy dependencies
try:
    from src.detection_algorithms import SBERTAlgorithm, EnsembleAlgorithm

    _ALGORITHM_REGISTRY["sbert"] = SBERTAlgorithm
    _ALGORITHM_REGISTRY["ensemble"] = EnsembleAlgorithm
except ImportError:
    logger.warning("sentence-transformers not installed; sbert/ensemble algorithms unavailable.")


def _build_algorithms(names: Optional[List[str]]) -> List:
    """Instantiate algorithm objects from a list of names."""
    if names is None:
        names = list(_ALGORITHM_REGISTRY.keys())
    algos = []
    for name in names:
        cls = _ALGORITHM_REGISTRY.get(name.lower())
        if cls is None:
            logger.warning("Unknown algorithm '%s'; skipping.", name)
            continue
        algos.append(cls())
    return algos


def run_benchmark(
    corpus_dir: str,
    output_dir: str,
    algorithms: Optional[List[str]] = None,
) -> Dict:
    """Run all specified algorithms on a corpus and save results.

    Args:
        corpus_dir: Root directory of the PAN-style corpus.
        output_dir: Directory for output JSON and CSV files.
        algorithms: List of algorithm names to run.  ``None`` runs all.

    Returns:
        Benchmark results dict mapping algorithm name → metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load corpus
    manager = PANDatasetManager()
    print(f"Loading corpus from: {corpus_dir}")
    pairs_data = manager.load_corpus(corpus_dir)
    if not pairs_data:
        print("ERROR: No document pairs found in the corpus directory.")
        return {}

    pairs = [(d["text1"], d["text2"]) for d in pairs_data]
    labels = [d["label"] for d in pairs_data]
    print(f"  Loaded {len(pairs)} document pairs ({sum(labels)} plagiarised).")

    # Build algorithms
    algo_objects = _build_algorithms(algorithms)
    if not algo_objects:
        print("ERROR: No valid algorithms selected.")
        return {}

    evaluator = ResearchEvaluator()
    benchmark_results: Dict = {
        "corpus_dir": str(corpus_dir),
        "total_pairs": len(pairs),
        "plagiarised_pairs": int(sum(labels)),
        "algorithms": {},
    }

    for algo in algo_objects:
        algo_name = getattr(algo, "name", repr(algo))
        print(f"\nRunning algorithm: {algo_name} ...")
        start = time.perf_counter()

        preds: List[int] = []
        scores: List[float] = []
        for text1, text2 in pairs:
            try:
                result = algo.detect(text1, text2)
                preds.append(int(result.get("is_plagiarized", False)))
                scores.append(float(result.get("similarity", 0.0)))
            except Exception as exc:  # noqa: BLE001
                logger.error("%s failed on a pair: %s", algo_name, exc)
                preds.append(0)
                scores.append(0.0)

        elapsed = time.perf_counter() - start
        metrics = evaluator.evaluate(preds, labels, scores=scores)
        metrics["elapsed_seconds"] = round(elapsed, 4)
        benchmark_results["algorithms"][algo_name] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}  "
              f"({elapsed:.2f}s)")

    # Print comparison table
    print("\n" + evaluator.format_results_table(benchmark_results["algorithms"]))

    # Save JSON
    json_path = output_path / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(benchmark_results, fh, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Save CSV
    csv_path = output_path / "benchmark_results.csv"
    _save_csv(benchmark_results, csv_path)
    print(f"CSV results saved to: {csv_path}")

    return benchmark_results


def _save_csv(benchmark_results: Dict, csv_path: Path) -> None:
    """Write benchmark results to a CSV file."""
    algo_data = benchmark_results.get("algorithms", {})
    if not algo_data:
        return
    all_metrics = sorted({m for metrics in algo_data.values() for m in metrics})
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["algorithm"] + all_metrics)
        writer.writeheader()
        for algo_name, metrics in algo_data.items():
            row = {"algorithm": algo_name}
            row.update({m: metrics.get(m, "") for m in all_metrics})
            writer.writerow(row)


def compare_algorithms(benchmark_results: Dict) -> str:
    """Return a formatted comparison table from benchmark results.

    Args:
        benchmark_results: Output of :func:`run_benchmark`.

    Returns:
        Formatted string table.
    """
    evaluator = ResearchEvaluator()
    return evaluator.format_results_table(benchmark_results.get("algorithms", {}))


def main() -> None:
    """CLI entry point for the benchmark suite."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run plagiarism detection benchmarks on a PAN-style corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample corpus and all algorithms
  python scripts/run_benchmarks.py

  # Run with a specific corpus and select algorithms
  python scripts/run_benchmarks.py \\
      --corpus-dir data/pan_corpus/sample \\
      --output-dir data/results \\
      --algorithms tfidf ngram embedding
""",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/pan_corpus/sample",
        help="Root directory of the PAN-style corpus (default: data/pan_corpus/sample).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory for output files (default: data/results).",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Algorithms to benchmark (default: all available).",
    )

    args = parser.parse_args()
    corpus_dir = args.corpus_dir

    # Auto-create sample corpus if the directory doesn't exist
    if not Path(corpus_dir).exists():
        print(f"Corpus directory not found: {corpus_dir}")
        print("Creating a sample corpus for demonstration...")
        manager = PANDatasetManager()
        manager.create_sample_corpus(corpus_dir)

    run_benchmark(
        corpus_dir=corpus_dir,
        output_dir=args.output_dir,
        algorithms=args.algorithms,
    )


if __name__ == "__main__":
    main()
