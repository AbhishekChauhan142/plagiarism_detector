"""Enhanced CLI for the plagiarism detection research toolkit.

Provides subcommands for downloading/creating corpora, evaluating algorithms,
comparing document pairs, running benchmarks, and generating reports.

Usage::

    python scripts/research_cli.py --help
    python scripts/research_cli.py download --output-dir data/pan_corpus
    python scripts/research_cli.py compare --text1 "Alice wrote this." --text2 "Alice wrote this."
    python scripts/research_cli.py evaluate --corpus-dir data/pan_corpus/sample
    python scripts/research_cli.py benchmark --corpus-dir data/pan_corpus/sample
    python scripts/research_cli.py report --results-file data/results/benchmark_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_algorithm(name: str, threshold: float):
    """Instantiate an algorithm by name."""
    name = name.lower()
    from src.detection_algorithms import TFIDFAlgorithm, NGramAlgorithm, EmbeddingAlgorithm

    registry = {
        "tfidf": lambda: TFIDFAlgorithm(threshold=threshold),
        "ngram": lambda: NGramAlgorithm(threshold=threshold),
        "embedding": lambda: EmbeddingAlgorithm(threshold=threshold),
    }
    try:
        from src.detection_algorithms import SBERTAlgorithm, EnsembleAlgorithm

        registry["sbert"] = lambda: SBERTAlgorithm(threshold=threshold)
        registry["ensemble"] = lambda: EnsembleAlgorithm(threshold=threshold)
    except ImportError:
        pass

    if name not in registry:
        print(f"ERROR: Unknown algorithm '{name}'. "
              f"Available: {', '.join(registry.keys())}")
        sys.exit(1)
    return registry[name]()


def _print_result(result: dict) -> None:
    """Pretty-print a detection result dict."""
    print("\n" + "=" * 50)
    print(f"  Algorithm    : {result['algorithm']}")
    print(f"  Similarity   : {result['similarity']:.4f}")
    print(f"  Threshold    : {result['threshold']:.2f}")
    verdict = "⚠  PLAGIARISM DETECTED" if result["is_plagiarized"] else "✓  No plagiarism"
    print(f"  Verdict      : {verdict}")
    for extra in ("tfidf_score", "sbert_score", "ngram_score"):
        if extra in result:
            print(f"  {extra:<14}: {result[extra]:.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    """Handle the ``download`` subcommand."""
    from src.pan_dataset_manager import PANDatasetManager

    manager = PANDatasetManager()
    manager.download_corpus(version=args.version, output_dir=args.output_dir)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Handle the ``evaluate`` subcommand."""
    from src.pan_dataset_manager import PANDatasetManager
    from src.research_evaluator import ResearchEvaluator
    from src.detection_algorithms import TFIDFAlgorithm, NGramAlgorithm, EmbeddingAlgorithm

    corpus_dir = args.corpus_dir
    if not Path(corpus_dir).exists():
        print(f"Corpus directory not found: {corpus_dir}")
        print("Run: python scripts/research_cli.py download --output-dir data/pan_corpus")
        sys.exit(1)

    manager = PANDatasetManager()
    pairs_data = manager.load_corpus(corpus_dir)
    if not pairs_data:
        print("No document pairs found.")
        sys.exit(1)

    pairs = [(d["text1"], d["text2"]) for d in pairs_data]
    labels = [d["label"] for d in pairs_data]
    print(f"Loaded {len(pairs)} pairs ({sum(labels)} plagiarised).")

    algo_names = args.algorithms or ["tfidf", "ngram", "embedding"]
    algo_objects = [_get_algorithm(n, args.threshold) for n in algo_names]

    evaluator = ResearchEvaluator()
    comparison = evaluator.compare_algorithms(algo_objects, pairs, labels)

    print("\nEvaluation Results:")
    print(evaluator.format_results_table(comparison))

    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / "evaluation_results.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"\nResults saved to: {out_file}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Handle the ``compare`` subcommand."""
    # Resolve text1 / text2
    text1 = args.text1
    text2 = args.text2

    if args.file1:
        path1 = Path(args.file1)
        if not path1.exists():
            print(f"ERROR: File not found: {args.file1}")
            sys.exit(1)
        text1 = path1.read_text(encoding="utf-8")

    if args.file2:
        path2 = Path(args.file2)
        if not path2.exists():
            print(f"ERROR: File not found: {args.file2}")
            sys.exit(1)
        text2 = path2.read_text(encoding="utf-8")

    if not text1 or not text2:
        print("ERROR: Provide both texts via --text1/--text2 or --file1/--file2.")
        sys.exit(1)

    algo = _get_algorithm(args.algorithm, args.threshold)
    result = algo.detect(text1, text2)
    _print_result(result)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle the ``benchmark`` subcommand."""
    from scripts.run_benchmarks import run_benchmark

    corpus_dir = args.corpus_dir
    if not Path(corpus_dir).exists():
        print(f"Corpus directory not found: {corpus_dir}")
        print("Creating a sample corpus for demonstration...")
        from src.pan_dataset_manager import PANDatasetManager

        PANDatasetManager().create_sample_corpus(corpus_dir)

    run_benchmark(
        corpus_dir=corpus_dir,
        output_dir=args.output_dir,
        algorithms=args.algorithms or None,
    )


def cmd_report(args: argparse.Namespace) -> None:
    """Handle the ``report`` subcommand."""
    from scripts.generate_research_report import (
        load_benchmark_results,
        generate_html_report,
        generate_text_report,
    )

    try:
        results = load_benchmark_results(args.results_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Run 'benchmark' subcommand first to generate results.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    if fmt in ("html", "both"):
        generate_html_report(results, str(output_dir / "research_report.html"))
    if fmt in ("text", "both"):
        generate_text_report(results, str(output_dir / "research_report.txt"))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="research_cli",
        description="Plagiarism Detection Research Toolkit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommand examples:
  python scripts/research_cli.py download --output-dir data/pan_corpus
  python scripts/research_cli.py compare --text1 "Hello world" --text2 "Hello world"
  python scripts/research_cli.py compare --file1 doc1.txt --file2 doc2.txt --algorithm sbert
  python scripts/research_cli.py evaluate --corpus-dir data/pan_corpus/sample --algorithms tfidf ngram
  python scripts/research_cli.py benchmark --corpus-dir data/pan_corpus/sample
  python scripts/research_cli.py report --results-file data/results/benchmark_results.json --format html
""",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose/debug logging."
    )

    sub = parser.add_subparsers(dest="command", title="subcommands")

    # ---- download ----
    dl = sub.add_parser(
        "download",
        help="Download or create a PAN corpus.",
        description=(
            "Provides instructions for downloading the official PAN corpus and "
            "creates a sample corpus for testing."
        ),
    )
    dl.add_argument(
        "--version",
        default="pan11",
        help="PAN corpus version/year (e.g. pan11, pan13). Default: pan11.",
    )
    dl.add_argument(
        "--output-dir",
        default="data/pan_corpus",
        help="Directory to store the corpus. Default: data/pan_corpus.",
    )

    # ---- evaluate ----
    ev = sub.add_parser(
        "evaluate",
        help="Evaluate algorithms on a corpus.",
        description="Runs one or more algorithms on a PAN-style corpus and reports metrics.",
    )
    ev.add_argument(
        "--corpus-dir",
        default="data/pan_corpus/sample",
        help="Root directory of the PAN-style corpus. Default: data/pan_corpus/sample.",
    )
    ev.add_argument(
        "--algorithms",
        nargs="+",
        choices=["tfidf", "sbert", "ngram", "embedding", "ensemble"],
        default=None,
        help="Algorithms to evaluate. Default: tfidf ngram embedding.",
    )
    ev.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Detection threshold (0–1). Default: 0.7.",
    )
    ev.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save evaluation JSON results.",
    )

    # ---- compare ----
    cmp = sub.add_parser(
        "compare",
        help="Compare two texts or files for plagiarism.",
        description="Detect plagiarism between two document inputs using a chosen algorithm.",
    )
    cmp.add_argument("--file1", help="Path to the first text file.")
    cmp.add_argument("--file2", help="Path to the second text file.")
    cmp.add_argument("--text1", help="First text string (alternative to --file1).")
    cmp.add_argument("--text2", help="Second text string (alternative to --file2).")
    cmp.add_argument(
        "--algorithm",
        default="tfidf",
        choices=["tfidf", "sbert", "ngram", "embedding", "ensemble"],
        help="Detection algorithm to use. Default: tfidf.",
    )
    cmp.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Detection threshold (0–1). Default: 0.7.",
    )

    # ---- benchmark ----
    bm = sub.add_parser(
        "benchmark",
        help="Run the full benchmark suite.",
        description="Benchmarks all algorithms on a corpus and saves JSON/CSV results.",
    )
    bm.add_argument(
        "--corpus-dir",
        default="data/pan_corpus/sample",
        help="Root directory of the PAN-style corpus.",
    )
    bm.add_argument(
        "--output-dir",
        default="data/results",
        help="Directory for benchmark output files. Default: data/results.",
    )
    bm.add_argument(
        "--algorithms",
        nargs="+",
        choices=["tfidf", "sbert", "ngram", "embedding", "ensemble"],
        default=None,
        help="Subset of algorithms to benchmark. Default: all available.",
    )

    # ---- report ----
    rp = sub.add_parser(
        "report",
        help="Generate a research report from benchmark results.",
        description="Reads a JSON results file and generates HTML and/or text reports.",
    )
    rp.add_argument(
        "--results-file",
        default="data/results/benchmark_results.json",
        help="Path to the benchmark JSON results file.",
    )
    rp.add_argument(
        "--output-dir",
        default="data/reports",
        help="Directory for output report files. Default: data/reports.",
    )
    rp.add_argument(
        "--format",
        choices=["html", "text", "both"],
        default="both",
        help="Report output format. Default: both.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "download": cmd_download,
        "evaluate": cmd_evaluate,
        "compare": cmd_compare,
        "benchmark": cmd_benchmark,
        "report": cmd_report,
    }

    handler = dispatch.get(command)
    if handler is None:
        print(f"ERROR: Unknown command '{command}'.")
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        logger.error("Command '%s' failed: %s", command, exc, exc_info=True)
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
