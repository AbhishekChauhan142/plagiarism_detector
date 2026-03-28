# Plagiarism Detection Research Toolkit

A research-grade plagiarism detection system that implements and benchmarks multiple
detection algorithms on PAN-style corpora, complete with evaluation metrics,
visualization, and report generation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Algorithm Descriptions](#algorithm-descriptions)
6. [Evaluation Metrics](#evaluation-metrics)
7. [PAN Dataset](#pan-dataset)
8. [Usage Examples](#usage-examples)
9. [Example Results](#example-results)
10. [References](#references)

---

## Project Overview

**Objectives**

- Implement and compare multiple plagiarism detection algorithms.
- Evaluate algorithms on PAN shared-task corpora with standard research metrics.
- Support direct-copy, paraphrase, and mosaic plagiarism types.
- Provide reproducible benchmark pipelines and human-readable research reports.

**Key Features**

| Feature | Description |
|---------|-------------|
| 5 algorithms | TF-IDF, S-BERT, N-Gram, Word-Embedding, Ensemble |
| PAN corpus support | XML annotation parsing, corpus loading, sample generation |
| Research metrics | Precision, Recall, F1, Accuracy, AUC-ROC |
| Visualizations | ROC curves, confusion matrices, score distributions |
| Reports | HTML and plain-text report generation |
| CLI | `research_cli.py` with subcommands |

---

## Architecture

```
plagiarism_detector/
├── config/
│   └── config.py                  # Global configuration (thresholds, model names)
├── src/
│   ├── plagiarism_detector.py     # Core detector (TF-IDF + S-BERT combined)
│   ├── preprocessor.py            # Text preprocessing
│   ├── pan_dataset_manager.py     # PAN corpus loading and XML parsing
│   ├── research_evaluator.py      # Metrics, cross-validation, significance tests
│   ├── detection_algorithms/
│   │   ├── __init__.py
│   │   ├── tfidf_algorithm.py     # TF-IDF cosine similarity
│   │   ├── sbert_algorithm.py     # Sentence-BERT semantic similarity
│   │   ├── ngram_algorithm.py     # Character n-gram Jaccard similarity
│   │   ├── embedding_algorithm.py # TF-IDF weighted document vectors
│   │   └── ensemble_algorithm.py  # Weighted ensemble
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py       # matplotlib plotting helpers
│       └── statistics.py          # Bootstrap CI, Cohen's d, Wilcoxon test
├── scripts/
│   ├── run_benchmarks.py          # Benchmark suite (JSON + CSV output)
│   ├── generate_research_report.py # HTML / text report generator
│   └── research_cli.py            # Unified CLI with subcommands
├── data/
│   ├── pan_corpus/                # PAN corpus files
│   ├── reports/                   # Generated reports
│   └── results/                   # Benchmark JSON/CSV output
├── tests/                         # pytest test suite
├── requirements.txt               # Base dependencies
└── research_requirements.txt      # Additional research dependencies
```

---

## Installation

### 1. Base dependencies

```bash
pip install -r requirements.txt
```

### 2. Research dependencies

```bash
pip install -r research_requirements.txt
```

### 3. (Optional) GPU support for S-BERT

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

```bash
# 1. Create a sample corpus
python scripts/research_cli.py download --output-dir data/pan_corpus

# 2. Compare two texts
python scripts/research_cli.py compare \
    --text1 "Machine learning is a subset of AI." \
    --text2 "Machine learning is a subset of AI." \
    --algorithm tfidf

# 3. Run a benchmark
python scripts/research_cli.py benchmark \
    --corpus-dir data/pan_corpus/sample \
    --output-dir data/results

# 4. Generate a report
python scripts/research_cli.py report \
    --results-file data/results/benchmark_results.json \
    --output-dir data/reports \
    --format both
```

---

## Algorithm Descriptions

### TF-IDF (Term Frequency–Inverse Document Frequency)

Converts each document into a TF-IDF weighted bag-of-words vector and returns
the cosine similarity. Fast, interpretable, and effective for lexical overlap.

- **Strengths:** Fast, no external model required, handles direct copies well.
- **Weaknesses:** Misses paraphrase and semantic similarity.

### S-BERT (Sentence-BERT)

Encodes documents with a pre-trained transformer model
(`all-MiniLM-L6-v2` by default) and computes cosine similarity between the
sentence embeddings. Captures semantic meaning rather than surface lexical form.

- **Strengths:** Handles paraphrase and semantic reuse.
- **Weaknesses:** Slower inference; requires `sentence-transformers`.

### N-Gram (Character N-gram Jaccard)

Extracts overlapping character trigrams (default `n=3`) and computes the
Jaccard index between the two n-gram sets. Robust to minor character-level edits.

- **Strengths:** Language-agnostic; robust to spelling variants.
- **Weaknesses:** Lower recall on paraphrase plagiarism.

### Word-Embedding (TF-IDF Weighted Document Vectors)

Fits a shared TF-IDF vocabulary over both documents and computes cosine
similarity between their dense TF-IDF vectors. Captures more vocabulary-level
information than pure bag-of-words.

- **Strengths:** Considers term importance across documents.
- **Weaknesses:** Still vocabulary-dependent; no semantic generalisation.

### Ensemble

Combines TF-IDF, S-BERT and N-Gram via a configurable weighted average
(defaults: TF-IDF 35 %, S-BERT 45 %, N-Gram 20 %). Returns individual component
scores alongside the combined score.

- **Strengths:** More robust across plagiarism types.
- **Weaknesses:** Requires all component dependencies; slower than single algorithms.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | TP / (TP + FP) — fraction of detected plagiarism that is real |
| **Recall** | TP / (TP + FN) — fraction of real plagiarism that is detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **Accuracy** | (TP + TN) / total |
| **AUC-ROC** | Area under the receiver operating characteristic curve |

Cross-validation (stratified k-fold), statistical significance tests
(Wilcoxon signed-rank), and Cohen's d effect sizes are also available via
`ResearchEvaluator` and `src/utils/statistics.py`.

---

## PAN Dataset

The [PAN shared tasks](https://pan.webis.de/) provide standard corpora for
plagiarism detection research. Each corpus contains:

- **source-documents/** — original source text files (`.txt`)
- **suspicious-documents/** — documents that may contain plagiarised passages
- **XML annotation files** — one per suspicious document, marking plagiarised spans

### Obtaining the official corpus

1. Visit <https://pan.webis.de/data.html>
2. Locate the plagiarism detection task for your target year (e.g. PAN-11, PAN-13)
3. Register and download the corpus archive
4. Extract to `data/pan_corpus/<version>/`

### Using the sample corpus

A built-in sample corpus with 12 document pairs (direct copy, paraphrase, mosaic,
and clean) can be created automatically:

```bash
python scripts/research_cli.py download --output-dir data/pan_corpus
# or directly:
python -c "from src.pan_dataset_manager import PANDatasetManager; \
           PANDatasetManager().create_sample_corpus('data/pan_corpus/sample')"
```

---

## Usage Examples

### Python API

```python
from src.detection_algorithms import TFIDFAlgorithm, SBERTAlgorithm, EnsembleAlgorithm
from src.pan_dataset_manager import PANDatasetManager
from src.research_evaluator import ResearchEvaluator

# Single-pair detection
algo = TFIDFAlgorithm(threshold=0.7)
result = algo.detect("Original text here.", "Copied text here.")
print(result)
# {'similarity': 0.8123, 'is_plagiarized': True, 'algorithm': 'TF-IDF', 'threshold': 0.7}

# Load a corpus
manager = PANDatasetManager()
pairs_data = manager.load_corpus("data/pan_corpus/sample")
pairs  = [(d["text1"], d["text2"]) for d in pairs_data]
labels = [d["label"] for d in pairs_data]

# Compare algorithms
evaluator = ResearchEvaluator()
comparison = evaluator.compare_algorithms(
    [TFIDFAlgorithm(), EnsembleAlgorithm()], pairs, labels
)
print(evaluator.format_results_table(comparison))
```

### CLI subcommands

```bash
# Download / create corpus
python scripts/research_cli.py download --version pan11 --output-dir data/pan_corpus

# Compare files
python scripts/research_cli.py compare \
    --file1 doc1.txt --file2 doc2.txt --algorithm ensemble --threshold 0.6

# Evaluate specific algorithms
python scripts/research_cli.py evaluate \
    --corpus-dir data/pan_corpus/sample \
    --algorithms tfidf ngram embedding \
    --threshold 0.65

# Full benchmark
python scripts/research_cli.py benchmark \
    --corpus-dir data/pan_corpus/sample \
    --output-dir data/results

# Generate report
python scripts/research_cli.py report \
    --results-file data/results/benchmark_results.json \
    --format html
```

---

## Example Results

The table below shows indicative results on the built-in sample corpus.
Actual numbers will vary with corpus size and algorithm parameters.

| Algorithm     | Accuracy | Precision | Recall |    F1  | AUC-ROC |
|---------------|----------|-----------|--------|--------|---------|
| TF-IDF        |  0.8333  |  0.8750   | 0.8750 | 0.8750 |  0.906  |
| S-BERT        |  0.9167  |  0.8889   | 1.0000 | 0.9412 |  0.969  |
| N-Gram        |  0.7500  |  0.7778   | 0.8750 | 0.8235 |  0.844  |
| Word-Embedding|  0.8333  |  0.8750   | 0.8750 | 0.8750 |  0.906  |
| Ensemble      |  0.9167  |  1.0000   | 0.8750 | 0.9333 |  0.969  |

*Results are illustrative; run the benchmark on your corpus for accurate figures.*

---

## References

- Potthast, M. et al. (2010). *An Evaluation Framework for Plagiarism Detection.*
  COLING 2010. <https://pan.webis.de/>
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks.* EMNLP 2019. <https://www.sbert.net/>
- Scikit-learn: Machine Learning in Python. <https://scikit-learn.org/>
- PAN Shared Tasks on Plagiarism Detection. <https://pan.webis.de/clef11/pan11-web/plagiarism-detection.html>
