# src/__init__.py
try:
    from .plagiarism_detector import PlagiarismDetector
except ImportError:
    PlagiarismDetector = None  # type: ignore[assignment,misc]

from .preprocessor import TextPreprocessor
from .similarity_calculator import SimilarityCalculator
from .utils import load_document, save_results

__all__ = [
    'PlagiarismDetector',
    'TextPreprocessor',
    'SimilarityCalculator',
    'load_document',
    'save_results'
]