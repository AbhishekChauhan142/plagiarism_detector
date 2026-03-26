# src/__init__.py
from .plagiarism_detector import PlagiarismDetector
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