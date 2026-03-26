# src/similarity_calculator.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimilarityCalculator:
    """Calculate similarity between texts"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if len(set1.union(set2)) == 0:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))