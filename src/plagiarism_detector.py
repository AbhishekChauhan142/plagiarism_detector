# src/plagiarism_detector.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from .preprocessor import TextPreprocessor

class PlagiarismDetector:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with S-BERT model and TF-IDF vectorizer"""
        self.sbert_model = SentenceTransformer(model_name)
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.preprocessor = TextPreprocessor()
    
    def get_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF based similarity"""
        processed_text1 = self.preprocessor.preprocess_text(text1)
        processed_text2 = self.preprocessor.preprocess_text(text2)
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    
    def get_sbert_similarity(self, text1: str, text2: str) -> float:
        """Calculate S-BERT semantic similarity"""
        embedding1 = self.sbert_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sbert_model.encode(text2, convert_to_tensor=True)
        
        similarity = cosine_similarity(
            embedding1.cpu().numpy().reshape(1, -1),
            embedding2.cpu().numpy().reshape(1, -1)
        )[0][0]
        return float(similarity)
    
    def detect_plagiarism(
        self, 
        text1: str, 
        text2: str,
        tfidf_weight: float = 0.4,
        sbert_weight: float = 0.6
    ) -> Dict[str, float]:
        """
        Detect plagiarism using combined TF-IDF and S-BERT scores
        """
        tfidf_score = self.get_tfidf_similarity(text1, text2)
        sbert_score = self.get_sbert_similarity(text1, text2)
        
        total_weight = tfidf_weight + sbert_weight
        tfidf_weight /= total_weight
        sbert_weight /= total_weight
        
        combined_score = (tfidf_score * tfidf_weight) + (sbert_score * sbert_weight)
        
        return {
            'tfidf_similarity': round(tfidf_score, 4),
            'sbert_similarity': round(sbert_score, 4),
            'combined_similarity': round(combined_score, 4),
            'plagiarism_detected': combined_score > 0.7
        }
    
    def compare_documents(self, documents: List[str]) -> Dict:
        """Compare multiple documents for plagiarism"""
        results = {}
        
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                comparison_key = f"doc_{i}_vs_doc_{j}"
                results[comparison_key] = self.detect_plagiarism(
                    documents[i], 
                    documents[j]
                )
        
        return results