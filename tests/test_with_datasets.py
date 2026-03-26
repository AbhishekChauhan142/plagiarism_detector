# tests/test_with_datasets.py
import pytest
from pathlib import Path
from src.plagiarism_detector import PlagiarismDetector

class TestWithDatasets:
    @pytest.fixture
    def detector(self):
        return PlagiarismDetector()
    
    @pytest.fixture
    def load_test_file(self):
        def _load(filename):
            filepath = Path(f"data/datasets/test/{filename}")
            with open(filepath, 'r') as f:
                return f.read()
        return _load
    
    def test_identical_documents(self, detector, load_test_file):
        """Test identical documents"""
        doc1 = load_test_file("identical_pair_1.txt")
        doc2 = load_test_file("identical_pair_2.txt")
        
        result = detector.detect_plagiarism(doc1, doc2)
        assert result['combined_similarity'] > 0.95
        assert result['plagiarism_detected'] is True
    
    def test_similar_documents(self, detector, load_test_file):
        """Test similar but paraphrased documents"""
        doc1 = load_test_file("similar_pair_1.txt")
        doc2 = load_test_file("similar_pair_2.txt")
        
        result = detector.detect_plagiarism(doc1, doc2)
        assert result['sbert_similarity'] > 0.7
    
    def test_different_documents(self, detector, load_test_file):
        """Test completely different documents"""
        doc1 = load_test_file("different_pair_1.txt")
        doc2 = load_test_file("different_pair_2.txt")
        
        result = detector.detect_plagiarism(doc1, doc2)
        assert result['combined_similarity'] < 0.5
        assert result['plagiarism_detected'] is False
    
    def test_plagiarism_variants(self, detector):
        """Test against plagiarism variants"""
        plagiarism_dir = Path("data/datasets/plagiarism")
        
        original = plagiarism_dir / "direct_copy.txt"
        if original.exists():
            with open(original) as f:
                original_text = f.read()
            
            # Test direct copy
            direct_copy = plagiarism_dir / "direct_copy.txt"
            if direct_copy.exists():
                with open(direct_copy) as f:
                    copy_text = f.read()
                
                result = detector.detect_plagiarism(original_text, copy_text)
                assert result['combined_similarity'] > 0.95