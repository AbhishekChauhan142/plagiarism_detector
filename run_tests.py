# run_tests.py
from pathlib import Path
from src.plagiarism_detector import PlagiarismDetector

def load_test_file(filename):
    filepath = Path(f"data/datasets/test/{filename}")
    with open(filepath, 'r') as f:
        return f.read()

def test_identical_documents():
    """Test identical documents"""
    detector = PlagiarismDetector()
    doc1 = load_test_file("identical_pair_1.txt")
    doc2 = load_test_file("identical_pair_2.txt")
    
    result = detector.detect_plagiarism(doc1, doc2)
    assert result['combined_similarity'] > 0.95, f"Expected > 0.95, got {result['combined_similarity']}"
    assert result['plagiarism_detected'] is True
    print("✓ test_identical_documents PASSED")

def test_similar_documents():
    """Test similar but paraphrased documents"""
    detector = PlagiarismDetector()
    doc1 = load_test_file("similar_pair_1.txt")
    doc2 = load_test_file("similar_pair_2.txt")
    
    result = detector.detect_plagiarism(doc1, doc2)
    assert result['sbert_similarity'] > 0.7, f"Expected > 0.7, got {result['sbert_similarity']}"
    print("✓ test_similar_documents PASSED")

def test_different_documents():
    """Test completely different documents"""
    detector = PlagiarismDetector()
    doc1 = load_test_file("different_pair_1.txt")
    doc2 = load_test_file("different_pair_2.txt")
    
    result = detector.detect_plagiarism(doc1, doc2)
    assert result['combined_similarity'] < 0.5, f"Expected < 0.5, got {result['combined_similarity']}"
    assert result['plagiarism_detected'] is False
    print("✓ test_different_documents PASSED")

def test_plagiarism_variants():
    """Test against plagiarism variants"""
    detector = PlagiarismDetector()
    plagiarism_dir = Path("data/datasets/plagiarism")
    
    # Load original
    with open(plagiarism_dir / "direct_copy.txt") as f:
        original_text = f.read()
    
    # Test direct copy
    with open(plagiarism_dir / "direct_copy.txt") as f:
        copy_text = f.read()
    
    result = detector.detect_plagiarism(original_text, copy_text)
    assert result['combined_similarity'] > 0.95
    print("✓ test_plagiarism_variants - direct_copy PASSED")
    
    # Test paraphrased
    with open(plagiarism_dir / "paraphrased.txt") as f:
        paraphrased_text = f.read()
    
    result = detector.detect_plagiarism(original_text, paraphrased_text)
    assert result['sbert_similarity'] > 0.5
    print("✓ test_plagiarism_variants - paraphrased PASSED")
    
    # Test heavy paraphrase
    with open(plagiarism_dir / "heavy_paraphrase.txt") as f:
        heavy_text = f.read()
    
    result = detector.detect_plagiarism(original_text, heavy_text)
    print(f"  Heavy paraphrase similarity: {result['combined_similarity']:.4f}")
    print("✓ test_plagiarism_variants - heavy_paraphrase PASSED")
    
    # Test mosaic
    with open(plagiarism_dir / "mosaic.txt") as f:
        mosaic_text = f.read()
    
    result = detector.detect_plagiarism(original_text, mosaic_text)
    print(f"  Mosaic similarity: {result['combined_similarity']:.4f}")
    print("✓ test_plagiarism_variants - mosaic PASSED")

if __name__ == "__main__":
    print("🧪 Running Plagiarism Detector Tests\n")
    print("=" * 60)
    
    try:
        test_identical_documents()
        test_similar_documents()
        test_different_documents()
        test_plagiarism_variants()
        
        print("=" * 60)
        print("\n✅ All tests PASSED!\n")
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")