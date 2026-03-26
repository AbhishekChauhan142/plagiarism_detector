# main.py
import sys
from pathlib import Path
from src.plagiarism_detector import PlagiarismDetector

def load_document(file_path: str) -> str:
    """Load document from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None

def save_results(results: dict, output_file: str = "data/results/comparison_results.txt"):
    """Save comparison results to file"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== PLAGIARISM DETECTION RESULTS ===\n\n")
        for key, value in results.items():
            f.write(f"{key}:\n")
            for metric, score in value.items():
                f.write(f"  {metric}: {score}\n")
            f.write("\n")
    print(f"Results saved to {output_file}")

def interactive_mode():
    """Interactive mode for pairwise comparison"""
    detector = PlagiarismDetector()
    
    print("\n=== PLAGIARISM DETECTOR - INTERACTIVE MODE ===\n")
    
    # Get first text
    print("1. Enter first document (or 'file:path/to/file' to load from file):")
    text1_input = input("> ").strip()
    
    if text1_input.startswith("file:"):
        text1 = load_document(text1_input[5:])
        if text1 is None:
            return
    else:
        text1 = text1_input
    
    # Get second text
    print("\n2. Enter second document (or 'file:path/to/file' to load from file):")
    text2_input = input("> ").strip()
    
    if text2_input.startswith("file:"):
        text2 = load_document(text2_input[5:])
        if text2 is None:
            return
    else:
        text2 = text2_input
    
    # Get weights
    print("\n3. Enter TF-IDF weight (default 0.4):")
    try:
        tfidf_w = float(input("> ") or 0.4)
    except ValueError:
        tfidf_w = 0.4
    
    print("4. Enter S-BERT weight (default 0.6):")
    try:
        sbert_w = float(input("> ") or 0.6)
    except ValueError:
        sbert_w = 0.6
    
    # Run detection
    print("\nAnalyzing documents...\n")
    result = detector.detect_plagiarism(text1, text2, tfidf_w, sbert_w)
    
    # Display results
    print("=" * 50)
    print("RESULTS:")
    print("=" * 50)
    print(f"TF-IDF Similarity:    {result['tfidf_similarity']:.4f}")
    print(f"S-BERT Similarity:    {result['sbert_similarity']:.4f}")
    print(f"Combined Similarity:  {result['combined_similarity']:.4f}")
    print(f"Plagiarism Detected:  {'YES ⚠️' if result['plagiarism_detected'] else 'NO ✓'}")
    print("=" * 50)
    
    # Save results
    save_choice = input("\nSave results? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_results({'comparison': result})

def batch_mode():
    """Batch mode for comparing multiple documents"""
    detector = PlagiarismDetector()
    
    print("\n=== PLAGIARISM DETECTOR - BATCH MODE ===\n")
    print("Enter file paths (one per line, empty line to finish):")
    
    files = []
    while True:
        file_path = input("> ").strip()
        if not file_path:
            break
        files.append(file_path)
    
    if len(files) < 2:
        print("Error: Need at least 2 files")
        return
    
    # Load documents
    documents = []
    for file_path in files:
        doc = load_document(file_path)
        if doc is None:
            return
        documents.append(doc)
    
    # Compare all documents
    print(f"\nComparing {len(documents)} documents...\n")
    results = detector.compare_documents(documents)
    
    # Display results
    print("=" * 60)
    print("BATCH COMPARISON RESULTS:")
    print("=" * 60)
    for comparison, scores in results.items():
        print(f"\n{comparison}:")
        print(f"  TF-IDF:     {scores['tfidf_similarity']:.4f}")
        print(f"  S-BERT:     {scores['sbert_similarity']:.4f}")
        print(f"  Combined:   {scores['combined_similarity']:.4f}")
        print(f"  Status:     {'⚠️ PLAGIARISM' if scores['plagiarism_detected'] else '✓ ORIGINAL'}")
    print("=" * 60)
    
    # Save results
    save_choice = input("\nSave results? (y/n): ").strip().lower()
    if save_choice == 'y':
        save_results(results)

def demo_mode():
    """Demo mode with sample texts"""
    detector = PlagiarismDetector()
    
    print("\n=== PLAGIARISM DETECTOR - DEMO MODE ===\n")
    
    # Sample texts
    sample_docs = {
        1: {
            "title": "Identical texts",
            "doc1": "Machine learning is a subset of artificial intelligence",
            "doc2": "Machine learning is a subset of artificial intelligence"
        },
        2: {
            "title": "Paraphrased content",
            "doc1": "The quick brown fox jumps over the lazy dog",
            "doc2": "A fast brown fox leaps over a lazy dog"
        },
        3: {
            "title": "Different texts",
            "doc1": "Python is a programming language",
            "doc2": "Cooking requires mixing ingredients and heat"
        }
    }
    
    print("Available demos:")
    for key, value in sample_docs.items():
        print(f"{key}. {value['title']}")
    
    choice = input("\nSelect demo (1-3): ").strip()
    
    if choice not in sample_docs:
        print("Invalid choice")
        return
    
    demo = sample_docs[int(choice)]
    print(f"\nRunning: {demo['title']}\n")
    
    result = detector.detect_plagiarism(demo['doc1'], demo['doc2'])
    
    print(f"Document 1: {demo['doc1'][:50]}...")
    print(f"Document 2: {demo['doc2'][:50]}...\n")
    print("=" * 50)
    print(f"TF-IDF Similarity:    {result['tfidf_similarity']:.4f}")
    print(f"S-BERT Similarity:    {result['sbert_similarity']:.4f}")
    print(f"Combined Similarity:  {result['combined_similarity']:.4f}")
    print(f"Plagiarism Detected:  {'YES ⚠️' if result['plagiarism_detected'] else 'NO ✓'}")
    print("=" * 50)

def main():
    """Main menu"""
    print("\n╔════════════════════════════════════════╗")
    print("║   PLAGIARISM DETECTOR - TF-IDF + SBERT║")
    print("╚════════════════════════════════════════╝\n")
    
    while True:
        print("\nSelect Mode:")
        print("1. Interactive (compare 2 documents)")
        print("2. Batch (compare multiple documents)")
        print("3. Demo (run sample comparisons)")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            interactive_mode()
        elif choice == '2':
            batch_mode()
        elif choice == '3':
            demo_mode()
        elif choice == '4':
            print("\nThank you for using Plagiarism Detector!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)