# src/utils.py
from pathlib import Path

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
    print(f"✓ Results saved to {output_file}")