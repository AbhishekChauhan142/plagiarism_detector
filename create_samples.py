# create_samples.py
from pathlib import Path

# Create data directories
Path("data/sample_documents").mkdir(parents=True, exist_ok=True)
Path("data/results").mkdir(parents=True, exist_ok=True)

# Sample documents
documents = {
    "doc1.txt": """Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models. These models enable computers to improve their performance on tasks through experience and data analysis. Machine learning has applications in various fields including computer vision, natural language processing, and predictive analytics.""",
    
    "doc2.txt": """Machine learning, a branch of AI, involves creating algorithms and statistical models that allow computers to enhance their performance by learning from data. This technology is applied in multiple domains such as image recognition, language understanding, and forecasting systems.""",
    
    "doc3.txt": """Deep learning is a neural network approach within machine learning that uses multiple layers. It has revolutionized computer vision and natural language processing. Deep neural networks can automatically learn the representations needed for detection or classification.""",
    
    "doc4.txt": """Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, data science, artificial intelligence, and scientific computing. Python has a large community and extensive libraries like NumPy, Pandas, and TensorFlow.""",
    
    "doc5.txt": """Cooking is an art and science that combines various ingredients and techniques. It requires understanding of flavors, temperatures, and timing. Professional chefs spend years mastering different cooking methods including baking, grilling, sauteing, and roasting."""
}

# Write files
for filename, content in documents.items():
    filepath = Path("data/sample_documents") / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created {filepath}")

print("\nAll sample documents created successfully!")