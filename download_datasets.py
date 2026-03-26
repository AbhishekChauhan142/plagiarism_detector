# download_datasets.py
import os
from pathlib import Path
import urllib.request
import zipfile

def download_pan_corpus():
    """Download PAN plagiarism corpus (Academic dataset)"""
    print("📥 Downloading PAN Plagiarism Corpus...")
    
    # Create datasets folder
    Path("data/datasets").mkdir(parents=True, exist_ok=True)
    
    # PAN corpus URLs (smaller versions for testing)
    urls = {
        'pan13': 'https://zenodo.org/record/3996424/files/pan13-plagiarism-short-document-corpus.zip',
        'pan14': 'https://zenodo.org/record/3996424/files/pan14-plagiarism-detection-corpus.zip'
    }
    
    for name, url in urls.items():
        try:
            filepath = f"data/datasets/{name}.zip"
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)
            
            # Extract
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(f"data/datasets/{name}")
            print(f"✓ {name} downloaded and extracted")
        except Exception as e:
            print(f"Note: {name} download requires manual setup - {e}")

def create_local_datasets():
    """Create local dataset from public sources"""
    print("📝 Creating local plagiarism detection dataset...")
    
    Path("data/datasets/local").mkdir(parents=True, exist_ok=True)
    
    # High-quality academic texts
    academic_texts = {
        "machine_learning_1.txt": """
Machine learning is a subset of artificial intelligence (AI) that provides systems 
the ability to automatically learn and improve from experience without being explicitly 
programmed. Machine learning focuses on the development of computer programs that can 
access data and use it to learn for themselves. The process of learning begins with 
observations or data, such as examples, direct experience, or instruction, in order 
to look for patterns in data and make better decisions in the future based on the 
examples that we provide. The primary aim is to allow computers to learn automatically 
without human intervention or assistance and adjust actions accordingly.""",
        
        "machine_learning_2.txt": """
Artificial Intelligence encompasses machine learning, which is the science of making 
computer systems that can learn and improve by themselves. These systems are able to 
automatically gain knowledge from data and experience without explicit programming. 
Machine learning algorithms process information and identify patterns within datasets, 
enabling systems to make decisions based on learned knowledge. This technology uses 
observations, historical data, and examples as inputs to discover meaningful patterns. 
The objective of machine learning is to create autonomous systems that enhance their 
performance through experience and data analysis.""",
        
        "neural_networks_1.txt": """
Neural networks are computing systems inspired by biological neural networks that 
constitute animal brains. Such artificial neural networks have interconnected nodes 
in layered structures that mimic biological neurons. An artificial neuron receives 
signals, processes them, and can signal neurons connected to it. The signals traveling 
between artificial neurons are real numbers, and the output of each neuron is computed 
by some non-linear function of the sum of its inputs. Neurons and edges typically have 
weights that adjust as learning proceeds. The network learns by modifying weights.""",
        
        "deep_learning_1.txt": """
Deep learning is part of a broader family of machine learning methods based on artificial 
neural networks with representation learning. Learning can be supervised, semi-supervised 
or unsupervised. Deep learning architectures such as convolutional neural networks, 
recurrent neural networks and transformers have been applied to fields including 
computer vision, speech recognition, natural language processing, audio recognition, 
social network filtering, machine translation, bioinformatics, drug discovery and 
game playing. Results on benchmark datasets such as MNIST, CIFAR and ImageNet show 
that deep neural networks are approaching or sometimes exceeding human level performance.""",
        
        "nlp_1.txt": """
Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language. NLP is used to apply machine learning algorithms to text and speech. 
The goal of NLP is to process and analyze large amounts of natural language data. 
Technology like chatbots, machine translation, sentiment analysis, and question 
answering systems rely on NLP. Challenges in NLP include handling ambiguity, 
context-dependence, and the vast diversity of human language.""",
        
        "data_science_1.txt": """
Data science is an inter-disciplinary field that uses scientific methods, processes, 
algorithms and systems to extract knowledge and insights from data in various forms, 
both structured and unstructured. Data science is related to data mining, machine learning 
and big data. Data scientists use machine learning and algorithms to build predictive 
models. The field encompasses statistics, programming, databases, and machine learning. 
Data science aims to use data-driven approaches to solve complex problems and make informed 
business decisions. Modern organizations rely heavily on data science for competitive advantage.""",
        
        "computer_vision_1.txt": """
Computer vision is an interdisciplinary scientific field that deals with how computers 
can gain high-level understanding from digital images and videos. From the perspective 
of engineering, it seeks to understand and automate tasks that the human visual system 
can do. Computer vision tasks include methods for acquiring, processing, analyzing and 
understanding digital images and extracting high-dimensional data from the real world 
to produce numerical or symbolic information. Technologies such as convolutional neural 
networks have dramatically improved the accuracy of computer vision systems. Applications 
include medical imaging, autonomous vehicles, and facial recognition.""",
        
        "python_programming_1.txt": """
Python is an interpreted, high-level, general-purpose programming language. Its design 
philosophy emphasizes code readability with the use of significant whitespace. Python is 
dynamically typed and garbage-collected. It supports multiple programming paradigms 
including procedural, object-oriented, and functional programming. Python was created by 
Guido van Rossum and first released in 1991. The language provides constructs intended to 
enable writing clear programs on both small and large scales. Python is widely used in 
scientific computing, data analysis, artificial intelligence, web development, and automation."""
    }
    
    # Save academic texts
    for filename, content in academic_texts.items():
        filepath = Path("data/datasets/local") / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"✓ Created {len(academic_texts)} academic documents")
    
    return academic_texts

def create_plagiarism_variants():
    """Create plagiarism variants from original texts"""
    print("📋 Creating plagiarism variants...")
    
    Path("data/datasets/plagiarism").mkdir(parents=True, exist_ok=True)
    
    original = """
Machine learning is a subset of artificial intelligence that provides systems 
the ability to automatically learn and improve from experience without being explicitly 
programmed. Machine learning focuses on the development of computer programs that can 
access data and use it to learn for themselves.
"""
    
    # Direct copy (100% plagiarism)
    with open("data/datasets/plagiarism/direct_copy.txt", 'w') as f:
        f.write(original)
    
    # Slight paraphrase (60-80% plagiarism)
    paraphrased = """
Machine learning, a component of artificial intelligence, enables computer systems 
to automatically learn and enhance performance based on experience without explicit 
programming instructions. Machine learning centers on creating computer applications 
that can obtain information and utilize it for self-learning purposes.
"""
    with open("data/datasets/plagiarism/paraphrased.txt", 'w') as f:
        f.write(paraphrased)
    
    # Heavy paraphrase (30-50% plagiarism)
    heavy_paraphrase = """
AI systems that incorporate machine learning can autonomously improve through experience. 
These systems are designed to function without explicit programming, focusing instead 
on developing applications that learn from available data.
"""
    with open("data/datasets/plagiarism/heavy_paraphrase.txt", 'w') as f:
        f.write(heavy_paraphrase)
    
    # Mosaic plagiarism (mixing multiple sources)
    mosaic = """
Artificial intelligence encompasses machine learning, which provides systems the ability 
to automatically learn. Computer programs can access data and use it for self-improvement. 
This technology focuses on developing applications that enhance performance through 
experience without explicit programming instructions.
"""
    with open("data/datasets/plagiarism/mosaic.txt", 'w') as f:
        f.write(mosaic)
    
    print("✓ Created plagiarism variants (direct, paraphrased, heavy, mosaic)")

def create_test_dataset():
    """Create comprehensive test dataset"""
    print("🧪 Creating test dataset...")
    
    test_data = {
        "identical_pair_1.txt": "Natural language processing enables computers to understand and process human language effectively.",
        "identical_pair_2.txt": "Natural language processing enables computers to understand and process human language effectively.",
        
        "similar_pair_1.txt": "Deep learning uses neural networks with multiple layers to process data.",
        "similar_pair_2.txt": "Neural networks with many layers are employed in deep learning to analyze information.",
        
        "different_pair_1.txt": "Python is a programming language used for software development.",
        "different_pair_2.txt": "Cooking requires combining ingredients with proper heat and timing.",
    }
    
    Path("data/datasets/test").mkdir(parents=True, exist_ok=True)
    
    for filename, content in test_data.items():
        with open(f"data/datasets/test/{filename}", 'w') as f:
            f.write(content)
    
    print("✓ Created test dataset pairs")

if __name__ == "__main__":
    print("📊 Setting up Plagiarism Detection Datasets\n")
    
    # Create local datasets
    create_local_datasets()
    
    # Create plagiarism variants
    create_plagiarism_variants()
    
    # Create test dataset
    create_test_dataset()
    
    # Optionally download PAN corpus (requires internet)
    try:
        download_pan_corpus()
    except:
        print("⚠️ PAN corpus download skipped (optional)")
    
    print("\n✅ Dataset setup complete!")
    print("📁 Datasets created in: data/datasets/")
    print("   - local/          : Academic texts")
    print("   - plagiarism/     : Plagiarism variants")
    print("   - test/           : Test pairs")