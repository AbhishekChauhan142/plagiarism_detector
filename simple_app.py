# simple_app.py
from flask import Flask, render_template, request, jsonify
from src.plagiarism_detector import PlagiarismDetector
from pathlib import Path
import json
from datetime import datetime
import PyPDF2
import os
import zipfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

detector = PlagiarismDetector()
UPLOAD_FOLDER = 'data/uploads'
DB_FOLDER = 'data/document_db'
PAN_FOLDER = 'data/datasets'

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(DB_FOLDER).mkdir(parents=True, exist_ok=True)

def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        return None, str(e)
    return text, None

def extract_text_file(file_path):
    """Extract text from .txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(), None
    except Exception as e:
        return None, str(e)

def load_document_database():
    """Load documents from database and PAN datasets"""
    documents = {}
    
    # Load from document_db folder
    db_path = Path(DB_FOLDER)
    for file in db_path.glob('*.txt'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content.strip()) > 50:  # Minimum length
                    documents[f"db_{file.stem}"] = content
        except:
            pass
    
    # Load from PAN datasets
    pan_path = Path(PAN_FOLDER)
    
    # Load from local dataset
    local_path = pan_path / 'local'
    if local_path.exists():
        for file in local_path.glob('*.txt'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content.strip()) > 50:
                        documents[f"pan_local_{file.stem}"] = content
            except:
                pass
    
    # Load from plagiarism variants
    plagiarism_path = pan_path / 'plagiarism'
    if plagiarism_path.exists():
        for file in plagiarism_path.glob('*.txt'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content.strip()) > 50:
                        documents[f"pan_plagiarism_{file.stem}"] = content
            except:
                pass
    
    # Load from test dataset
    test_path = pan_path / 'test'
    if test_path.exists():
        for file in test_path.glob('*.txt'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content.strip()) > 50:
                        documents[f"pan_test_{file.stem}"] = content
            except:
                pass
    
    return documents

def initialize_database():
    """Initialize database with sample documents if empty"""
    db_docs = load_document_database()
    
    if not db_docs:
        # Create sample documents
        samples = {
            'machine_learning': """
Machine learning is a subset of artificial intelligence that provides systems 
the ability to automatically learn and improve from experience without being 
explicitly programmed. Machine learning focuses on the development of computer 
programs that can access data and use it to learn for themselves.
            """,
            'python_basics': """
Python is a high-level, interpreted programming language known for its simplicity 
and readability. It supports multiple programming paradigms and has a large standard 
library. Python is widely used in web development, data science, and automation.
            """,
            'deep_learning': """
Deep learning is a branch of machine learning that uses artificial neural networks 
with multiple layers. It has revolutionized computer vision and natural language processing 
by enabling automatic learning of representations needed for detection and classification.
            """,
        }
        
        for name, content in samples.items():
            file_path = Path(DB_FOLDER) / f"{name}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())

@app.route('/')
def index():
    """Main page"""
    initialize_database()
    return render_template('simple_index.html')

@app.route('/api/check', methods=['POST'])
def check_plagiarism():
    """Check if document is plagiarized"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    allowed_extensions = {'txt', 'pdf'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Only TXT and PDF files allowed'}), 400
    
    try:
        # Extract text
        if file_ext == 'pdf':
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            text, error = extract_pdf_text(file_path)
            if error:
                return jsonify({'error': f'Failed to read PDF: {error}'}), 400
        else:  # txt
            text = file.read().decode('utf-8')
        
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Document too short (min 50 characters)'}), 400
        
        # Load database documents (including PAN dataset)
        db_docs = load_document_database()
        
        if not db_docs:
            return jsonify({'error': 'Document database is empty. Please add documents first.'}), 400
        
        # Check against all documents
        results = []
        for doc_name, doc_text in db_docs.items():
            try:
                similarity = detector.detect_plagiarism(text, doc_text)
                results.append({
                    'document': doc_name.replace('_', ' ').replace('pan ', 'PAN ').replace('db ', ''),
                    'tfidf': similarity['tfidf_similarity'],
                    'sbert': similarity['sbert_similarity'],
                    'combined': similarity['combined_similarity'],
                    'plagiarism': similarity['plagiarism_detected']
                })
            except:
                continue
        
        # Sort by combined similarity (highest first)
        results = sorted(results, key=lambda x: x['combined'], reverse=True)
        
        # Calculate overall plagiarism status
        max_similarity = max([r['combined'] for r in results]) if results else 0
        is_plagiarized = max_similarity > 0.7
        
        return jsonify({
            'filename': file.filename,
            'document_length': len(text),
            'max_similarity': max_similarity,
            'plagiarism_detected': is_plagiarized,
            'matches': results[:10],  # Top 10 matches
            'total_matches': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-to-db', methods=['POST'])
def add_to_database():
    """Add document to database"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    allowed_extensions = {'txt', 'pdf'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Only TXT and PDF files allowed'}), 400
    
    try:
        # Extract text
        if file_ext == 'pdf':
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            text, error = extract_pdf_text(file_path)
            if error:
                return jsonify({'error': f'Failed to read PDF: {error}'}), 400
        else:
            text = file.read().decode('utf-8')
        
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Document too short'}), 400
        
        # Save to database
        doc_name = Path(file.filename).stem
        db_file = Path(DB_FOLDER) / f"{doc_name}.txt"
        
        with open(db_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return jsonify({
            'message': 'Document added to database',
            'filename': doc_name,
            'size': len(text)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/db-stats', methods=['GET'])
def get_db_stats():
    """Get database statistics including PAN dataset info"""
    db_docs = load_document_database()
    
    # Categorize documents
    categories = {
        'database': [],
        'pan_local': [],
        'pan_plagiarism': [],
        'pan_test': [],
    }
    
    for doc_name in db_docs.keys():
        if doc_name.startswith('db_'):
            categories['database'].append(doc_name)
        elif 'pan_local' in doc_name:
            categories['pan_local'].append(doc_name)
        elif 'pan_plagiarism' in doc_name:
            categories['pan_plagiarism'].append(doc_name)
        elif 'pan_test' in doc_name:
            categories['pan_test'].append(doc_name)
    
    return jsonify({
        'total_documents': len(db_docs),
        'categories': categories,
        'total_size': sum(len(text) for text in db_docs.values()),
        'sources': {
            'database': len(categories['database']),
            'pan_local': len(categories['pan_local']),
            'pan_plagiarism': len(categories['pan_plagiarism']),
            'pan_test': len(categories['pan_test']),
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)