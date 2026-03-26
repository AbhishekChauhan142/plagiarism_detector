# app.py
from flask import Flask, render_template, request, jsonify
from src.plagiarism_detector import PlagiarismDetector
from pathlib import Path
import json
from datetime import datetime

app = Flask(__name__)
detector = PlagiarismDetector()
results_history = []

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    """Detect plagiarism"""
    data = request.json
    
    text1 = data.get('text1', '').strip()
    text2 = data.get('text2', '').strip()
    tfidf_weight = float(data.get('tfidf_weight', 0.4))
    sbert_weight = float(data.get('sbert_weight', 0.6))
    
    if not text1 or not text2:
        return jsonify({'error': 'Both documents required'}), 400
    
    if len(text1) < 10 or len(text2) < 10:
        return jsonify({'error': 'Documents must be at least 10 characters'}), 400
    
    try:
        result = detector.detect_plagiarism(text1, text2, tfidf_weight, sbert_weight)
        
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'text1_preview': text1[:100],
            'text2_preview': text2[:100],
            'result': result
        }
        results_history.append(result_entry)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo', methods=['POST'])
def demo():
    """Run demo"""
    demos = {
        'identical': ('Machine learning is AI', 'Machine learning is AI'),
        'paraphrased': ('The quick brown fox jumps', 'A fast brown fox leaps'),
        'different': ('Python programming', 'Cooking recipes')
    }
    
    demo_type = request.json.get('type', 'identical')
    text1, text2 = demos.get(demo_type, demos['identical'])
    
    try:
        result = detector.detect_plagiarism(text1, text2, 0.4, 0.6)
        return jsonify({
            'text1': text1,
            'text2': text2,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['GET'])
def get_report():
    """Get report"""
    if not results_history:
        return jsonify({'error': 'No results yet'}), 400
    
    report_path = Path('data/results/web_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results_history, f, indent=2)
    
    return jsonify({
        'message': 'Report generated',
        'path': str(report_path),
        'results': results_history
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)