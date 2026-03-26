# generate_report.py (Complete Fixed Version)
from src.plagiarism_detector import PlagiarismDetector
from pathlib import Path
import json
import csv
from datetime import datetime

def generate_plagiarism_report():
    """Generate detailed plagiarism reports (HTML, JSON, CSV)"""
    detector = PlagiarismDetector()
    
    # Load sample documents
    doc_folder = Path("data/sample_documents")
    documents = {}
    doc_names = []
    
    for doc_file in sorted(doc_folder.glob("*.txt")):
        with open(doc_file, 'r', encoding='utf-8') as f:
            documents[doc_file.stem] = f.read()
            doc_names.append(doc_file.stem)
    
    # Compare all
    doc_list = list(documents.values())
    results = detector.compare_documents(doc_list)
    
    # Prepare data for reports
    report_data = []
    for comparison, scores in results.items():
        # Parse comparison key: "doc_0_vs_doc_1"
        parts = comparison.split('_vs_')
        doc1_idx = int(parts[0].replace('doc_', ''))
        doc2_idx = int(parts[1].replace('doc_', ''))
        
        report_data.append({
            'doc1': doc_names[doc1_idx],
            'doc2': doc_names[doc2_idx],
            'tfidf': scores['tfidf_similarity'],
            'sbert': scores['sbert_similarity'],
            'combined': scores['combined_similarity'],
            'plagiarism': scores['plagiarism_detected']
        })
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 1. Generate HTML Report
    generate_html_report(report_data, timestamp)
    
    # 2. Generate JSON Report
    generate_json_report(report_data, timestamp)
    
    # 3. Generate CSV Report
    generate_csv_report(report_data, timestamp)
    
    print("\n✓ All reports generated successfully!")
    print(f"  📊 HTML: data/results/plagiarism_report.html")
    print(f"  📄 JSON: data/results/plagiarism_report.json")
    print(f"  📋 CSV:  data/results/plagiarism_report.csv")

def generate_html_report(data, timestamp):
    """Generate HTML report"""
    plagiarism_count = sum(1 for d in data if d['plagiarism'])
    original_count = len(data) - plagiarism_count
    avg_similarity = sum(d['combined'] for d in data) / len(data) if data else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Plagiarism Detection Report</title>
        <style>
            * {{ margin: 0; padding: 0; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ 
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                padding: 30px;
            }}
            h1 {{ color: #333; margin-bottom: 10px; }}
            .info {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
            table {{ 
                width: 100%; 
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th {{ 
                background-color: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{ padding: 12px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .plagiarism {{ background-color: #ffebee; color: #c62828; font-weight: bold; }}
            .original {{ background-color: #e8f5e9; color: #2e7d32; font-weight: bold; }}
            .high {{ background-color: #fff3e0; }}
            .summary {{ 
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-box {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-value {{ font-size: 24px; font-weight: bold; }}
            .stat-label {{ font-size: 14px; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📋 Plagiarism Detection Report</h1>
            <div class="info">Generated: {timestamp}</div>
            
            <div class="summary">
                <div class="stat-box">
                    <div class="stat-value">{len(data)}</div>
                    <div class="stat-label">Total Comparisons</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{plagiarism_count}</div>
                    <div class="stat-label">Plagiarism Detected</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{original_count}</div>
                    <div class="stat-label">Original</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{avg_similarity:.2%}</div>
                    <div class="stat-label">Avg Similarity</div>
                </div>
            </div>
            
            <table>
                <tr>
                    <th>Document 1</th>
                    <th>Document 2</th>
                    <th>TF-IDF</th>
                    <th>S-BERT</th>
                    <th>Combined</th>
                    <th>Status</th>
                </tr>
    """
    
    for item in data:
        status = "⚠️ PLAGIARISM" if item['plagiarism'] else "✓ ORIGINAL"
        row_class = "plagiarism" if item['plagiarism'] else "original"
        combined_color = "high" if item['combined'] > 0.7 else ""
        
        html_content += f"""
                <tr>
                    <td>{item['doc1']}</td>
                    <td>{item['doc2']}</td>
                    <td>{item['tfidf']:.4f}</td>
                    <td>{item['sbert']:.4f}</td>
                    <td class="{combined_color}">{item['combined']:.4f}</td>
                    <td class="{row_class}">{status}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open("data/results/plagiarism_report.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

def generate_json_report(data, timestamp):
    """Generate JSON report"""
    json_data = {
        'timestamp': timestamp,
        'total_comparisons': len(data),
        'plagiarism_detected': sum(1 for d in data if d['plagiarism']),
        'average_similarity': round(sum(d['combined'] for d in data) / len(data), 4) if data else 0,
        'results': data
    }
    
    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open("data/results/plagiarism_report.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

def generate_csv_report(data, timestamp):
    """Generate CSV report"""
    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open("data/results/plagiarism_report.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['doc1', 'doc2', 'tfidf', 'sbert', 'combined', 'plagiarism'])
        writer.writeheader()
        for item in data:
            writer.writerow({
                'doc1': item['doc1'],
                'doc2': item['doc2'],
                'tfidf': f"{item['tfidf']:.4f}",
                'sbert': f"{item['sbert']:.4f}",
                'combined': f"{item['combined']:.4f}",
                'plagiarism': "YES" if item['plagiarism'] else "NO"
            })

if __name__ == "__main__":
    generate_plagiarism_report()