# gui_plagiarism.py
import PySimpleGUI as sg
from src.plagiarism_detector import PlagiarismDetector
from pathlib import Path
import json
from datetime import datetime

# Set theme
sg.theme('DarkBlue3')
sg.set_options(font=('Arial', 11))

# Initialize detector
detector = PlagiarismDetector()

def create_window():
    """Create main GUI window"""
    
    layout = [
        # Title
        [sg.Text('🔍 PLAGIARISM DETECTOR', font=('Arial', 20, 'bold'), text_color='#00FF00')]
        ,
        [sg.Text('TF-IDF + S-BERT Hybrid Detection System', font=('Arial', 10), text_color='#CCCCCC')]
        ,
        [sg.Separator()]
        ,
        
        # Mode Selection
        [sg.Text('Detection Mode:', font=('Arial', 12, 'bold'))]
        ,
        [
            sg.Radio('Compare Two Texts', 'MODE', default=True, key='MODE_COMPARE'),
            sg.Radio('Compare Multiple Files', 'MODE', key='MODE_BATCH'),
            sg.Radio('Demo Mode', 'MODE', key='MODE_DEMO')
        ]
        ,
        [sg.Separator()]
        ,
        
        # Document 1 Input
        [sg.Text('Document 1 / Primary Text:', font=('Arial', 11, 'bold'))]
        ,
        [
            sg.Multiline(size=(70, 8), key='TEXT1', border_width=2)
        ]
        ,
        [
            sg.Button('Load File', key='LOAD1', size=(12, 1)),
            sg.InputText(key='FILE1', size=(50, 1), visible=False)
        ]
        ,
        [sg.Separator()]
        ,
        
        # Document 2 Input
        [sg.Text('Document 2 / Comparison Text:', font=('Arial', 11, 'bold'))]
        ,
        [
            sg.Multiline(size=(70, 8), key='TEXT2', border_width=2)
        ]
        ,
        [
            sg.Button('Load File', key='LOAD2', size=(12, 1)),
            sg.InputText(key='FILE2', size=(50, 1), visible=False)
        ]
        ,
        [sg.Separator()]
        ,
        
        # Configuration
        [sg.Text('Configuration:', font=('Arial', 11, 'bold'))]
        ,
        [
            sg.Text('TF-IDF Weight:', size=(15, 1)),
            sg.Slider(range=(0, 1), default_value=0.4, resolution=0.1, orientation='h', size=(30, 15), key='TFIDF_WEIGHT'),
            sg.Text('0.4', key='TFIDF_TEXT', size=(5, 1))
        ]
        ,
        [
            sg.Text('S-BERT Weight:', size=(15, 1)),
            sg.Slider(range=(0, 1), default_value=0.6, resolution=0.1, orientation='h', size=(30, 15), key='SBERT_WEIGHT'),
            sg.Text('0.6', key='SBERT_TEXT', size=(5, 1))
        ]
        ,
        [sg.Separator()]
        ,
        
        # Action Buttons
        [
            sg.Button('🔍 Detect Plagiarism', size=(20, 2), button_color=('white', '#00AA00')),
            sg.Button('🧹 Clear All', size=(20, 2)),
            sg.Button('📊 Generate Report', size=(20, 2)),
            sg.Button('❌ Exit', size=(20, 2))
        ]
        ,
        [sg.Separator()]
        ,
        
        # Results Display
        [sg.Text('Results:', font=('Arial', 12, 'bold'))]
        ,
        [
            sg.Output(size=(70, 15), key='OUTPUT', background_color='#1a1a1a', text_color='#00FF00')
        ]
        ,
    ]
    
    return sg.Window('Plagiarism Detector - GUI', layout, finalize=True)

def update_weight_display(window, tfidf_val, sbert_val):
    """Update weight display"""
    window['TFIDF_TEXT'].update(f'{tfidf_val:.1f}')
    window['SBERT_TEXT'].update(f'{sbert_val:.1f}')

def detect_plagiarism(text1, text2, tfidf_w, sbert_w):
    """Run plagiarism detection"""
    if not text1.strip() or not text2.strip():
        print("❌ Error: Both documents must contain text!")
        return None
    
    if len(text1.strip()) < 10 or len(text2.strip()) < 10:
        print("❌ Error: Documents must be at least 10 characters!")
        return None
    
    print("🔄 Analyzing documents...\n")
    
    try:
        result = detector.detect_plagiarism(text1, text2, tfidf_w, sbert_w)
        
        print("=" * 60)
        print("PLAGIARISM DETECTION RESULTS")
        print("=" * 60)
        print(f"\n📊 Similarity Scores:")
        print(f"   TF-IDF Similarity:    {result['tfidf_similarity']:.4f} ({result['tfidf_similarity']*100:.2f}%)")
        print(f"   S-BERT Similarity:    {result['sbert_similarity']:.4f} ({result['sbert_similarity']*100:.2f}%)")
        print(f"   Combined Similarity:  {result['combined_similarity']:.4f} ({result['combined_similarity']*100:.2f}%)")
        
        print(f"\n⚖️ Threshold: 0.7000")
        print(f"📈 Score vs Threshold: {result['combined_similarity']:.4f} vs 0.7000")
        
        print(f"\n🎯 VERDICT:")
        if result['plagiarism_detected']:
            print(f"   ⚠️  PLAGIARISM DETECTED!")
            print(f"   High similarity score indicates potential plagiarism.")
        else:
            print(f"   ✓ ORIGINAL CONTENT")
            print(f"   Low similarity score - content appears to be original.")
        
        print("\n" + "=" * 60)
        
        return result
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def load_file_dialog(window):
    """Open file dialog"""
    filename = sg.popup_get_file('Select a text file', file_types=(("Text Files", "*.txt"), ("All Files", "*.*")))
    return filename

def main():
    """Main GUI loop"""
    window = create_window()
    
    results_history = []
    
    while True:
        event, values = window.read()
        
        # Update weight display
        update_weight_display(window, values['TFIDF_WEIGHT'], values['SBERT_WEIGHT'])
        
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        
        elif event == 'LOAD1':
            filename = load_file_dialog(window)
            if filename:
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    window['TEXT1'].update(content)
                    print(f"✓ Loaded: {filename}")
                except Exception as e:
                    print(f"❌ Error loading file: {e}")
        
        elif event == 'LOAD2':
            filename = load_file_dialog(window)
            if filename:
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    window['TEXT2'].update(content)
                    print(f"✓ Loaded: {filename}")
                except Exception as e:
                    print(f"❌ Error loading file: {e}")
        
        elif event == '🔍 Detect Plagiarism':
            print("\n" + "="*60)
            
            if values['MODE_COMPARE']:
                text1 = values['TEXT1']
                text2 = values['TEXT2']
                tfidf_w = values['TFIDF_WEIGHT']
                sbert_w = values['SBERT_WEIGHT']
                
                result = detect_plagiarism(text1, text2, tfidf_w, sbert_w)
                if result:
                    results_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'text1_preview': text1[:100],
                        'text2_preview': text2[:100],
                        'result': result
                    })
            
            elif values['MODE_BATCH']:
                sg.popup('Batch Mode', 'Batch comparison feature coming soon!')
            
            elif values['MODE_DEMO']:
                print("📋 Running Demo...\n")
                demo_texts = {
                    "Identical": ("Machine learning is AI", "Machine learning is AI"),
                    "Similar": ("The fox jumps over dog", "A fox leaps over a dog"),
                    "Different": ("Python programming", "Cooking recipes")
                }
                
                for demo_name, (text1, text2) in demo_texts.items():
                    print(f"\n📌 Demo: {demo_name}")
                    detect_plagiarism(text1, text2, 0.4, 0.6)
        
        elif event == '🧹 Clear All':
            window['TEXT1'].update('')
            window['TEXT2'].update('')
            window['OUTPUT'].update('')
            print("✓ All fields cleared")
        
        elif event == '📊 Generate Report':
            if results_history:
                report_path = Path('data/results/gui_session_report.json')
                report_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(report_path, 'w') as f:
                    json.dump(results_history, f, indent=2)
                
                print(f"\n✓ Report saved: {report_path}")
            else:
                print("\n❌ No results to report. Run a detection first!")
    
    window.close()

if __name__ == '__main__':
    main()