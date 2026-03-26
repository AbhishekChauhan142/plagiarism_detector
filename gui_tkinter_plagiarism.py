# gui_tkinter_plagiarism.py
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from src.plagiarism_detector import PlagiarismDetector
from pathlib import Path
import threading
from datetime import datetime
import json

class PlagiarismDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Plagiarism Detector - GUI")
        self.root.geometry("1200x900")
        self.root.configure(bg='#1e1e1e')
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#1e1e1e')
        self.style.configure('TLabel', background='#1e1e1e', foreground='#ffffff')
        self.style.configure('TButton', background='#0d7377', foreground='#ffffff')
        self.style.map('TButton', background=[('active', '#14a085')])
        
        self.detector = PlagiarismDetector()
        self.results_history = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI components"""
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#2d2d2d', pady=15)
        title_frame.pack(fill=tk.X)
        
        title = tk.Label(
            title_frame,
            text='🔍 PLAGIARISM DETECTOR',
            font=('Arial', 24, 'bold'),
            fg='#00FF00',
            bg='#2d2d2d'
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text='TF-IDF + S-BERT Hybrid Detection System',
            font=('Arial', 11),
            fg='#CCCCCC',
            bg='#2d2d2d'
        )
        subtitle.pack()
        
        # Separator
        sep = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Input
        left_frame = tk.Frame(main_frame, bg='#1e1e1e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Document 1
        doc1_label = tk.Label(
            left_frame,
            text="📄 Document 1:",
            font=('Arial', 11, 'bold'),
            fg='#00FF00',
            bg='#1e1e1e'
        )
        doc1_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.text1 = scrolledtext.ScrolledText(
            left_frame, height=12, width=50,
            bg='#2d2d2d', fg='#00FF00',
            insertbackground='#00FF00',
            font=('Courier', 10)
        )
        self.text1.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        btn1_frame = tk.Frame(left_frame, bg='#1e1e1e')
        btn1_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn1_frame, text="📂 Load File 1", command=self.load_file1).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn1_frame, text="🗑️ Clear 1", command=lambda: self.text1.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=2)
        
        # Document 2
        doc2_label = tk.Label(
            left_frame,
            text="📄 Document 2:",
            font=('Arial', 11, 'bold'),
            fg='#00FF00',
            bg='#1e1e1e'
        )
        doc2_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.text2 = scrolledtext.ScrolledText(
            left_frame, height=12, width=50,
            bg='#2d2d2d', fg='#00FF00',
            insertbackground='#00FF00',
            font=('Courier', 10)
        )
        self.text2.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        btn2_frame = tk.Frame(left_frame, bg='#1e1e1e')
        btn2_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn2_frame, text="📂 Load File 2", command=self.load_file2).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn2_frame, text="🗑️ Clear 2", command=lambda: self.text2.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=2)
        
        # Right panel - Controls & Results
        right_frame = tk.Frame(main_frame, bg='#1e1e1e')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Configuration Frame
        config_frame = tk.LabelFrame(
            right_frame,
            text="⚙️ Configuration",
            font=('Arial', 10, 'bold'),
            bg='#2d2d2d',
            fg='#00FF00',
            padx=10, pady=10
        )
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # TF-IDF Weight
        tfidf_label = tk.Label(
            config_frame,
            text="TF-IDF Weight (Lexical):",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#CCCCCC'
        )
        tfidf_label.pack(anchor=tk.W, pady=(0, 3))
        
        tfidf_frame = tk.Frame(config_frame, bg='#2d2d2d')
        tfidf_frame.pack(fill=tk.X, pady=5)
        
        self.tfidf_var = tk.DoubleVar(value=0.4)
        self.tfidf_scale = ttk.Scale(
            tfidf_frame, from_=0, to=1, orient=tk.HORIZONTAL,
            variable=self.tfidf_var, command=self.update_weights
        )
        self.tfidf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.tfidf_label = tk.Label(
            tfidf_frame,
            text="0.40",
            font=('Arial', 11, 'bold'),
            bg='#2d2d2d',
            fg='#00FF00',
            width=5
        )
        self.tfidf_label.pack(side=tk.LEFT)
        
        # S-BERT Weight
        sbert_label = tk.Label(
            config_frame,
            text="S-BERT Weight (Semantic):",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#CCCCCC'
        )
        sbert_label.pack(anchor=tk.W, pady=(10, 3))
        
        sbert_frame = tk.Frame(config_frame, bg='#2d2d2d')
        sbert_frame.pack(fill=tk.X, pady=5)
        
        self.sbert_var = tk.DoubleVar(value=0.6)
        self.sbert_scale = ttk.Scale(
            sbert_frame, from_=0, to=1, orient=tk.HORIZONTAL,
            variable=self.sbert_var, command=self.update_weights
        )
        self.sbert_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.sbert_label = tk.Label(
            sbert_frame,
            text="0.60",
            font=('Arial', 11, 'bold'),
            bg='#2d2d2d',
            fg='#00FF00',
            width=5
        )
        self.sbert_label.pack(side=tk.LEFT)
        
        # Buttons
        button_frame = tk.Frame(right_frame, bg='#1e1e1e')
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame,
            text="🔍 Detect Plagiarism",
            command=self.detect,
            width=20
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="📋 Demo Mode",
            command=self.demo,
            width=20
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="📊 Generate Report",
            command=self.generate_report,
            width=20
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="🧹 Clear All",
            command=self.clear_all,
            width=20
        ).pack(fill=tk.X, pady=5)
        
        # Results Frame
        results_frame = tk.LabelFrame(
            right_frame,
            text="📋 Results",
            font=('Arial', 10, 'bold'),
            bg='#2d2d2d',
            fg='#00FF00',
            padx=10, pady=10
        )
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=25, width=50,
            bg='#1a1a1a', fg='#00FF00',
            insertbackground='#00FF00',
            font=('Courier', 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def update_weights(self, val):
        """Update weight display"""
        self.tfidf_label.config(text=f"{self.tfidf_var.get():.2f}")
        self.sbert_label.config(text=f"{self.sbert_var.get():.2f}")
    
    def load_file1(self):
        """Load file 1"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.text1.delete('1.0', tk.END)
                self.text1.insert('1.0', content)
                self.log(f"✓ Loaded Document 1: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def load_file2(self):
        """Load file 2"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.text2.delete('1.0', tk.END)
                self.text2.insert('1.0', content)
                self.log(f"✓ Loaded Document 2: {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def detect(self):
        """Run detection"""
        text1 = self.text1.get('1.0', tk.END).strip()
        text2 = self.text2.get('1.0', tk.END).strip()
        
        if not text1 or not text2:
            messagebox.showerror("Error", "Both documents must contain text!")
            return
        
        if len(text1) < 10 or len(text2) < 10:
            messagebox.showerror("Error", "Documents must be at least 10 characters!")
            return
        
        # Run in thread to prevent UI freeze
        thread = threading.Thread(
            target=self._detect_thread,
            args=(text1, text2, self.tfidf_var.get(), self.sbert_var.get())
        )
        thread.daemon = True
        thread.start()
    
    def _detect_thread(self, text1, text2, tfidf_w, sbert_w):
        """Detection thread"""
        self.log("\n" + "="*50)
        self.log("🔄 Analyzing documents...\n")
        
        try:
            result = self.detector.detect_plagiarism(text1, text2, tfidf_w, sbert_w)
            
            self.log("="*50)
            self.log("PLAGIARISM DETECTION RESULTS")
            self.log("="*50)
            self.log(f"\n📊 Similarity Scores:")
            self.log(f"   TF-IDF:    {result['tfidf_similarity']:.4f} ({result['tfidf_similarity']*100:.2f}%)")
            self.log(f"   S-BERT:    {result['sbert_similarity']:.4f} ({result['sbert_similarity']*100:.2f}%)")
            self.log(f"   Combined:  {result['combined_similarity']:.4f} ({result['combined_similarity']*100:.2f}%)")
            
            self.log(f"\n⚖️ Threshold: 0.7000")
            self.log(f"📈 Score vs Threshold: {result['combined_similarity']:.4f} vs 0.7000")
            
            self.log(f"\n🎯 VERDICT:")
            if result['plagiarism_detected']:
                self.log(f"   ⚠️  PLAGIARISM DETECTED!")
                self.log(f"   High similarity indicates potential plagiarism.")
            else:
                self.log(f"   ✓ ORIGINAL CONTENT")
                self.log(f"   Low similarity - content appears original.")
            
            self.log("\n" + "="*50 + "\n")
            
            self.results_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': result
            })
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}\n")
    
    def demo(self):
        """Run demo"""
        self.log("\n" + "="*50)
        self.log("📋 DEMO MODE - Running Sample Tests")
        self.log("="*50 + "\n")
        
        demos = {
            "Identical Texts": ("Machine learning is AI", "Machine learning is AI"),
            "Paraphrased": ("The quick brown fox jumps", "A fast brown fox leaps"),
            "Different": ("Python programming language", "Cooking with recipes")
        }
        
        for demo_name, (text1, text2) in demos.items():
            self.log(f"\n📌 Demo: {demo_name}")
            self.log("-" * 40)
            
            try:
                result = self.detector.detect_plagiarism(text1, text2, 0.4, 0.6)
                self.log(f"   TF-IDF:   {result['tfidf_similarity']:.4f}")
                self.log(f"   S-BERT:   {result['sbert_similarity']:.4f}")
                self.log(f"   Combined: {result['combined_similarity']:.4f}")
                verdict = "⚠️  PLAGIARISM" if result['plagiarism_detected'] else "✓ ORIGINAL"
                self.log(f"   Result:   {verdict}")
            except Exception as e:
                self.log(f"   ❌ Error: {e}")
        
        self.log("\n" + "="*50 + "\n")
    
    def generate_report(self):
        """Generate report"""
        if not self.results_history:
            messagebox.showwarning("No Results", "Run a detection first!")
            return
        
        report_path = Path('data/results/gui_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results_history, f, indent=2)
        
        messagebox.showinfo("Success", f"✓ Report saved:\n{report_path}")
        self.log(f"\n✓ Report generated: {report_path}\n")
    
    def clear_all(self):
        """Clear all"""
        self.text1.delete('1.0', tk.END)
        self.text2.delete('1.0', tk.END)
        self.results_text.delete('1.0', tk.END)
        self.log("✓ All fields cleared\n")
    
    def log(self, message):
        """Log message to results"""
        self.results_text.insert(tk.END, message + '\n')
        self.results_text.see(tk.END)
        self.root.update()

if __name__ == '__main__':
    root = tk.Tk()
    app = PlagiarismDetectorGUI(root)
    root.mainloop()