# tests/__init__.py
"""Tests package for plagiarism detector"""
import sys
from pathlib import Path

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))