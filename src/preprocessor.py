# src/preprocessor.py
import re

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s\.\,\!\?]', '', text)
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove common stopwords"""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'
        }
        words = text.split()
        return ' '.join([w for w in words if w not in stopwords])
    
    def tokenize(self, text: str) -> list:
        """Tokenize text into words"""
        text = self.preprocess_text(text)
        return text.split()