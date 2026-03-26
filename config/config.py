# config/config.py
class Config:
    # Model settings
    SBERT_MODEL = 'all-MiniLM-L6-v2'
    TFIDF_MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
    
    # Similarity thresholds
    PLAGIARISM_THRESHOLD = 0.7
    WARNING_THRESHOLD = 0.5
    
    # Weights for combined scoring
    TFIDF_WEIGHT = 0.4
    SBERT_WEIGHT = 0.6
    
    # Preprocessing settings
    REMOVE_STOPWORDS = True
    LOWERCASE = True