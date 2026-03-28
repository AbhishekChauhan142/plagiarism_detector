"""TF-IDF baseline algorithm for plagiarism detection."""

import logging
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TFIDFAlgorithm:
    """TF-IDF cosine similarity baseline for plagiarism detection.

    Vectorises each pair of documents with a freshly fitted
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer` and returns
    the cosine similarity of the resulting document vectors.
    """

    name: str = "TF-IDF"
    description: str = (
        "Term Frequency–Inverse Document Frequency baseline algorithm. "
        "Computes cosine similarity between TF-IDF weighted document vectors."
    )

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        threshold: float = 0.7,
    ) -> None:
        """Initialise the TF-IDF algorithm.

        Args:
            max_features: Maximum vocabulary size passed to TfidfVectorizer.
            ngram_range: The lower and upper boundary of the n-gram range.
            threshold: Similarity score above which a pair is flagged as
                plagiarised.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.threshold = threshold

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute the TF-IDF cosine similarity between two texts.

        Args:
            text1: First document string.
            text2: Second document string.

        Returns:
            Cosine similarity score in [0, 1].
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except ValueError as exc:
            logger.warning("TF-IDF similarity computation failed: %s", exc)
            return 0.0

    def detect(self, text1: str, text2: str) -> Dict:
        """Detect plagiarism between two texts.

        Args:
            text1: First document string.
            text2: Second document string.

        Returns:
            Dict with keys: similarity, is_plagiarized, algorithm, threshold.
        """
        similarity = self.compute_similarity(text1, text2)
        return {
            "similarity": similarity,
            "is_plagiarized": similarity >= self.threshold,
            "algorithm": self.name,
            "threshold": self.threshold,
        }

    def batch_detect(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Run detection on a list of text pairs.

        Args:
            pairs: List of ``(text1, text2)`` tuples.

        Returns:
            List of result dicts in the same order as *pairs*.
        """
        results = []
        for idx, (text1, text2) in enumerate(pairs):
            try:
                results.append(self.detect(text1, text2))
            except Exception as exc:  # noqa: BLE001
                logger.error("Batch detect error at index %d: %s", idx, exc)
                results.append(
                    {
                        "similarity": 0.0,
                        "is_plagiarized": False,
                        "algorithm": self.name,
                        "threshold": self.threshold,
                        "error": str(exc),
                    }
                )
        return results
