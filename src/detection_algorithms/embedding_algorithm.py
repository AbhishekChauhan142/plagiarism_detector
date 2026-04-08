"""Word embedding cosine similarity algorithm for plagiarism detection."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class EmbeddingAlgorithm:
    """Document embedding cosine similarity for plagiarism detection.

    Fits a :class:`~sklearn.feature_extraction.text.TfidfVectorizer` on the
    two documents, extracts the sparse TF-IDF matrix, and computes the cosine
    similarity between the resulting dense document vectors.  When a word
    appears in only one document it still contributes to the directional
    difference, making this sensitive to vocabulary overlap.
    """

    name: str = "Word-Embedding"
    description: str = (
        "TF-IDF weighted word vector cosine similarity algorithm. "
        "Converts each document to an average word vector weighted by TF-IDF "
        "scores and computes the cosine similarity between those vectors."
    )

    def __init__(self, threshold: float = 0.6) -> None:
        """Initialise the Embedding algorithm.

        Args:
            threshold: Similarity score above which a pair is flagged as
                plagiarised.
        """
        self.threshold = threshold

    def _text_to_vector(self, text: str, vectorizer: Optional[TfidfVectorizer] = None) -> np.ndarray:
        """Convert a document to a dense TF-IDF vector.

        If a pre-fitted *vectorizer* is supplied it is used directly;
        otherwise a new one is fitted on *text* alone.

        Args:
            text: Input document string.
            vectorizer: Optional pre-fitted TfidfVectorizer.

        Returns:
            1-D dense numpy array.
        """
        if vectorizer is not None:
            return vectorizer.transform([text]).toarray()[0]

        vect = TfidfVectorizer(stop_words="english")
        return vect.fit_transform([text]).toarray()[0]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute the TF-IDF weighted embedding cosine similarity.

        Both documents are transformed with a shared vocabulary so that the
        resulting vectors live in the same feature space.

        Args:
            text1: First document string.
            text2: Second document string.

        Returns:
            Cosine similarity score in [0, 1].
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2]).toarray()
        except ValueError as exc:
            logger.warning("Embedding similarity computation failed: %s", exc)
            return 0.0

        vec1: np.ndarray = tfidf_matrix[0]
        vec2: np.ndarray = tfidf_matrix[1]

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        return max(0.0, min(1.0, similarity))

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
