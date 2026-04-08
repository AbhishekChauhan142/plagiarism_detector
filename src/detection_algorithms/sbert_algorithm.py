"""Sentence-BERT semantic similarity algorithm for plagiarism detection."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SBERTAlgorithm:
    """Semantic plagiarism detection using Sentence-BERT embeddings.

    Encodes each document with a pre-trained
    :class:`~sentence_transformers.SentenceTransformer` model and returns the
    cosine similarity between the resulting sentence vectors.

    The model is loaded lazily on the first call to :meth:`compute_similarity`
    to avoid heavy startup costs.
    """

    name: str = "S-BERT"
    description: str = (
        "Sentence-BERT semantic similarity algorithm. "
        "Encodes documents with a pre-trained transformer and computes "
        "cosine similarity between sentence embeddings."
    )

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.7,
    ) -> None:
        """Initialise the S-BERT algorithm.

        Args:
            model_name: Name of the sentence-transformers model to use.
            threshold: Similarity score above which a pair is flagged as
                plagiarised.
        """
        self.model_name = model_name
        self.threshold = threshold
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        """Lazily load the SentenceTransformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            logger.info("Loading S-BERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("S-BERT model loaded successfully.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SBERTAlgorithm. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute the semantic cosine similarity between two texts.

        Args:
            text1: First document string.
            text2: Second document string.

        Returns:
            Cosine similarity score in [0, 1].
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        self._load_model()
        embeddings = self._model.encode([text1, text2], convert_to_numpy=True)
        vec1: np.ndarray = embeddings[0]
        vec2: np.ndarray = embeddings[1]

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        # Clamp to [0, 1] – cosine can technically be negative
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

        Encodes all unique texts in a single forward pass for efficiency.

        Args:
            pairs: List of ``(text1, text2)`` tuples.

        Returns:
            List of result dicts in the same order as *pairs*.
        """
        if not pairs:
            return []

        self._load_model()

        # Encode all texts at once
        all_texts = [text for pair in pairs for text in pair]
        try:
            all_embeddings = self._model.encode(all_texts, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("Batch encoding failed, falling back to single: %s", exc)
            return [self.detect(t1, t2) for t1, t2 in pairs]

        results = []
        for i, (text1, text2) in enumerate(pairs):
            vec1 = all_embeddings[i * 2]
            vec2 = all_embeddings[i * 2 + 1]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0 or not text1.strip() or not text2.strip():
                similarity = 0.0
            else:
                similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
                similarity = max(0.0, min(1.0, similarity))
            results.append(
                {
                    "similarity": similarity,
                    "is_plagiarized": similarity >= self.threshold,
                    "algorithm": self.name,
                    "threshold": self.threshold,
                }
            )
        return results
