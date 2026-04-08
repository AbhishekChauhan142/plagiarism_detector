"""Ensemble algorithm combining multiple plagiarism detection methods."""

import logging
from typing import Dict, List, Optional, Tuple

from .tfidf_algorithm import TFIDFAlgorithm
from .sbert_algorithm import SBERTAlgorithm
from .ngram_algorithm import NGramAlgorithm

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "tfidf": 0.35,
    "sbert": 0.45,
    "ngram": 0.20,
}


class EnsembleAlgorithm:
    """Weighted ensemble of TF-IDF, S-BERT, and N-Gram algorithms.

    Each component algorithm produces an independent similarity score.
    Those scores are combined as a weighted average to obtain the final
    ensemble score.  The individual component scores are also returned in
    the result dict for transparency.
    """

    name: str = "Ensemble"
    description: str = (
        "Hybrid ensemble method combining TF-IDF, S-BERT, and N-Gram "
        "algorithms via a weighted average of their similarity scores."
    )

    def __init__(
        self,
        threshold: float = 0.6,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialise the Ensemble algorithm.

        Args:
            threshold: Similarity score above which a pair is flagged as
                plagiarised.
            weights: Optional dict with keys ``"tfidf"``, ``"sbert"``, and
                ``"ngram"`` mapping to non-negative floats.  Values are
                normalised to sum to 1.  Defaults to
                ``{"tfidf": 0.35, "sbert": 0.45, "ngram": 0.20}``.
        """
        self.threshold = threshold
        raw_weights = weights if weights is not None else dict(_DEFAULT_WEIGHTS)
        total = sum(raw_weights.values())
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive number.")
        self.weights: Dict[str, float] = {k: v / total for k, v in raw_weights.items()}

        self._tfidf = TFIDFAlgorithm()
        self._sbert = SBERTAlgorithm()
        self._ngram = NGramAlgorithm()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute the weighted ensemble similarity between two texts.

        Args:
            text1: First document string.
            text2: Second document string.

        Returns:
            Weighted average similarity score in [0, 1].
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        scores: Dict[str, float] = {}
        try:
            scores["tfidf"] = self._tfidf.compute_similarity(text1, text2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TF-IDF component failed: %s", exc)
            scores["tfidf"] = 0.0

        try:
            scores["sbert"] = self._sbert.compute_similarity(text1, text2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("S-BERT component failed: %s", exc)
            scores["sbert"] = 0.0

        try:
            scores["ngram"] = self._ngram.compute_similarity(text1, text2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("N-Gram component failed: %s", exc)
            scores["ngram"] = 0.0

        combined = sum(self.weights.get(k, 0.0) * v for k, v in scores.items())
        return float(max(0.0, min(1.0, combined)))

    def detect(self, text1: str, text2: str) -> Dict:
        """Detect plagiarism between two texts.

        Returns individual component scores as well as the combined score.

        Args:
            text1: First document string.
            text2: Second document string.

        Returns:
            Dict with keys: similarity, is_plagiarized, algorithm, threshold,
            tfidf_score, sbert_score, ngram_score, weights.
        """
        tfidf_score = 0.0
        sbert_score = 0.0
        ngram_score = 0.0

        try:
            tfidf_score = self._tfidf.compute_similarity(text1, text2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TF-IDF component failed: %s", exc)

        try:
            sbert_score = self._sbert.compute_similarity(text1, text2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("S-BERT component failed: %s", exc)

        try:
            ngram_score = self._ngram.compute_similarity(text1, text2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("N-Gram component failed: %s", exc)

        similarity = (
            self.weights.get("tfidf", 0.0) * tfidf_score
            + self.weights.get("sbert", 0.0) * sbert_score
            + self.weights.get("ngram", 0.0) * ngram_score
        )
        similarity = float(max(0.0, min(1.0, similarity)))

        return {
            "similarity": similarity,
            "is_plagiarized": similarity >= self.threshold,
            "algorithm": self.name,
            "threshold": self.threshold,
            "tfidf_score": tfidf_score,
            "sbert_score": sbert_score,
            "ngram_score": ngram_score,
            "weights": dict(self.weights),
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
