"""Character N-gram Jaccard similarity algorithm for plagiarism detection."""

import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class NGramAlgorithm:
    """Character n-gram Jaccard similarity for plagiarism detection.

    Extracts overlapping character n-grams from each document and computes
    the Jaccard index of the two n-gram sets.  This approach is robust to
    minor character-level edits and is commonly used as a baseline in PAN
    shared tasks.
    """

    name: str = "N-Gram"
    description: str = (
        "Character n-gram Jaccard similarity algorithm. "
        "Extracts overlapping character n-grams and computes the Jaccard "
        "index between the two n-gram sets."
    )

    def __init__(self, n: int = 3, threshold: float = 0.3) -> None:
        """Initialise the N-Gram algorithm.

        Args:
            n: Character n-gram size.
            threshold: Jaccard score above which a pair is flagged as
                plagiarised.
        """
        self.n = n
        self.threshold = threshold

    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Extract overlapping character n-grams from *text*.

        Whitespace is collapsed and the text is lowercased before extraction.

        Args:
            text: Input string.
            n: N-gram size.

        Returns:
            Set of character n-gram strings.
        """
        cleaned = " ".join(text.lower().split())
        return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute the Jaccard similarity of character n-gram sets.

        Args:
            text1: First document string.
            text2: Second document string.

        Returns:
            Jaccard similarity score in [0, 1].
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        ngrams1 = self._get_ngrams(text1, self.n)
        ngrams2 = self._get_ngrams(text2, self.n)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        return len(intersection) / len(union)

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
