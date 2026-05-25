"""Embedding backend with a lightweight local fallback.

Sentence-transformers is the preferred backend because it understands semantic
similarity ("not cooling" and "low chilled water flow" can still land near one
another). On locked-down laptops or new Python versions where torch wheels are
hard to install, the TF-IDF fallback keeps the proof-of-concept runnable.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingBackend:
    """Encode text using sentence-transformers, with TF-IDF as a fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.backend_name = "tfidf"
        self._model = None

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.backend_name = "sentence-transformers"
        except Exception as exc:
            print(
                "  Could not load sentence-transformers "
                f"({exc}); using TF-IDF fallback."
            )

    def encode_pairs(
        self, structured_text: list[str], description_text: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode two comparable text lists into the same vector space."""
        if self._model is not None:
            struct_emb = np.asarray(
                self._model.encode(structured_text, show_progress_bar=False)
            )
            desc_emb = np.asarray(
                self._model.encode(description_text, show_progress_bar=False)
            )
            return l2_normalize(struct_emb), l2_normalize(desc_emb)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        all_text = structured_text + description_text
        matrix = vectorizer.fit_transform(all_text).toarray()
        split = len(structured_text)
        return l2_normalize(matrix[:split]), l2_normalize(matrix[split:])
