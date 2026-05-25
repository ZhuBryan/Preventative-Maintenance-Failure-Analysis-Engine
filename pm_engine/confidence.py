"""Confidence scoring: does the technician's description match the asset?

The structured fields (AssetCategory, EquipmentType, Trade, ...) come from
dropdowns and are treated as ground truth. The free-text description is
human-written and can be wrong. We embed both, take the cosine similarity,
and call anything below the configured threshold a mismatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .embeddings import EmbeddingBackend
from .io_utils import Config, build_structured_profile


def compute_confidence(
    df: pd.DataFrame, config: Config, embedder: EmbeddingBackend
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Annotate `df` with `confidence_score` and `mismatch_flag`.

    Returns the annotated frame plus the two L2-normalised embedding matrices
    (structured, description) so the caller can reuse them for clustering
    without re-embedding.
    """
    structured_text = df.apply(build_structured_profile, axis=1).tolist()
    description_text = df["description"].astype(str).tolist()

    struct_emb, desc_emb = embedder.encode_pairs(structured_text, description_text)

    similarity = np.sum(struct_emb * desc_emb, axis=1)
    similarity = (similarity + 1.0) / 2.0

    annotated = df.copy()
    annotated["confidence_score"] = np.round(similarity, 4)
    annotated["mismatch_flag"] = np.where(
        similarity < config.mismatch_threshold, "MISMATCH - review needed", ""
    )

    return annotated, struct_emb, desc_emb
