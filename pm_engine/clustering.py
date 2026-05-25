"""Cluster trusted work orders by likely failure cause.

We build the clustering signal as a weighted average of the structured-field
embedding and the description embedding. This is what implements the
"down-weight noisy human text" requirement: the structured fields get the
larger weight (0.7 by default) and so dominate the geometry of the embedding
space the clusterer sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .embeddings import l2_normalize
from .io_utils import Config


def combine_embeddings(
    struct_emb: np.ndarray, desc_emb: np.ndarray, config: Config
) -> np.ndarray:
    """Weighted average of the two embedding spaces, then re-normalised.

    Re-normalising matters: cosine-based clusterers expect unit-length vectors,
    and a weighted average of two unit vectors is not unit-length in general.
    """
    w_s = config.weights["structured"]
    w_d = config.weights["description"]
    combined = w_s * struct_emb + w_d * desc_emb
    return l2_normalize(combined)


def cluster_orders(
    trusted_df: pd.DataFrame,
    struct_emb: np.ndarray,
    desc_emb: np.ndarray,
    config: Config,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Assign a cluster_id to each trusted work order.

    HDBSCAN may label some rows as noise (-1); those are kept in the output but
    surfaced as their own "Unclustered / outliers" bucket in the summary.
    """
    combined = combine_embeddings(struct_emb, desc_emb, config)

    algo = config.clustering.get("algorithm", "hdbscan").lower()
    if algo == "kmeans":
        n_clusters = int(config.clustering.get("n_clusters", 10))
        n_clusters = max(2, min(n_clusters, len(trusted_df)))
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(
            combined
        )
    elif algo == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            print("  hdbscan is not installed; falling back to KMeans.")
            n_clusters = int(config.clustering.get("n_clusters", 10))
            n_clusters = max(2, min(n_clusters, len(trusted_df)))
            labels = KMeans(
                n_clusters=n_clusters, n_init=10, random_state=42
            ).fit_predict(combined)
            out = trusted_df.copy()
            out["cluster_id"] = labels
            return out, combined

        min_size = int(config.clustering.get("min_cluster_size", 8))
        min_size = max(2, min(min_size, len(trusted_df)))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(combined)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")

    out = trusted_df.copy()
    out["cluster_id"] = labels
    return out, combined
