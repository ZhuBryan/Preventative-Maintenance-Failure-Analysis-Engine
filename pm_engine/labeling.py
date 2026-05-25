"""Turn cluster IDs into short plain-English labels.

Two backends:

* LLM (Claude via the Anthropic API) - best quality, used when
  `ANTHROPIC_API_KEY` is set and `labeling.use_llm` is true.
* TF-IDF keyword fallback - always available, no network calls. Picks the
  top distinguishing terms for each cluster and joins them.

Both return a `dict[cluster_id -> label]`. Cluster id `-1` (HDBSCAN noise) is
always labeled "Unclustered / outliers".
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .io_utils import Config

NOISE_LABEL = "Unclustered / outliers"


def _keyword_labels(df: pd.DataFrame) -> dict[int, str]:
    """TF-IDF top-terms fallback."""
    labels: dict[int, str] = {}
    cluster_ids = sorted(int(c) for c in df["cluster_id"].unique())

    docs = []
    for cid in cluster_ids:
        if cid == -1:
            docs.append("")
            continue
        rows = df[df["cluster_id"] == cid]
        text = " ".join(
            (rows["asset_category"].fillna("") + " " + rows["equipment_type"].fillna("")
             + " " + rows["description"].fillna("")).tolist()
        )
        docs.append(text)

    if not any(docs):
        return {cid: NOISE_LABEL if cid == -1 else f"Cluster {cid}" for cid in cluster_ids}

    vec = TfidfVectorizer(
        max_features=2000, stop_words="english", ngram_range=(1, 2), min_df=1
    )
    matrix = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    for i, cid in enumerate(cluster_ids):
        if cid == -1:
            labels[cid] = NOISE_LABEL
            continue
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:3]
        terms = [vocab[j] for j in top_idx if row[j] > 0]
        if not terms:
            rows = df[df["cluster_id"] == cid]
            common = Counter(rows["asset_category"].fillna("").tolist()).most_common(1)
            labels[cid] = (common[0][0] or f"Cluster {cid}").title()
        else:
            labels[cid] = " / ".join(t.title() for t in terms)

    return labels


def _llm_labels(df: pd.DataFrame, config: Config) -> dict[int, str] | None:
    """Ask Claude to name each cluster. Returns None if the API isn't available."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    client = Anthropic(api_key=api_key)
    samples_per_cluster = int(config.labeling.get("samples_per_cluster", 12))
    model_name = config.labeling.get("model", "claude-haiku-4-5")

    labels: dict[int, str] = {}
    cluster_ids = sorted(int(c) for c in df["cluster_id"].unique())

    for cid in cluster_ids:
        if cid == -1:
            labels[cid] = NOISE_LABEL
            continue

        rows = df[df["cluster_id"] == cid].head(samples_per_cluster)
        examples = []
        for _, r in rows.iterrows():
            structured = " | ".join(
                str(r[f]).strip()
                for f in ["asset_category", "equipment_type", "trade"]
                if str(r[f]).strip()
            )
            examples.append(f"- [{structured}] {r['description']}")

        prompt = (
            "These maintenance work orders all belong to the same failure-cause "
            "cluster. Give it a short label (3-6 words) describing the underlying "
            "failure cause, in title case, no quotes, no trailing period.\n\n"
            + "\n".join(examples)
            + "\n\nLabel:"
        )

        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=40,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip().splitlines()[0].strip()
            text = text.strip('"').strip("'").rstrip(".")
            labels[cid] = text or f"Cluster {cid}"
        except Exception as exc:
            print(f"  LLM labeling failed for cluster {cid} ({exc}); using keyword fallback")
            return None

    return labels


def label_clusters(df: pd.DataFrame, config: Config) -> dict[int, str]:
    """Public entrypoint: returns {cluster_id: label}."""
    if config.labeling.get("use_llm", True):
        llm = _llm_labels(df, config)
        if llm is not None:
            return llm
    return _keyword_labels(df)
