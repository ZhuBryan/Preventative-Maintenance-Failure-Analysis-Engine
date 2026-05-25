"""CLI entry point for the Preventative Maintenance Failure Analysis Engine.

Usage:
    python analyze.py --input data/work_orders.xlsx --output data/results.csv

The script does the three things laid out in the design doc:
    1. Score every row's confidence (free-text vs. structured asset fields)
       and flag mismatches.
    2. Cluster trusted rows by likely failure cause using a weighted
       combination of structured-field and description embeddings.
    3. Label each cluster (Claude if ANTHROPIC_API_KEY is set, otherwise
       TF-IDF keywords) and print a ranked summary.
"""

from __future__ import annotations

import argparse
import sys

from pm_engine.clustering import cluster_orders
from pm_engine.confidence import compute_confidence
from pm_engine.embeddings import EmbeddingBackend
from pm_engine.io_utils import Config, load_work_orders
from pm_engine.labeling import label_clusters
from pm_engine.reporting import print_summary, write_results_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to the work-order Excel/CSV.")
    p.add_argument(
        "--output",
        default="data/results.csv",
        help="Where to write the per-row results CSV.",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config (column mappings, thresholds, weights).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    config = Config.load(args.config)

    print(f"Loading {args.input} ...")
    df = load_work_orders(args.input, config)
    if df.empty:
        print("No work orders with non-empty descriptions found. Nothing to do.")
        return 1
    print(f"  loaded {len(df)} work orders with descriptions")

    print(f"Loading embedding backend: {config.embedding_model}")
    embedder = EmbeddingBackend(config.embedding_model)
    print(f"  embedding backend: {embedder.backend_name}")

    print("Scoring confidence (description vs. structured asset fields) ...")
    annotated, struct_emb, desc_emb = compute_confidence(df, config, embedder=embedder)
    mismatch_mask = annotated["mismatch_flag"] != ""
    mismatch_count = int(mismatch_mask.sum())
    print(
        f"  {mismatch_count} of {len(annotated)} rows flagged as mismatches "
        f"(threshold={config.mismatch_threshold:.2f})"
    )

    trusted_idx = annotated.index[~mismatch_mask]
    trusted_df = annotated.loc[trusted_idx].reset_index(drop=True)
    trusted_struct = struct_emb[trusted_idx.values]
    trusted_desc = desc_emb[trusted_idx.values]

    if len(trusted_df) < max(2, int(config.clustering.get("min_cluster_size", 8))):
        print(
            "  Too few trusted rows to cluster meaningfully. "
            "Lower the mismatch_threshold in config.yaml or fix the input data."
        )
        annotated["cluster_id"] = -1
        annotated["cluster_label"] = "Unclustered / outliers"
        write_results_csv(annotated, args.output)
        return 0

    print(
        f"Clustering {len(trusted_df)} trusted orders "
        f"(structured weight={config.weights['structured']}, "
        f"description weight={config.weights['description']}) ..."
    )
    clustered_df, _ = cluster_orders(trusted_df, trusted_struct, trusted_desc, config)
    n_real = int((clustered_df["cluster_id"] >= 0).sum())
    n_noise = int((clustered_df["cluster_id"] == -1).sum())
    print(f"  {n_real} rows assigned to clusters, {n_noise} rows marked as noise")

    print("Labeling clusters ...")
    labels = label_clusters(clustered_df, config)
    clustered_df["cluster_label"] = clustered_df["cluster_id"].map(labels)

    full = annotated.merge(
        clustered_df[["work_order_id", "cluster_id", "cluster_label"]],
        on="work_order_id",
        how="left",
    )
    full["cluster_id"] = full["cluster_id"].astype("Int64")
    full.loc[mismatch_mask, "cluster_label"] = ""

    out_path = write_results_csv(full, args.output)
    print(f"Wrote per-row results to {out_path}")

    print_summary(clustered_df, mismatch_count=mismatch_count, total=len(annotated))

    if mismatch_count:
        print(
            f"Tip: open {out_path}, filter by mismatch_flag != '', and spot-check those "
            f"{mismatch_count} rows. That's the human-error queue."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
