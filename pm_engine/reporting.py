"""Summary table + per-row CSV writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_results_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    cols = [
        "work_order_id",
        "asset_category",
        "equipment_type",
        "description",
        "confidence_score",
        "mismatch_flag",
        "cluster_id",
        "cluster_label",
    ]
    cols = [c for c in cols if c in df.columns]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].rename(columns={"work_order_id": "WorkOrderID"}).to_csv(out, index=False)
    return out


def print_summary(df: pd.DataFrame, mismatch_count: int, total: int) -> None:
    clustered = df[df["cluster_id"].notna()].copy()
    grouped = (
        clustered.groupby(["cluster_id", "cluster_label"])
        .agg(
            count=("work_order_id", "size"),
            sample_ids=("work_order_id", lambda s: ", ".join(list(s)[:5])),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    total_clustered = int(grouped["count"].sum()) or 1

    print()
    print("=" * 78)
    print(f"  Analyzed {total} work orders")
    print(f"  Flagged {mismatch_count} as MISMATCH (description doesn't match asset)")
    print(f"  Clustered {total_clustered} trusted orders into {len(grouped)} groups")
    print("=" * 78)
    print()
    print(f"{'#':>3}  {'Cluster label':<42} {'Count':>6} {'% of trusted':>13}")
    print("-" * 78)
    for i, row in enumerate(grouped.itertuples(index=False), start=1):
        pct = 100.0 * row.count / total_clustered
        label = (row.cluster_label or "")[:42]
        print(f"{i:>3}  {label:<42} {row.count:>6} {pct:>12.1f}%")
        print(f"      sample WOs: {row.sample_ids}")
    print()
