"""Excel loading and logical-field mapping.

Real-world maintenance exports never use the column names you wish they did.
This module is the only place that knows the user's actual headers; everything
downstream works on the logical fields defined in `LOGICAL_FIELDS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGICAL_FIELDS = [
    "work_order_id",
    "asset_category",
    "equipment_type",
    "location",
    "trade",
    "priority",
    "description",
    "resolution_notes",
]

STRUCTURED_FIELDS = ["asset_category", "equipment_type", "trade", "location", "priority"]
REQUIRED_FIELDS = ["work_order_id", "description"]


@dataclass
class Config:
    columns: dict[str, str | None]
    embedding_model: str
    mismatch_threshold: float
    weights: dict[str, float]
    clustering: dict[str, Any]
    labeling: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        weights = raw.get("weights", {"structured": 0.7, "description": 0.3})
        total = weights["structured"] + weights["description"]
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"weights.structured + weights.description must equal 1.0, got {total}"
            )

        columns = {f: raw.get("columns", {}).get(f) for f in LOGICAL_FIELDS}
        for req in REQUIRED_FIELDS:
            if not columns.get(req):
                raise ValueError(
                    f"config.yaml is missing a column mapping for required field '{req}'"
                )

        return cls(
            columns=columns,
            embedding_model=raw.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            mismatch_threshold=float(raw.get("mismatch_threshold", 0.40)),
            weights=weights,
            clustering=raw.get(
                "clustering",
                {"algorithm": "hdbscan", "min_cluster_size": 8, "n_clusters": 10},
            ),
            labeling=raw.get(
                "labeling",
                {"use_llm": True, "model": "claude-haiku-4-5", "samples_per_cluster": 12},
            ),
        )


def load_work_orders(input_path: str | Path, config: Config) -> pd.DataFrame:
    """Load the Excel sheet and return a frame keyed by logical field names.

    Missing optional columns are filled with empty strings so downstream code
    can treat every logical field as present.
    """
    path = Path(input_path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        raw = pd.read_excel(path, engine="openpyxl" if path.suffix.lower() != ".xls" else None)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input file extension: {path.suffix}")

    out = pd.DataFrame(index=raw.index)
    for logical, header in config.columns.items():
        if header and header in raw.columns:
            out[logical] = raw[header].astype("string").fillna("")
        else:
            out[logical] = ""

    out["work_order_id"] = out["work_order_id"].replace("", pd.NA).fillna(
        pd.Series([f"ROW_{i}" for i in range(len(out))], index=out.index, dtype="string")
    )

    out = out[out["description"].str.strip() != ""].reset_index(drop=True)

    return out


def build_structured_profile(row: pd.Series) -> str:
    """Render the reliable dropdown fields as one short comparable string.

    Example: "HVAC | Chiller | Mechanical | Floor 3 | High"
    Empty fields are skipped so we don't pollute the embedding with noise.
    """
    parts = [str(row[f]).strip() for f in STRUCTURED_FIELDS if str(row[f]).strip()]
    return " | ".join(parts) if parts else "unspecified asset"
