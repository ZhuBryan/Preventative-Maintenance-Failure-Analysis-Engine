# Preventative Maintenance Failure Analysis Engine

A Python tool that takes a building's preventative-maintenance work order Excel
sheet (~500 rows) and produces a ranked list of likely failure causes, while
also flagging work orders whose technician-written description does not match
the structured asset fields.

The goal: turn a full day of manual review into a few minutes of acting on
patterns.

## How it works

The pipeline runs in three stages:

1. **Confidence scoring (mismatch detection).** Each work order's free-text
   `Description` is embedded and compared (cosine similarity) to a "structured
   profile" built from reliable dropdown fields (`AssetCategory`,
   `EquipmentType`, `Trade`, ...). Orders below the configured threshold are
   flagged `MISMATCH - review needed` so a human can spot-check them quickly
   instead of reading every record.

2. **Failure clustering.** For trusted orders, the structured-field embedding
   and the description embedding are combined with configurable weights
   (default: structured 0.7, description 0.3) and re-normalised. Records are
   then clustered with HDBSCAN (no need to pre-pick `k`) or KMeans.

3. **Cluster labeling.** Each cluster gets a short plain-English label. If
   `ANTHROPIC_API_KEY` is set the labels come from Claude; otherwise a TF-IDF
   keyword fallback is used so the tool still works fully offline.

The final output is a per-row CSV (`WorkOrderID, confidence_score,
mismatch_flag, cluster_id, cluster_label`) plus a printed ranked summary table.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional, higher-quality semantic embeddings + LLM labels:
# pip install -r requirements-optional.txt

# 1. Generate a synthetic Excel so you can test before you have real data
python scripts/generate_sample_data.py --out data/sample_work_orders.xlsx

# 2. Run the analyzer on it
python analyze.py --input data/sample_work_orders.xlsx --output data/results.csv
```

If `sentence-transformers` is installed, the first run downloads a small
embedding model (~90 MB) and caches it locally. If it is not installed, the tool
automatically falls back to TF-IDF vectors so the proof-of-concept still runs.

## Using your real data

The City's Excel will not have the exact column names assumed in the example.
Edit `config.yaml` and map your actual column headers to the logical fields the
engine uses:

```yaml
columns:
  work_order_id: "WO Number"
  asset_category: "Asset Class"
  equipment_type: "Equipment"
  location: "Location"
  trade: "Craft"
  priority: "Priority"
  description: "Long Description"
  resolution_notes: "Completion Notes"
```

Any field can be left blank if your sheet doesn't have it; the engine just
won't use it. As long as `description` is present, the tool will run.

## Tuning

In `config.yaml`:

- `mismatch_threshold` (default `0.40`) - similarity below this triggers a
  mismatch flag. After running once on real data, look at the distribution of
  `confidence_score` and pick a threshold that catches the obviously wrong
  rows without false-flagging short-but-valid descriptions.
- `weights.structured` / `weights.description` (default `0.7 / 0.3`) - how much
  the dropdown fields vs. the free text influence clustering. Increase
  structured weight if your technicians' free text is noisy.
- `clustering.algorithm` - `hdbscan` (recommended if installed) or `kmeans`.
- `clustering.min_cluster_size` - for HDBSCAN, the smallest group worth
  reporting as its own failure pattern (default 8).

## Project layout

```
analyze.py                    CLI entry point
config.yaml                   column mapping, weights, thresholds
pm_engine/
  io_utils.py                 Excel loading + column mapping
  confidence.py               Mismatch detection (structured vs. description)
  clustering.py               Weighted-embedding clustering
  labeling.py                 LLM / keyword cluster labels
  reporting.py                Summary table + CSV writer
scripts/
  generate_sample_data.py     Synthetic Excel with intentional mismatches
data/                         Your inputs and outputs (gitignored)
```

## Validation plan (the intern's first concrete deliverable)

Before showing your team anything: take 50-100 past work orders, label them by
hand into categories, run them through this engine, and compare. The agreement
rate is the proof-of-concept your team will want to see.
