"""Construct event-level datasets and sequence views across warning horizons.

Strictly reads data/raw/esa/train_data.csv using only DIRECT features.
Outputs:
- data/processed/events/events_H2.csv
- data/processed/events/events_H3.csv
- data/processed/events/events_H5.csv
- data/processed/events/sequences_H2.csv
- data/processed/events/sequences_H3.csv
- data/processed/events/sequences_H5.csv
- data/manifests/event_dataset_manifest.json
- reports/event_dataset_summary.md
"""

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from orvexa.event_builder import (
    DIRECT_FEATURE_COLUMNS,
    EXCLUDED_IDENTIFIERS,
    EXCLUDED_TARGETS,
    NEEDS_HUMAN_REVIEW_FEATURES,
    NOT_AVAILABLE_FEATURES,
    build_event_prefixes_from_raw,
    compute_file_sha256,
)


def compute_percentiles(values, percentiles=(0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)):
    """Compute exact percentiles."""
    if not values:
        return {f"p{int(p*100)}": None for p in percentiles}
    sorted_v = sorted(values)
    n = len(sorted_v)
    res = {}
    for p in percentiles:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            val = sorted_v[int(k)]
        else:
            d0 = sorted_v[int(f)] * (c - k)
            d1 = sorted_v[int(c)] * (k - f)
            val = d0 + d1
        p_name = f"p{int(p*100)}" if p * 100 == int(p * 100) else f"p{p*100:.1f}"
        res[p_name] = float(val)
    return res


def median(values):
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_v[mid])
    else:
        return float(sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def main():
    raw_path = os.path.abspath("data/raw/esa/train_data.csv")
    expected_sha256 = "ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6"

    print("==================================================")
    print("ORVEXA EVENT-LEVEL DATASET BUILDER")
    print("==================================================")

    # 1. Checksum verification BEFORE
    if not os.path.exists(raw_path):
        print(f"Error: Raw file not found at {raw_path}")
        sys.exit(1)

    print("Verifying raw dataset SHA-256 before processing...")
    sha256_before = compute_file_sha256(raw_path)
    if sha256_before != expected_sha256:
        print(f"ERROR: Raw CSV checksum mismatch! Expected: {expected_sha256}, got: {sha256_before}")
        sys.exit(1)
    print(f"Checksum verified: {sha256_before}")

    # 2. Read raw rows and group by event_id in exact file order
    print("Reading and grouping raw CDMs by event_id...")
    raw_rows_by_event: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    raw_row_count = 0

    with open(raw_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_row_count += 1
            ev_id = row["event_id"]
            raw_rows_by_event[ev_id].append(row)

    total_unique_events = len(raw_rows_by_event)
    print(f"Loaded {raw_row_count:,} raw rows across {total_unique_events:,} unique events.")

    # 3. Construct horizon prefixes for H in [2, 3, 5]
    horizons = [2.0, 3.0, 5.0]
    expected_counts = {2.0: 11942, 3.0: 11273, 5.0: 9484}
    
    os.makedirs("data/processed/events", exist_ok=True)
    os.makedirs("data/manifests", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    horizon_summaries = {}

    for h in horizons:
        h_int = int(h)
        print(f"\nConstructing dataset for Horizon H = {h_int} days...")
        records, stats = build_event_prefixes_from_raw(
            raw_rows_by_event=raw_rows_by_event,
            horizon_days=h,
            feature_columns=DIRECT_FEATURE_COLUMNS,
        )

        n_events = len(records)
        print(f"  H = {h_int}d: Generated {n_events:,} eligible events.")
        
        # Verify counts match audit exactly
        if n_events != expected_counts[h]:
            print(f"ERROR: Event count for H={h} ({n_events}) does not match audit expected ({expected_counts[h]})!")
            sys.exit(1)

        # 4. Write event metadata CSV (events_H<H>.csv) and sequence CSV (sequences_H<H>.csv)
        events_csv_path = f"data/processed/events/events_H{h_int}.csv"
        seq_csv_path = f"data/processed/events/sequences_H{h_int}.csv"

        # Metadata table headers: event identifiers, sequence summary, final target, and anchor snapshot features
        event_fieldnames = [
            "event_id",
            "horizon_days",
            "sequence_length",
            "earliest_time_to_tca",
            "anchor_time_to_tca",
            "final_risk",
        ] + DIRECT_FEATURE_COLUMNS

        seq_fieldnames = [
            "event_id",
            "step_index",
            "horizon_days",
        ] + DIRECT_FEATURE_COLUMNS

        # Collect sequence length stats and missing value stats for this horizon
        seq_lens = []
        final_risks = []
        feature_missing_counts = defaultdict(int)
        total_horizon_cdms = 0

        with open(events_csv_path, "w", encoding="utf-8", newline="") as f_ev, \
             open(seq_csv_path, "w", encoding="utf-8", newline="") as f_seq:

            ev_writer = csv.DictWriter(f_ev, fieldnames=event_fieldnames)
            seq_writer = csv.DictWriter(f_seq, fieldnames=seq_fieldnames)

            ev_writer.writeheader()
            seq_writer.writeheader()

            for rec in records:
                seq_lens.append(rec.sequence_length)
                final_risks.append(rec.final_risk)
                total_horizon_cdms += rec.sequence_length

                # Anchor CDM is the last CDM in the prefix (qualifying row with smallest time_to_tca)
                anchor_cdm = rec.cdms[-1]

                ev_row = {
                    "event_id": rec.event_id,
                    "horizon_days": rec.horizon_days,
                    "sequence_length": rec.sequence_length,
                    "earliest_time_to_tca": rec.earliest_time_to_tca,
                    "anchor_time_to_tca": rec.anchor_time_to_tca,
                    "final_risk": rec.final_risk,
                }
                for col in DIRECT_FEATURE_COLUMNS:
                    ev_row[col] = anchor_cdm[col]
                ev_writer.writerow(ev_row)

                # Write sequence rows
                for step_idx, cdm in enumerate(rec.cdms):
                    seq_row = {
                        "event_id": rec.event_id,
                        "step_index": step_idx,
                        "horizon_days": rec.horizon_days,
                    }
                    for col in DIRECT_FEATURE_COLUMNS:
                        val = cdm[col]
                        seq_row[col] = val
                        if not val or val.lower() in ("nan", "null", "none", ""):
                            feature_missing_counts[col] += 1
                    seq_writer.writerow(seq_row)

        print(f"  Wrote {events_csv_path} ({n_events:,} events)")
        print(f"  Wrote {seq_csv_path} ({total_horizon_cdms:,} sequence CDMs)")

        # Compile statistics for summary
        seq_percentiles = compute_percentiles(seq_lens)
        risk_percentiles = compute_percentiles(final_risks)

        horizon_summaries[f"H{h_int}"] = {
            "horizon_days": h_int,
            "eligible_events": n_events,
            "total_cdms": total_horizon_cdms,
            "sequence_length": {
                "min": min(seq_lens),
                "max": max(seq_lens),
                "mean": round(sum(seq_lens) / len(seq_lens), 4),
                "median": median(seq_lens),
                "percentiles": seq_percentiles,
            },
            "target": {
                "name": "final_risk",
                "valid_count": len(final_risks),
                "missing_count": 0,
                "min": min(final_risks),
                "max": max(final_risks),
                "mean": round(sum(final_risks) / len(final_risks), 4),
                "median": median(final_risks),
                "percentiles": risk_percentiles,
            },
            "feature_missing_counts": {
                col: {
                    "missing_count": feature_missing_counts[col],
                    "missing_percentage": round(feature_missing_counts[col] / total_horizon_cdms * 100, 4)
                }
                for col in DIRECT_FEATURE_COLUMNS
            }
        }

    # 5. Checksum verification AFTER
    print("\nVerifying raw dataset SHA-256 after processing...")
    sha256_after = compute_file_sha256(raw_path)
    if sha256_after != expected_sha256:
        print(f"ERROR: Raw file was modified! Expected {expected_sha256}, got {sha256_after}")
        sys.exit(1)
    print(f"Checksum verified unchanged: {sha256_after}")

    # 6. Write Manifest
    manifest_data = {
        "manifest_version": "1.0",
        "dataset_source": "data/raw/esa/train_data.csv",
        "source_sha256": sha256_after,
        "raw_rows_count": raw_row_count,
        "raw_events_count": total_unique_events,
        "horizons": [2, 3, 5],
        "target": {
            "source_column": "risk",
            "event_target_name": "final_risk",
            "description": "Risk value from final CDM (minimum time_to_tca) of each event"
        },
        "grouping": {
            "column": "event_id",
            "ordering": "strictly decreasing time_to_tca"
        },
        "direct_features_used": DIRECT_FEATURE_COLUMNS,
        "direct_features_count": len(DIRECT_FEATURE_COLUMNS),
        "needs_human_review_features_excluded": NEEDS_HUMAN_REVIEW_FEATURES,
        "needs_human_review_features_count": len(NEEDS_HUMAN_REVIEW_FEATURES),
        "not_available_features_excluded": NOT_AVAILABLE_FEATURES,
        "not_available_features_count": len(NOT_AVAILABLE_FEATURES),
        "identifiers_excluded": EXCLUDED_IDENTIFIERS,
        "target_excluded_from_inputs": EXCLUDED_TARGETS,
        "output_files": {
            "events_H2": "data/processed/events/events_H2.csv",
            "sequences_H2": "data/processed/events/sequences_H2.csv",
            "events_H3": "data/processed/events/events_H3.csv",
            "sequences_H3": "data/processed/events/sequences_H3.csv",
            "events_H5": "data/processed/events/events_H5.csv",
            "sequences_H5": "data/processed/events/sequences_H5.csv",
        },
        "horizon_summary": {
            "H2": {"eligible_events": horizon_summaries["H2"]["eligible_events"], "total_cdms": horizon_summaries["H2"]["total_cdms"]},
            "H3": {"eligible_events": horizon_summaries["H3"]["eligible_events"], "total_cdms": horizon_summaries["H3"]["total_cdms"]},
            "H5": {"eligible_events": horizon_summaries["H5"]["eligible_events"], "total_cdms": horizon_summaries["H5"]["total_cdms"]},
        }
    }

    manifest_path = "data/manifests/event_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nWrote manifest to {manifest_path}")

    # 7. Write Event Dataset Summary Report
    summary_path = "reports/event_dataset_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# ORVEXA Event-Level Dataset Construction Summary\n\n")
        f.write("## 1. Dataset Overview & Integrity\n\n")
        f.write(f"- **Source Dataset**: `data/raw/esa/train_data.csv`\n")
        f.write(f"- **Source SHA-256**: `{sha256_after}` (Verified Untouched)\n")
        f.write(f"- **Total Raw Events**: `{total_unique_events:,}`\n")
        f.write(f"- **Total Raw CDMs**: `{raw_row_count:,}`\n")
        f.write(f"- **Direct Features Used**: `{len(DIRECT_FEATURE_COLUMNS)}`\n")
        f.write(f"- **Excluded `NEEDS_HUMAN_REVIEW` Features**: `{len(NEEDS_HUMAN_REVIEW_FEATURES)}`\n")
        f.write(f"- **Excluded `NOT_AVAILABLE` Features**: `{len(NOT_AVAILABLE_FEATURES)}`\n")
        f.write(f"- **Excluded Identifiers**: `event_id` (grouping key), `mission_id` (identity leakage control)\n")
        f.write(f"- **Target**: `risk` -> `final_risk` (excluded from model input features)\n\n")
        f.write("---\n\n")

        f.write("## 2. Horizon Eligibility & Sequence Statistics\n\n")
        f.write("| Horizon | Eligible Events | % Total Events | Total CDMs | Min Len | Max Len | Mean Len | Median Len | p25 | p75 | p90 | p95 | p99 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for h_key in ["H2", "H3", "H5"]:
            h_stat = horizon_summaries[h_key]
            seq_p = h_stat["sequence_length"]["percentiles"]
            pct = round(h_stat["eligible_events"] / total_unique_events * 100, 2)
            f.write(f"| **{h_key} ({h_stat['horizon_days']}d)** | **{h_stat['eligible_events']:,}** | {pct}% | {h_stat['total_cdms']:,} | {h_stat['sequence_length']['min']} | {h_stat['sequence_length']['max']} | {h_stat['sequence_length']['mean']} | {h_stat['sequence_length']['median']} | {seq_p['p25']} | {seq_p['p75']} | {seq_p['p90']} | {seq_p['p95']} | {seq_p['p99']} |\n")
        f.write("\n*Note on H = 7 days: Verified in audit that max(time_to_tca) = 6.9938 days, so 7-day cutoff produces exactly 0 eligible events and was not created.*\n\n")
        f.write("---\n\n")

        f.write("## 3. Target Distribution (`final_risk`)\n\n")
        f.write("| Horizon | Valid Targets | Missing Targets | Min | Max | Mean | Median | p10 | p25 | p50 | p75 | p90 | p95 | p99 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for h_key in ["H2", "H3", "H5"]:
            t_stat = horizon_summaries[h_key]["target"]
            t_p = t_stat["percentiles"]
            f.write(f"| **{h_key}** | {t_stat['valid_count']:,} | {t_stat['missing_count']} | {t_stat['min']:.4f} | {t_stat['max']:.4f} | {t_stat['mean']:.4f} | {t_stat['median']:.4f} | {t_p['p10']:.4f} | {t_p['p25']:.4f} | {t_p['p50']:.4f} | {t_p['p75']:.4f} | {t_p['p90']:.4f} | {t_p['p95']:.4f} | {t_p['p99']:.4f} |\n")
        f.write("\n---\n\n")

        f.write("## 4. Missing Values per DIRECT Feature by Horizon\n\n")
        f.write("| # | Feature Name | H=2d Missing (%) | H=3d Missing (%) | H=5d Missing (%) |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, col in enumerate(DIRECT_FEATURE_COLUMNS, 1):
            m2 = horizon_summaries["H2"]["feature_missing_counts"][col]
            m3 = horizon_summaries["H3"]["feature_missing_counts"][col]
            m5 = horizon_summaries["H5"]["feature_missing_counts"][col]
            f.write(f"| {idx} | `{col}` | {m2['missing_count']:,} ({m2['missing_percentage']}%) | {m3['missing_count']:,} ({m3['missing_percentage']}%) | {m5['missing_count']:,} ({m5['missing_percentage']}%) |\n")

        f.write("\n---\n\n")
        f.write("## 5. Excluded Features Registry\n\n")
        f.write("### Excluded `NEEDS_HUMAN_REVIEW` Features (18 items)\n")
        f.write("The following features remain excluded pending human domain sign-off:\n")
        for item in NEEDS_HUMAN_REVIEW_FEATURES:
            f.write(f"- `{item}`\n")
        f.write("\n### Excluded `NOT_AVAILABLE` Features (8 items)\n")
        for item in NOT_AVAILABLE_FEATURES:
            f.write(f"- `{item}`\n")
        f.write("\n### Excluded Identifiers & Target Leakage Controls\n")
        f.write("- `event_id`: Grouping identifier only\n")
        f.write("- `mission_id`: Excluded from feature representation\n")
        f.write("- `risk`: Excluded from inputs; used strictly as label source (`final_risk`)\n")

    print(f"Wrote summary report to {summary_path}")
    print("\nEVENT-LEVEL DATASET CONSTRUCTION PIPELINE SUCCESSFUL.")


if __name__ == "__main__":
    main()
