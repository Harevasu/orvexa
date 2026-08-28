"""
Inspect candidate freeze manifests and blind test metrics.
"""
import os
import json
import pandas as pd

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def inspect_historical_metrics():
    print("=== HISTORICAL BLIND TEST METRICS ===")
    p3b_summary = os.path.join(WORKSPACE, "reports", "phase3b", "step4b_blind_test_summary.json")
    if os.path.exists(p3b_summary):
        with open(p3b_summary, "r") as f:
            p3b = json.load(f)
        print("Phase 3B Blind Test Summary:")
        print(json.dumps(p3b, indent=2))

    p4a_summary = os.path.join(WORKSPACE, "reports", "phase4a", "step4_blind_test_summary.json")
    if os.path.exists(p4a_summary):
        with open(p4a_summary, "r") as f:
            p4a = json.load(f)
        print("Phase 4A Step 4 Blind Test Summary:")
        print(json.dumps(p4a, indent=2))

    p4a_correction = os.path.join(WORKSPACE, "reports", "phase4a", "step5_scientific_record_correction.json")
    if os.path.exists(p4a_correction):
        with open(p4a_correction, "r") as f:
            p4a_c = json.load(f)
        print("Phase 4A Step 5 Record Correction:")
        print(json.dumps(p4a_c, indent=2))

if __name__ == "__main__":
    inspect_historical_metrics()
