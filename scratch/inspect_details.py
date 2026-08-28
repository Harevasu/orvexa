"""
Inspect models, preprocessors, splitting, and metrics in detail.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"
sys.path.insert(0, os.path.join(WORKSPACE, "src"))

def inspect_details():
    import orvexa.models_tcn as mt
    import orvexa.preprocessing_phase3b as prep
    import orvexa.splitting as sp
    import orvexa.regression_metrics as rm
    import orvexa.ranking_metrics as rkm

    print("=== TCN ARCHITECTURE & LOSSES ===")
    print("Classes in models_tcn:", [x for x in dir(mt) if not x.startswith("_")])
    # check HuberLoss or loss functions
    if hasattr(mt, "HuberLoss"):
        print("mt.HuberLoss exists")
    if hasattr(mt, "TemporalLoss"):
        print("mt.TemporalLoss exists")
    if hasattr(mt, "TCNModel"):
        print("TCNModel class found")

    print("\n=== PREPROCESSING PHASE 3B ===")
    print("Classes in prep:", [x for x in dir(prep) if not x.startswith("_")])
    
    print("\n=== SPLITTING LOGIC ===")
    print("Functions in splitting:", [x for x in dir(sp) if not x.startswith("_")])

    print("\n=== REGRESSION METRICS ===")
    print("Functions in rm:", [x for x in dir(rm) if not x.startswith("_")])

    print("\n=== RANKING METRICS ===")
    print("Functions in rkm:", [x for x in dir(rkm) if not x.startswith("_")])

if __name__ == "__main__":
    inspect_details()
