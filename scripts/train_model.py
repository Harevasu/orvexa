"""Train conjunction risk baseline models (Physics ESA baseline, Ridge, XGBoost)."""

import argparse
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scripts.run_phase1_baselines import main as run_baselines_main


def main() -> None:
    parser = argparse.ArgumentParser(description="ORVEXA Phase 1 Baseline Training")
    parser.add_argument("--horizon", choices=["2", "3", "5", "all"], default="all", help="Horizon to train")
    parser.add_argument("--model", choices=["esa", "ridge", "xgboost", "all"], default="all", help="Model family to train")
    args = parser.parse_args()

    print(f"Starting ORVEXA Phase 1 Baseline Training Pipeline (horizon: {args.horizon}, model: {args.model})...")
    run_baselines_main()


if __name__ == "__main__":
    main()
