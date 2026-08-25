"""Command-line interface for the ORVEXA pipeline.

Provides commands for:
- audit-data: Validates raw ESA CSV and produces schema report
- build-events: Builds horizon-specific event views and sequences
- make-splits: Generates event-aware chronological partitions
- train: Trains models (physics, linear, XGBoost, TCN)
- evaluate: Computes metrics, rankings, calibration, and intervals
- run-ablation: Runs ablation experiments
- explain: Generates model explanations and attributions
"""

import sys


def main() -> None:
    """CLI entry point placeholder."""
    print("ORVEXA CLI - Implementation pending.")
    sys.exit(0)


if __name__ == "__main__":
    main()
