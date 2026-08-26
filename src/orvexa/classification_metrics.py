"""Classification and probabilistic metrics: PR-AUC, ROC-AUC, Brier score, and Expected Calibration Error (ECE)."""

import math
from typing import Any, Dict, List, Optional, Tuple


def compute_brier_score(y_true: List[int], y_prob: List[float]) -> float:
    """Compute mean squared error between predicted probabilities and binary outcomes."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_prob)) / len(y_true)


def compute_expected_calibration_error(
    y_true: List[int], y_prob: List[float], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) across probability bins."""
    n = len(y_true)
    if n == 0:
        return 0.0

    bins: List[List[Tuple[int, float]]] = [[] for _ in range(n_bins)]
    for yt, yp in zip(y_true, y_prob):
        p_clamped = min(max(yp, 0.0), 1.0)
        bin_idx = min(int(p_clamped * n_bins), n_bins - 1)
        bins[bin_idx].append((yt, p_clamped))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        bin_size = len(b)
        avg_confidence = sum(p for _, p in b) / bin_size
        avg_accuracy = sum(y for y, _ in b) / bin_size
        ece += (bin_size / n) * abs(avg_accuracy - avg_confidence)

    return ece


def compute_pr_auc(y_true: List[int], y_prob: List[float]) -> float:
    """Compute Area Under the Precision-Recall Curve (PR-AUC) using trapezoidal rule."""
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0

    # Sort by predicted probability descending
    paired = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)

    precisions = [1.0]
    recalls = [0.0]

    tp = 0
    fp = 0

    for _, yt in paired:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        rec = tp / n_pos
        prec = tp / (tp + fp)
        recalls.append(rec)
        precisions.append(prec)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(recalls)):
        dx = recalls[i] - recalls[i - 1]
        avg_y = (precisions[i] + precisions[i - 1]) / 2.0
        auc += dx * avg_y

    return min(max(auc, 0.0), 1.0)


def compute_classification_metrics(
    y_true_binary: List[int], y_prob: List[float], n_bins: int = 10
) -> Dict[str, Any]:
    """Compute PR-AUC, Brier score, and Expected Calibration Error (ECE).
    
    Args:
        y_true_binary: Binary truth indicators (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of bins for ECE calculation.
        
    Returns:
        Dictionary of probabilistic classification metrics.
    """
    if not y_true_binary or not y_prob:
        raise ValueError("Input lists cannot be empty.")
    if len(y_true_binary) != len(y_prob):
        raise ValueError(f"Length mismatch: {len(y_true_binary)} vs {len(y_prob)}.")

    brier = compute_brier_score(y_true_binary, y_prob)
    ece = compute_expected_calibration_error(y_true_binary, y_prob, n_bins=n_bins)
    pr_auc = compute_pr_auc(y_true_binary, y_prob)

    return {
        "n_samples": len(y_true_binary),
        "positive_count": sum(y_true_binary),
        "brier_score": round(brier, 5),
        "expected_calibration_error": round(ece, 5),
        "pr_auc": round(pr_auc, 5),
    }
