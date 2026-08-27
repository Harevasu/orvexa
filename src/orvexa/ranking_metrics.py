"""Operational alert budget ranking metrics: Recall@K, Precision@K, NDCG@K, and missed high-risk events."""

import math
from typing import Any, Dict, List, Optional


def compute_ndcg_at_k(relevance_ordered: List[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    if k <= 0 or not relevance_ordered:
        return 0.0

    subset = relevance_ordered[:k]
    # DCG = sum_i (2^rel - 1) / log2(i + 2)
    dcg = sum((math.pow(2, rel) - 1.0) / math.log2(i + 2.0) for i, rel in enumerate(subset))

    # Ideal DCG
    ideal_ordered = sorted(relevance_ordered, reverse=True)[:k]
    idcg = sum((math.pow(2, rel) - 1.0) / math.log2(i + 2.0) for i, rel in enumerate(ideal_ordered))

    if idcg <= 1e-12:
        return 1.0 if dcg <= 1e-12 else 0.0
    return dcg / idcg


def compute_ranking_metrics(
    y_true_risk: List[float],
    y_pred_risk: List[float],
    alert_budgets: Optional[List[float]] = None,
    threshold_log10: float = -5.0,
) -> Dict[str, Any]:
    """Compute Recall@K, Precision@K, NDCG@K, and missed events under operator alert budgets.
    
    Args:
        y_true_risk: True final event log-risk scores.
        y_pred_risk: Predicted continuous event log-risk scores.
        alert_budgets: List of alert budget fractions (default: [0.01, 0.05, 0.10] for 1%, 5%, 10%).
        threshold_log10: Domain log-risk threshold defining high-risk events (e.g. -5.0).
        
    Returns:
        Dictionary of operational ranking metrics across each alert budget.
    """
    if not y_true_risk or not y_pred_risk:
        raise ValueError("Inputs cannot be empty.")
    if len(y_true_risk) != len(y_pred_risk):
        raise ValueError(f"Length mismatch: y_true ({len(y_true_risk)}) vs y_pred ({len(y_pred_risk)}).")

    budgets = alert_budgets or [0.01, 0.05, 0.10]
    n_total = len(y_true_risk)

    # 1. Binary ground truth: is event high risk?
    binary_true = [1 if yt >= threshold_log10 else 0 for yt in y_true_risk]
    n_high_risk = sum(binary_true)

    # 2. Sort by predicted risk descending
    indexed_preds = list(enumerate(zip(y_pred_risk, binary_true)))
    indexed_preds.sort(key=lambda x: x[1][0], reverse=True)

    ordered_binary_relevance = [item[1][1] for item in indexed_preds]

    results: Dict[str, Any] = {
        "total_events": n_total,
        "high_risk_events_count": n_high_risk,
        "high_risk_prevalence": n_high_risk / max(n_total, 1),
        "threshold_log10": threshold_log10,
    }

    for b in budgets:
        pct_label = int(round(b * 100))
        k = max(1, int(math.floor(b * n_total)))
        k = min(k, n_total)

        top_k_binary = ordered_binary_relevance[:k]
        tp_in_top_k = sum(top_k_binary)

        recall_k = tp_in_top_k / n_high_risk if n_high_risk > 0 else 1.0
        precision_k = tp_in_top_k / k if k > 0 else 0.0
        missed_k = n_high_risk - tp_in_top_k
        ndcg_k = compute_ndcg_at_k(ordered_binary_relevance, k)

        results[f"budget_pct_{pct_label}"] = {
            "budget_fraction": b,
            "cutoff_rank_k": k,
            "true_positives": tp_in_top_k,
            "recall": round(recall_k, 5),
            "precision": round(precision_k, 5),
            "missed_high_risk": missed_k,
            "ndcg": round(ndcg_k, 5),
        }

    return results
