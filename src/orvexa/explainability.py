"""Model explainability: Tree SHAP attributions and TCN temporal occlusion.

Disclaimer: Feature attributions indicate model prediction mechanisms, not physical causality.
"""

from typing import Any, Dict


def explain_tree_model(model: Any, X_sample: Any) -> Dict[str, Any]:
    """Compute global and local SHAP feature attributions for tree models."""
    raise NotImplementedError("SHAP explainability not yet implemented.")


def explain_tcn_temporal(model: Any, sequence_tensor: Any, mask_tensor: Any) -> Dict[str, Any]:
    """Compute temporal occlusion feature importance for TCN models."""
    raise NotImplementedError("TCN occlusion explainability not yet implemented.")
