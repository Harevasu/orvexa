"""Gradient Boosted Decision Tree models for snapshot and temporal-summary risk estimation."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union


class DecisionTreeNode:
    """Represents a node in a decision tree (split or leaf)."""

    def __init__(
        self,
        is_leaf: bool,
        value: float = 0.0,
        feature_index: int = 0,
        threshold: float = 0.0,
        gain: float = 0.0,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        n_samples: int = 0,
    ) -> None:
        self.is_leaf = is_leaf
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.gain = gain
        self.left = left
        self.right = right
        self.n_samples = n_samples

    def to_dict(self) -> Dict[str, Any]:
        if self.is_leaf:
            return {"type": "leaf", "value": self.value, "n_samples": self.n_samples}
        return {
            "type": "split",
            "feature": self.feature_index,
            "threshold": self.threshold,
            "gain": self.gain,
            "n_samples": self.n_samples,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionTreeNode":
        if data.get("type") == "leaf" or data.get("is_leaf"):
            return cls(is_leaf=True, value=data.get("value", 0.0), n_samples=data.get("n_samples", 0))
        left_node = cls.from_dict(data["left"]) if data.get("left") else None
        right_node = cls.from_dict(data["right"]) if data.get("right") else None
        return cls(
            is_leaf=False,
            feature_index=data.get("feature", 0),
            threshold=data.get("threshold", 0.0),
            gain=data.get("gain", 0.0),
            left=left_node,
            right=right_node,
            n_samples=data.get("n_samples", 0),
        )


class FastRegressionTree:
    """Histogram-binned regression tree for gradient boosted residual fitting."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[DecisionTreeNode] = None

    def fit(
        self,
        X_binned: Any,
        residuals: Any,
        bin_thresholds: List[List[float]],
        col_indices: List[int],
        feature_gains: Optional[List[float]] = None,
    ) -> "FastRegressionTree":
        """Fit decision tree on binned integer matrices and float residuals."""
        import numpy as np

        self.root = self._build_tree(
            X_binned, residuals, bin_thresholds, col_indices, depth=0, feature_gains=feature_gains
        )
        return self

    def _build_tree(
        self,
        X_b: Any,
        res: Any,
        bin_thresholds: List[List[float]],
        col_indices: List[int],
        depth: int,
        feature_gains: Optional[List[float]] = None,
    ) -> DecisionTreeNode:
        import numpy as np

        n_samples = len(res)
        leaf_val = float(np.mean(res)) if n_samples > 0 else 0.0

        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return DecisionTreeNode(is_leaf=True, value=leaf_val, n_samples=n_samples)

        total_sum = float(np.sum(res))
        total_sq = float(np.sum(res**2))
        best_gain = 0.0
        best_feat = -1
        best_bin = -1
        best_thresh = 0.0

        for f_idx in col_indices:
            col_b = X_b[:, f_idx]
            max_b = int(np.max(col_b)) if len(col_b) > 0 else 0
            if max_b <= 0:
                continue

            counts = np.bincount(col_b, minlength=max_b + 1)
            sums = np.bincount(col_b, weights=res, minlength=max_b + 1)

            left_count = 0
            left_sum = 0.0

            for b in range(max_b):
                c = counts[b]
                if c == 0:
                    continue
                left_count += c
                left_sum += sums[b]
                right_count = n_samples - left_count

                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                right_sum = total_sum - left_sum
                gain = (left_sum**2) / left_count + (right_sum**2) / right_count - (total_sum**2) / n_samples
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_bin = b
                    thresh_list = bin_thresholds[f_idx]
                    best_thresh = thresh_list[b] if b < len(thresh_list) else 0.0

        if best_gain <= 1e-7 or best_feat < 0:
            return DecisionTreeNode(is_leaf=True, value=leaf_val, n_samples=n_samples)

        if feature_gains is not None and best_feat < len(feature_gains):
            feature_gains[best_feat] += best_gain

        left_mask = X_b[:, best_feat] <= best_bin
        right_mask = ~left_mask

        left_child = self._build_tree(
            X_b[left_mask], res[left_mask], bin_thresholds, col_indices, depth + 1, feature_gains
        )
        right_child = self._build_tree(
            X_b[right_mask], res[right_mask], bin_thresholds, col_indices, depth + 1, feature_gains
        )

        return DecisionTreeNode(
            is_leaf=False,
            feature_index=best_feat,
            threshold=best_thresh,
            gain=best_gain,
            left=left_child,
            right=right_child,
            n_samples=n_samples,
        )

    def predict_one(self, node: DecisionTreeNode, x: Any) -> float:
        if node.is_leaf:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self.predict_one(node.left, x) if node.left else node.value
        else:
            return self.predict_one(node.right, x) if node.right else node.value

    def predict(self, X: Any) -> Any:
        import numpy as np

        if self.root is None:
            return np.zeros(len(X), dtype=np.float64)
        return np.array([self.predict_one(self.root, row) for row in X], dtype=np.float64)


class XGBoostRiskModel:
    """Gradient boosted decision tree model for conjunction event risk prediction."""

    model_name: str = "xgboost_risk"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.feature_names = feature_names or []
        self.config = config or {}
        self.backend_: str = "orvexa_gbdt"
        self.base_prediction_: float = -20.0
        self.trees_: List[FastRegressionTree] = []
        self.feature_importances_: List[float] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: List[List[float]],
        y_train: List[float],
        X_valid: Optional[List[List[float]]] = None,
        y_valid: Optional[List[float]] = None,
    ) -> "XGBoostRiskModel":
        """Fit gradient boosted trees on training data with validation tracking."""
        if not X_train or not y_train:
            raise ValueError("Training data cannot be empty.")
        if len(X_train) != len(y_train):
            raise ValueError(f"Feature count ({len(X_train)}) does not match target count ({len(y_train)}).")

        import numpy as np

        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64)
        n_samples, n_features = X_tr.shape

        if not self.feature_names:
            self.feature_names = [f"f_{i}" for i in range(n_features)]

        # Quantile histogram binning (32 bins)
        n_bins = 32
        bin_thresholds: List[List[float]] = []
        X_train_binned = np.zeros_like(X_tr, dtype=np.int32)

        for j in range(n_features):
            col = X_tr[:, j]
            qs = [float(np.percentile(col, q)) for q in np.linspace(0, 100, n_bins + 1)[1:-1]]
            unique_qs = sorted(list(set(qs)))
            bin_thresholds.append(unique_qs)
            if unique_qs:
                X_train_binned[:, j] = np.digitize(col, unique_qs)
            else:
                X_train_binned[:, j] = 0

        rng = random.Random(self.random_state)
        self.base_prediction_ = float(np.mean(y_tr))
        f_train = np.full_like(y_tr, self.base_prediction_)

        k_cols = max(1, int(round(self.colsample_bytree * n_features)))
        k_rows = max(1, int(round(self.subsample * n_samples)))

        raw_gains = [0.0] * n_features
        self.trees_ = []

        for _ in range(self.n_estimators):
            residuals = y_tr - f_train

            # Subsample rows and columns deterministically
            row_idx = rng.sample(range(n_samples), min(k_rows, n_samples))
            col_idx = sorted(rng.sample(range(n_features), min(k_cols, n_features)))

            tree = FastRegressionTree(
                max_depth=self.max_depth,
                min_samples_split=10,
                min_samples_leaf=5,
            )
            tree.fit(
                X_train_binned[row_idx],
                residuals[row_idx],
                bin_thresholds,
                col_idx,
                feature_gains=raw_gains,
            )
            self.trees_.append(tree)

            step_preds = tree.predict(X_tr)
            f_train += self.learning_rate * step_preds

        # Normalize feature importances
        total_gain = sum(raw_gains)
        if total_gain > 1e-12:
            self.feature_importances_ = [float(g / total_gain) for g in raw_gains]
        else:
            self.feature_importances_ = [1.0 / n_features] * n_features

        self.backend_ = "orvexa_gbdt_hist"
        self.is_fitted_ = True
        return self

    def predict_risk(self, X: List[List[float]]) -> List[float]:
        """Predict continuous log-risk score for each feature record."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predicting.")

        import numpy as np

        X_arr = np.asarray(X, dtype=np.float64)
        preds = np.full(len(X_arr), self.base_prediction_, dtype=np.float64)

        for tree in self.trees_:
            preds += self.learning_rate * tree.predict(X_arr)

        return [float(p) for p in preds]

    def predict_probability(self, X: List[List[float]], threshold_log10: float) -> List[float]:
        """Predict probability that risk exceeds threshold_log10."""
        scores = self.predict_risk(X)
        probs: List[float] = []
        for s in scores:
            diff = s - threshold_log10
            try:
                p = 1.0 / (1.0 + math.exp(-2.0 * diff))
            except OverflowError:
                p = 1.0 if diff > 0 else 0.0
            probs.append(p)
        return probs

    def save(self, path: Union[Path, str]) -> None:
        """Serialize model artifact and metadata to disk."""
        p = Path(path)
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)

        tree_dicts = [tree.root.to_dict() for tree in self.trees_ if tree.root]

        meta = {
            "model_name": self.model_name,
            "backend": self.backend_,
            "n_estimators": len(self.trees_),
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "base_prediction": self.base_prediction_,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances_,
            "is_fitted": self.is_fitted_,
            "config": self.config,
            "trees": tree_dicts,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "XGBoostRiskModel":
        """Load model metadata and tree structures from disk."""
        p = Path(path)
        if p.suffix != ".json":
            p = p.with_suffix(".json")
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            n_estimators=meta.get("n_estimators", 100),
            max_depth=meta.get("max_depth", 5),
            learning_rate=meta.get("learning_rate", 0.05),
            subsample=meta.get("subsample", 0.8),
            colsample_bytree=meta.get("colsample_bytree", 0.8),
            random_state=meta.get("random_state", 42),
            feature_names=meta.get("feature_names", []),
            config=meta.get("config", {}),
        )
        instance.backend_ = meta.get("backend", "orvexa_gbdt_hist")
        instance.base_prediction_ = meta.get("base_prediction", -20.0)
        instance.feature_importances_ = meta.get("feature_importances", [])
        instance.is_fitted_ = meta.get("is_fitted", True)

        instance.trees_ = []
        for t_dict in meta.get("trees", []):
            tree = FastRegressionTree(max_depth=instance.max_depth)
            tree.root = DecisionTreeNode.from_dict(t_dict)
            instance.trees_.append(tree)

        return instance
