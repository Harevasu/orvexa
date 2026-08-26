"""Physics-derived baseline and analytical collision probability models.

References and Attributions:
- Foster (1992): "The Analytic Calculation of Sub-Satellite Collision Probabilities", NASA.
- Frisbee (2015): "Maximum Probability of Collision with Single Object Positional Uncertainty", AAS/AIAA.
- Adapted algorithmic structure from reference_repo/orbveil (src/orbveil/core/probability.py, Apache-2.0)
  and reference_repo/CARA_Analysis_Tools (DistributedMatlab/ProbabilityOfCollision/Pc2D_Foster.m, NOSA v1.3).
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def project_to_encounter_bplane(
    rel_pos: Tuple[float, float, float] | List[float],
    rel_vel: Tuple[float, float, float] | List[float],
    cov_3x3: List[List[float]],
) -> Tuple[List[float], List[List[float]]]:
    """Project 3D encounter geometry and combined covariance onto the B-plane (conjunction plane).
    
    The B-plane is orthogonal to the relative velocity vector.
    
    Attribution: Formulated based on Foster (1992) and reference_repo/orbveil.
    
    Args:
        rel_pos: 3D relative position vector [r, t, n] in km.
        rel_vel: 3D relative velocity vector [vr, vt, vn] in km/s.
        cov_3x3: 3x3 combined position covariance matrix in km^2.
        
    Returns:
        Tuple of (miss_2d [2], cov_2d [2x2]).
    """
    rx, ry, rz = rel_pos[0], rel_pos[1], rel_pos[2]
    vx, vy, vz = rel_vel[0], rel_vel[1], rel_vel[2]
    
    v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if v_norm < 1e-9:
        # Near-zero relative velocity fallback
        return [rx, ry], [[cov_3x3[0][0], cov_3x3[0][1]], [cov_3x3[1][0], cov_3x3[1][1]]]

    # Unit vector along relative velocity (z_hat)
    zx, zy, zz = vx / v_norm, vy / v_norm, vz / v_norm

    # Choose reference vector to construct orthogonal basis
    ref_x, ref_y, ref_z = 0.0, 0.0, 1.0
    # Cross product: z_hat x ref
    xx = zy * ref_z - zz * ref_y
    xy = zz * ref_x - zx * ref_z
    xz = zx * ref_y - zy * ref_x
    x_norm = math.sqrt(xx * xx + xy * xy + xz * xz)

    if x_norm < 1e-9:
        # z_hat is parallel to z-axis; use x-axis as reference
        ref_x, ref_y, ref_z = 1.0, 0.0, 0.0
        xx = zy * ref_z - zz * ref_y
        xy = zz * ref_x - zx * ref_z
        xz = zx * ref_y - zy * ref_x
        x_norm = math.sqrt(xx * xx + xy * xy + xz * xz)

    # Unit vector x_hat in B-plane
    xx, xy, xz = xx / x_norm, xy / x_norm, xz / x_norm

    # Unit vector y_hat = z_hat x x_hat
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx

    # Projection matrix P (2x3): row 0 is x_hat, row 1 is y_hat
    # 2D miss vector = P * rel_pos
    m0 = xx * rx + xy * ry + xz * rz
    m1 = yx * rx + yy * ry + yz * rz
    miss_2d = [m0, m1]

    # Project covariance: C_2d = P * Cov * P^T
    # Step 1: M = Cov * P^T (3x2)
    m_col0 = [
        cov_3x3[0][0] * xx + cov_3x3[0][1] * xy + cov_3x3[0][2] * xz,
        cov_3x3[1][0] * xx + cov_3x3[1][1] * xy + cov_3x3[1][2] * xz,
        cov_3x3[2][0] * xx + cov_3x3[2][1] * xy + cov_3x3[2][2] * xz,
    ]
    m_col1 = [
        cov_3x3[0][0] * yx + cov_3x3[0][1] * yy + cov_3x3[0][2] * yz,
        cov_3x3[1][0] * yx + cov_3x3[1][1] * yy + cov_3x3[1][2] * yz,
        cov_3x3[2][0] * yx + cov_3x3[2][1] * yy + cov_3x3[2][2] * yz,
    ]

    # Step 2: C_2d = P * M (2x2)
    c00 = xx * m_col0[0] + xy * m_col0[1] + xz * m_col0[2]
    c01 = xx * m_col1[0] + xy * m_col1[1] + xz * m_col1[2]
    c10 = yx * m_col0[0] + yy * m_col0[1] + yz * m_col0[2]
    c11 = yx * m_col1[0] + yy * m_col1[1] + yz * m_col1[2]

    # Symmetrize
    c_sym = 0.5 * (c01 + c10)
    cov_2d = [[c00, c_sym], [c_sym, c11]]

    return miss_2d, cov_2d


def compute_foster_2d_pc(
    miss_2d: List[float],
    cov_2d: List[List[float]],
    hard_body_radius: float = 0.01,
    n_theta_steps: int = 64,
    n_r_steps: int = 32,
) -> float:
    """Compute Foster (1992) analytical 2D collision probability.
    
    Integrates 2D Gaussian probability density over hard-body circular disc of radius HBR.
    
    Attribution: Foster (1992), adapted numerically in pure Python without external dependencies.
    
    Args:
        miss_2d: [x, y] coordinates of encounter miss vector in B-plane (km).
        cov_2d: 2x2 combined covariance matrix in B-plane (km^2).
        hard_body_radius: Combined hard-body collision radius in km (default: 0.01 km = 10 m).
        n_theta_steps: Numerical integration angular divisions.
        n_r_steps: Numerical integration radial divisions.
        
    Returns:
        Probability of collision in [0.0, 1.0].
    """
    if hard_body_radius <= 0:
        return 0.0

    c00, c01 = cov_2d[0][0], cov_2d[0][1]
    c10, c11 = cov_2d[1][0], cov_2d[1][1]
    det = c00 * c11 - c01 * c10

    if det <= 1e-20:
        return 0.0

    # Covariance inverse
    inv_c00 = c11 / det
    inv_c01 = -c01 / det
    inv_c10 = -c10 / det
    inv_c11 = c00 / det

    mx, my = miss_2d[0], miss_2d[1]
    norm_factor = 1.0 / (2.0 * math.pi * math.sqrt(det))

    # Numerical integration in polar coordinates (r from 0 to HBR, theta from 0 to 2*pi)
    # Midpoint rule integration
    dr = hard_body_radius / n_r_steps
    dtheta = (2.0 * math.pi) / n_theta_steps
    total_integral = 0.0

    for ir in range(n_r_steps):
        r = (ir + 0.5) * dr
        for itheta in range(n_theta_steps):
            theta = (itheta + 0.5) * dtheta
            # Physical point on the object disc
            px = r * math.cos(theta)
            py = r * math.sin(theta)
            
            # Difference vector from Gaussian center (miss vector)
            dx = px - mx
            dy = py - my

            # Exponent: -0.5 * (d^T * Cov^-1 * d)
            quad_form = dx * (inv_c00 * dx + inv_c01 * dy) + dy * (inv_c10 * dx + inv_c11 * dy)
            if quad_form < 100.0:  # Prevent underflow
                pdf_val = norm_factor * math.exp(-0.5 * quad_form)
                total_integral += pdf_val * r * dr * dtheta

    return min(max(total_integral, 0.0), 1.0)


def compute_frisbee_max_pc(
    miss_distance: float,
    hard_body_radius: float = 0.01,
) -> float:
    """Compute Frisbee (2015) / Alfano (2005) maximum possible collision probability under dilution.
    
    The theoretical upper bound on 2D collision probability occurs when the covariance
    eigenvalues are optimally distended along the miss vector.
    
    Formula: MaxPc = (HBR^2) / (e * d^2) for d >= HBR, capped at 1.0.
    
    Attribution: Frisbee (2015), Alfano (2005), reference_repo/CARA_Analysis_Tools.
    
    Args:
        miss_distance: Distance of closest approach in km.
        hard_body_radius: Combined hard-body radius in km (default: 0.01 km = 10 m).
        
    Returns:
        Maximum theoretical collision probability in [0.0, 1.0].
    """
    if hard_body_radius <= 0.0:
        return 0.0
    if miss_distance <= hard_body_radius:
        return 1.0

    # e = exp(1) approx 2.718281828459045
    ratio = hard_body_radius / miss_distance
    max_pc = (ratio * ratio) / math.e
    return min(max(max_pc, 0.0), 1.0)


def compute_mahalanobis_distance(
    rel_pos: List[float],
    cov_3x3: List[List[float]],
) -> float:
    """Calculate Mahalanobis distance between primary and secondary objects."""
    c00, c01, c02 = cov_3x3[0][0], cov_3x3[0][1], cov_3x3[0][2]
    c10, c11, c12 = cov_3x3[1][0], cov_3x3[1][1], cov_3x3[1][2]
    c20, c21, c22 = cov_3x3[2][0], cov_3x3[2][1], cov_3x3[2][2]

    # Determinant of 3x3
    det = (
        c00 * (c11 * c22 - c12 * c21)
        - c01 * (c10 * c22 - c12 * c20)
        + c02 * (c10 * c21 - c11 * c20)
    )
    if det <= 1e-25:
        # Fallback to Euclidean normalized
        return math.sqrt(sum(x * x for x in rel_pos))

    # Inverse 3x3
    inv00 = (c11 * c22 - c12 * c21) / det
    inv01 = (c02 * c21 - c01 * c22) / det
    inv02 = (c01 * c12 - c02 * c11) / det

    inv10 = (c12 * c20 - c10 * c22) / det
    inv11 = (c00 * c22 - c02 * c20) / det
    inv12 = (c02 * c10 - c00 * c12) / det

    inv20 = (c10 * c21 - c11 * c20) / det
    inv21 = (c01 * c20 - c00 * c21) / det
    inv22 = (c00 * c11 - c01 * c10) / det

    rx, ry, rz = rel_pos[0], rel_pos[1], rel_pos[2]
    vx = inv00 * rx + inv01 * ry + inv02 * rz
    vy = inv10 * rx + inv11 * ry + inv12 * rz
    vz = inv20 * rx + inv21 * ry + inv22 * rz

    quad = rx * vx + ry * vy + rz * vz
    return math.sqrt(max(quad, 0.0))


class PhysicsMaxRiskModel:
    """Baseline ranker and risk estimator derived directly from ESA max_risk_estimate.
    
    This model serves as the physical benchmark: it predicts the final event risk
    using the physics-derived max_risk_estimate (and optional scaling) provided in the CDM.
    """

    model_name: str = "physics_max_risk"

    def __init__(
        self,
        risk_col: str = "max_risk_estimate",
        scaling_col: Optional[str] = None,
        default_risk: float = -30.0,
    ) -> None:
        self.risk_col = risk_col
        self.scaling_col = scaling_col
        self.default_risk = default_risk

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_valid: Optional[Any] = None,
        y_valid: Optional[Any] = None,
    ) -> "PhysicsMaxRiskModel":
        """Physics baseline is deterministic and parameter-free; fit is a no-op."""
        return self

    def predict_risk(self, records: List[Dict[str, Any]]) -> List[float]:
        """Return max_risk_estimate as continuous log-risk predictor."""
        predictions: List[float] = []
        for rec in records:
            raw_risk = rec.get(self.risk_col)
            if raw_risk is None:
                val = self.default_risk
            else:
                try:
                    fval = float(raw_risk)
                    val = self.default_risk if (math.isnan(fval) or math.isinf(fval)) else fval
                except (ValueError, TypeError):
                    val = self.default_risk

            if self.scaling_col and self.scaling_col in rec:
                try:
                    scaling = float(rec[self.scaling_col])
                    if not (math.isnan(scaling) or math.isinf(scaling)):
                        val = val + scaling
                except (ValueError, TypeError):
                    pass

            predictions.append(val)
        return predictions

    def predict_probability(self, records: List[Dict[str, Any]], threshold_log10: float) -> List[float]:
        """Predict binary step probability that log-risk >= threshold_log10."""
        scores = self.predict_risk(records)
        # Soft-step probability transition using steep sigmoid around threshold
        probs: List[float] = []
        for s in scores:
            if s >= threshold_log10:
                probs.append(1.0)
            elif s < threshold_log10 - 2.0:
                probs.append(0.0)
            else:
                # Smooth 2-decade transition
                diff = s - threshold_log10
                p = 1.0 / (1.0 + math.exp(-3.0 * diff))
                probs.append(p)
        return probs

    def save(self, path: Union[Path, str]) -> None:
        """Serialize model configuration to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_name": self.model_name,
            "risk_col": self.risk_col,
            "scaling_col": self.scaling_col,
            "default_risk": self.default_risk,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "PhysicsMaxRiskModel":
        """Load model configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            risk_col=data.get("risk_col", "max_risk_estimate"),
            scaling_col=data.get("scaling_col"),
            default_risk=data.get("default_risk", -30.0),
        )
