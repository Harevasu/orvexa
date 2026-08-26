"""SGP4 and analytical orbit propagation wrapper for satellite state trajectories.

References:
- Vallado et al. (2006): "Revisiting Spacetrack Report #3: Rev 2", AIAA.
- Brandon Rhodes python-sgp4 wrapper (MIT License).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from orvexa.orbit.coordinate_utils import julian_date, teme_to_geodetic
from orvexa.orbit.tle_parser import parse_tle


def _kepler_solve(mean_anomaly_rad: float, eccentricity: float, max_iter: int = 30) -> float:
    """Solve Kepler's equation for eccentric anomaly E using Newton-Raphson."""
    e_anom = mean_anomaly_rad
    for _ in range(max_iter):
        f = e_anom - eccentricity * math.sin(e_anom) - mean_anomaly_rad
        f_prime = 1.0 - eccentricity * math.cos(e_anom)
        if abs(f_prime) < 1e-12:
            break
        delta = f / f_prime
        e_anom -= delta
        if abs(delta) < 1e-10:
            break
    return e_anom


def _two_body_propagate(
    elements: Dict[str, Any], dt_days: float
) -> Tuple[List[float], List[float]]:
    """Analytical two-body Keplerian state propagation fallback."""
    mu = 398600.4418  # Earth gravitational parameter km^3/s^2
    a = elements["semi_major_axis_km"]
    e = elements["eccentricity"]
    inc = math.radians(elements["inclination_deg"])
    raan0 = math.radians(elements["raan_deg"])
    argp = math.radians(elements["arg_perigee_deg"])
    m0 = math.radians(elements["mean_anomaly_deg"])
    n_rad_s = elements["mean_motion_rev_day"] * 2.0 * math.pi / 86400.0

    # J2 nodal regression rate approximation
    r_earth = 6378.137
    j2 = 1.08263e-3
    p = a * (1.0 - e * e)
    if p > 0:
        d_raan_dt = -1.5 * n_rad_s * j2 * ((r_earth / p) ** 2) * math.cos(inc)
    else:
        d_raan_dt = 0.0

    dt_sec = dt_days * 86400.0
    raan = raan0 + d_raan_dt * dt_sec
    m = (m0 + n_rad_s * dt_sec) % (2.0 * math.pi)

    # Solve eccentric anomaly
    e_anom = _kepler_solve(m, e)

    # True anomaly
    cos_e = math.cos(e_anom)
    sin_e = math.sin(e_anom)
    nu = math.atan2(math.sqrt(1.0 - e * e) * sin_e, cos_e - e)

    # Distance
    r = a * (1.0 - e * cos_e)

    # Position in orbital plane
    x_orb = r * math.cos(nu)
    y_orb = r * math.sin(nu)

    # Velocity in orbital plane
    h = math.sqrt(mu * a * (1.0 - e * e))
    vx_orb = -(mu / h) * math.sin(nu)
    vy_orb = (mu / h) * (e + math.cos(nu))

    # Rotation from perifocal to ECI
    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_inc, sin_inc = math.cos(inc), math.sin(inc)
    cos_argp, sin_argp = math.cos(argp), math.sin(argp)

    p11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_inc
    p12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc
    p21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_inc
    p22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc
    p31 = sin_argp * sin_inc
    p32 = cos_argp * sin_inc

    pos = [
        p11 * x_orb + p12 * y_orb,
        p21 * x_orb + p22 * y_orb,
        p31 * x_orb + p32 * y_orb,
    ]
    vel = [
        p11 * vx_orb + p12 * vy_orb,
        p21 * vx_orb + p22 * vy_orb,
        p31 * vx_orb + p32 * vy_orb,
    ]
    return pos, vel


def propagate_orbit(
    tle_line1: str,
    tle_line2: str,
    start_jd: float,
    end_jd: float,
    step_seconds: float = 60.0,
) -> List[Dict[str, Any]]:
    """Propagate satellite state trajectory from TLE over a time interval.
    
    Tries Python sgp4 library first; falls back to J2-perturbed Keplerian propagation.
    
    Args:
        tle_line1: TLE Line 1.
        tle_line2: TLE Line 2.
        start_jd: Start Julian Date.
        end_jd: End Julian Date.
        step_seconds: Step size in seconds.
        
    Returns:
        List of trajectory state dictionaries with timestamp, pos, vel, lat, lon, alt.
    """
    elements = parse_tle(tle_line1, tle_line2)

    # Compute TLE epoch JD
    epoch_year = elements["epoch_year"]
    epoch_day = elements["epoch_day"]
    epoch_jd_base = julian_date(epoch_year, 1, 1, 0, 0, 0) - 1.0 + epoch_day

    # Check for sgp4 availability
    sgp4_satrec = None
    try:
        from sgp4.api import Satrec, WGS72
        sgp4_satrec = Satrec.twoline2rv(tle_line1.strip(), tle_line2.strip(), WGS72)
    except (ImportError, Exception):
        sgp4_satrec = None

    trajectory: List[Dict[str, Any]] = []
    step_days = step_seconds / 86400.0
    current_jd = start_jd

    while current_jd <= end_jd + 1e-9:
        if sgp4_satrec is not None:
            # SGP4 propagation
            jd_int = int(current_jd)
            jd_frac = current_jd - jd_int
            err, r, v = sgp4_satrec.sgp4(jd_int, jd_frac)
            if err == 0:
                pos = [r[0], r[1], r[2]]
                vel = [v[0], v[1], v[2]]
            else:
                # Fallback on error
                dt_days = current_jd - epoch_jd_base
                pos, vel = _two_body_propagate(elements, dt_days)
        else:
            # Two-body fallback
            dt_days = current_jd - epoch_jd_base
            pos, vel = _two_body_propagate(elements, dt_days)

        lat, lon, alt = teme_to_geodetic(pos[0], pos[1], pos[2], current_jd)

        trajectory.append({
            "jd": current_jd,
            "x_km": pos[0],
            "y_km": pos[1],
            "z_km": pos[2],
            "vx_km_s": vel[0],
            "vy_km_s": vel[1],
            "vz_km_s": vel[2],
            "latitude_deg": lat,
            "longitude_deg": lon,
            "altitude_km": alt,
        })

        current_jd += step_days

    return trajectory
