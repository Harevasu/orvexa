"""Orbital reference frame transformations (TEME, ECI/J2000, ECEF, RTN, Geodetic).

References:
- Vallado (2013), Fundamentals of Astrodynamics and Applications (4th Ed).
- Adapted frame definitions from reference_repo/satkit (src/frametransform/mod.rs)
  and reference_repo/CARA_Analysis_Tools (DistributedMatlab/Utils/CoordinateTransformations/).
"""

import math
from typing import List, Tuple


# WGS-84 Earth ellipsoid constants
WGS84_A_KM = 6378.137  # Semi-major axis in km
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)  # Semi-minor axis in km
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F  # First eccentricity squared


def julian_date(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0
) -> float:
    """Calculate Julian Date (JD) from calendar date and time (UTC)."""
    if month <= 2:
        year -= 1
        month += 12

    a = math.floor(year / 100.0)
    b = 2 - a + math.floor(a / 4.0)

    day_fraction = day + (hour + minute / 60.0 + second / 3600.0) / 24.0
    jd = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day_fraction + b - 1524.5
    return jd


def greenwich_mean_sidereal_time_rad(jd: float) -> float:
    """Compute Greenwich Mean Sidereal Time (GMST) in radians for a given Julian Date."""
    t_ut1 = (jd - 2451545.0) / 36525.0
    # IAU 1982 GMST expression in seconds
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t_ut1
        + 0.093104 * (t_ut1 ** 2)
        - 6.2e-6 * (t_ut1 ** 3)
    )
    # Convert seconds to radians modulo 2*pi
    gmst_rad = (gmst_sec * (2.0 * math.pi / 86400.0)) % (2.0 * math.pi)
    if gmst_rad < 0:
        gmst_rad += 2.0 * math.pi
    return gmst_rad


def teme_to_geodetic(x: float, y: float, z: float, epoch_jd: float) -> Tuple[float, float, float]:
    """Convert TEME Cartesian coordinates (km) to Geodetic Latitude (deg), Longitude (deg), Altitude (km).
    
    Uses Bowring's iterative method on the WGS-84 ellipsoid with GMST rotation.
    
    Args:
        x: X-coordinate in km (TEME frame).
        y: Y-coordinate in km (TEME frame).
        z: Z-coordinate in km (TEME frame).
        epoch_jd: Julian Date of the coordinates.
        
    Returns:
        Tuple of (latitude_degrees, longitude_degrees, altitude_km).
    """
    # 1. Rotate TEME into Earth-Fixed (ECEF) via GMST
    theta = greenwich_mean_sidereal_time_rad(epoch_jd)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    x_ecef = cos_t * x + sin_t * y
    y_ecef = -sin_t * x + cos_t * y
    z_ecef = z

    # 2. Longitude in degrees [-180, +180]
    lon_rad = math.atan2(y_ecef, x_ecef)
    lon_deg = math.degrees(lon_rad)

    # 3. Bowring's method for geodetic latitude and altitude
    p = math.sqrt(x_ecef * x_ecef + y_ecef * y_ecef)
    if p < 1e-6:
        # Near poles
        lat_deg = 90.0 if z_ecef >= 0 else -90.0
        alt_km = abs(z_ecef) - WGS84_B_KM
        return lat_deg, lon_deg, alt_km

    # Parametric angle approximation
    e_prime_sq = (WGS84_A_KM * WGS84_A_KM - WGS84_B_KM * WGS84_B_KM) / (WGS84_B_KM * WGS84_B_KM)
    u = math.atan2(z_ecef * WGS84_A_KM, p * WGS84_B_KM)

    sin_u = math.sin(u)
    cos_u = math.cos(u)

    lat_rad = math.atan2(
        z_ecef + e_prime_sq * WGS84_B_KM * (sin_u ** 3),
        p - WGS84_E2 * WGS84_A_KM * (cos_u ** 3),
    )

    # Radius of curvature in prime vertical
    sin_lat = math.sin(lat_rad)
    n_rad = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    alt_km = p / math.cos(lat_rad) - n_rad
    lat_deg = math.degrees(lat_rad)

    return lat_deg, lon_deg, alt_km


def eci_to_rtn_matrix(
    pos: Tuple[float, float, float] | List[float],
    vel: Tuple[float, float, float] | List[float],
) -> List[List[float]]:
    """Compute 3x3 rotation matrix from ECI/J2000 to Radial-Transverse-Normal (RTN/RIC) frame.
    
    Basis definitions:
    - R_hat = r / |r| (radial out)
    - N_hat = (r x v) / |r x v| (normal/cross-track)
    - T_hat = N_hat x R_hat (transverse/along-track)
    
    Returns:
        3x3 rotation matrix M such that v_RTN = M * v_ECI.
    """
    rx, ry, rz = pos[0], pos[1], pos[2]
    vx, vy, vz = vel[0], vel[1], vel[2]

    # R_hat
    r_norm = math.sqrt(rx * rx + ry * ry + rz * rz)
    if r_norm < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rx_hat, ry_hat, rz_hat = rx / r_norm, ry / r_norm, rz / r_norm

    # Angular momentum: h = r x v
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h_norm = math.sqrt(hx * hx + hy * hy + hz * hz)

    if h_norm < 1e-9:
        # Radial motion fallback
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    nx_hat, ny_hat, nz_hat = hx / h_norm, hy / h_norm, hz / h_norm

    # T_hat = N_hat x R_hat
    tx_hat = ny_hat * rz_hat - nz_hat * ry_hat
    ty_hat = nz_hat * rx_hat - nx_hat * rz_hat
    tz_hat = nx_hat * ry_hat - ny_hat * rx_hat

    # Matrix rows are R_hat, T_hat, N_hat
    return [
        [rx_hat, ry_hat, rz_hat],
        [tx_hat, ty_hat, tz_hat],
        [nx_hat, ny_hat, nz_hat],
    ]


def rtn_to_eci_matrix(
    pos: Tuple[float, float, float] | List[float],
    vel: Tuple[float, float, float] | List[float],
) -> List[List[float]]:
    """Compute 3x3 rotation matrix from RTN to ECI/J2000 (transpose of ECI-to-RTN)."""
    m = eci_to_rtn_matrix(pos, vel)
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
