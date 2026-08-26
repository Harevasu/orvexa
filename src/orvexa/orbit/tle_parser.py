"""Two-Line Element (TLE) parsing and structural validation.

References:
- Spacetrack Report No. 3: Models for Propagation of NORAD Element Sets (Hoots & Roehrich, 1980).
- Standard NORAD / NASA Two-Line Element Format specification.
"""

from typing import Any, Dict


def compute_tle_checksum(line: str) -> int:
    """Compute standard NORAD TLE line modulo-10 checksum.
    
    Letters, blanks, periods, and plus signs count as 0.
    Digits 0-9 count as their numeric value.
    Minus signs count as 1.
    """
    total = 0
    # Process first 68 characters (69th is the checksum itself)
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def parse_tle(line1: str, line2: str) -> Dict[str, Any]:
    """Parse standard two-line element set (TLE) into orbital parameters with validation.
    
    Args:
        line1: First line of TLE (69 characters).
        line2: Second line of TLE (69 characters).
        
    Returns:
        Dictionary of parsed orbital elements and metadata.
        
    Raises:
        ValueError: If TLE lines are malformed or fail checksum verification.
    """
    l1 = line1.strip()
    l2 = line2.strip()

    if len(l1) != 69 or len(l2) != 69:
        raise ValueError(f"Invalid TLE line length: Line 1 ({len(l1)} chars), Line 2 ({len(l2)} chars). Expected 69.")

    if not l1.startswith("1 ") or not l2.startswith("2 "):
        raise ValueError("Invalid TLE format: Line 1 must start with '1 ' and Line 2 with '2 '.")

    # Checksum validation
    chk1_expected = int(l1[68])
    chk1_computed = compute_tle_checksum(l1)
    if chk1_expected != chk1_computed:
        raise ValueError(f"TLE Line 1 checksum failure: expected {chk1_expected}, computed {chk1_computed}.")

    chk2_expected = int(l2[68])
    chk2_computed = compute_tle_checksum(l2)
    if chk2_expected != chk2_computed:
        raise ValueError(f"TLE Line 2 checksum failure: expected {chk2_expected}, computed {chk2_computed}.")

    # --- Parse Line 1 ---
    sat_num1 = int(l1[2:7].strip())
    classification = l1[7].strip()
    intl_desig = l1[9:17].strip()

    epoch_year_short = int(l1[18:20])
    epoch_year = 2000 + epoch_year_short if epoch_year_short < 57 else 1900 + epoch_year_short
    epoch_day = float(l1[20:32])

    # First derivative of mean motion (ballistic coefficient)
    mean_motion_dot = float(l1[33:43].strip())

    # Second derivative of mean motion (decimal with implied decimal point and exponent)
    n_ddot_str = l1[44:52].strip()
    if n_ddot_str and n_ddot_str != "00000-0" and n_ddot_str != "0":
        mantissa = float(n_ddot_str[:-2]) * 1e-5
        exp = int(n_ddot_str[-2:])
        mean_motion_ddot = mantissa * (10 ** exp)
    else:
        mean_motion_ddot = 0.0

    # BSTAR radiation pressure / atmospheric drag parameter
    bstar_str = l1[53:61].strip()
    if bstar_str and bstar_str != "00000-0" and bstar_str != "0":
        mantissa_b = float(bstar_str[:-2]) * 1e-5
        exp_b = int(bstar_str[-2:])
        bstar = mantissa_b * (10 ** exp_b)
    else:
        bstar = 0.0

    ephemeris_type = int(l1[62].strip() or "0")
    element_set_number = int(l1[64:68].strip() or "0")

    # --- Parse Line 2 ---
    sat_num2 = int(l2[2:7].strip())
    if sat_num1 != sat_num2:
        raise ValueError(f"Satellite number mismatch between Line 1 ({sat_num1}) and Line 2 ({sat_num2}).")

    inclination_deg = float(l2[8:16].strip())
    raan_deg = float(l2[17:25].strip())
    eccentricity = float("0." + l2[26:33].strip())
    arg_perigee_deg = float(l2[34:42].strip())
    mean_anomaly_deg = float(l2[43:51].strip())
    mean_motion_rev_day = float(l2[52:63].strip())
    rev_number_at_epoch = int(l2[63:68].strip() or "0")

    # Approximate semi-major axis from mean motion (rev/day)
    # n (rad/s) = mean_motion * 2*pi / 86400
    # mu_earth = 398600.4418 km^3/s^2
    # a = (mu / n^2)^(1/3)
    n_rad_s = mean_motion_rev_day * 2.0 * 3.141592653589793 / 86400.0
    if n_rad_s > 0:
        semi_major_axis_km = (398600.4418 / (n_rad_s ** 2)) ** (1.0 / 3.0)
        perigee_alt_km = semi_major_axis_km * (1.0 - eccentricity) - 6378.137
        apogee_alt_km = semi_major_axis_km * (1.0 + eccentricity) - 6378.137
    else:
        semi_major_axis_km = 0.0
        perigee_alt_km = 0.0
        apogee_alt_km = 0.0

    return {
        "satellite_number": sat_num1,
        "classification": classification,
        "international_designator": intl_desig,
        "epoch_year": epoch_year,
        "epoch_day": epoch_day,
        "mean_motion_dot": mean_motion_dot,
        "mean_motion_ddot": mean_motion_ddot,
        "bstar": bstar,
        "ephemeris_type": ephemeris_type,
        "element_set_number": element_set_number,
        "inclination_deg": inclination_deg,
        "raan_deg": raan_deg,
        "eccentricity": eccentricity,
        "arg_perigee_deg": arg_perigee_deg,
        "mean_anomaly_deg": mean_anomaly_deg,
        "mean_motion_rev_day": mean_motion_rev_day,
        "rev_number_at_epoch": rev_number_at_epoch,
        "semi_major_axis_km": round(semi_major_axis_km, 3),
        "perigee_altitude_km": round(perigee_alt_km, 3),
        "apogee_altitude_km": round(apogee_alt_km, 3),
    }
