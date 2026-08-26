"""TLE and OMM input validation and orbital parameter sanity checks."""

from typing import Any, Dict, List, Tuple

from orvexa.orbit.tle_parser import compute_tle_checksum


def validate_tle_checksum(line: str) -> bool:
    """Verify standard modulo-10 TLE checksum for a 69-character line."""
    clean = line.strip()
    if len(clean) != 69:
        return False
    try:
        expected = int(clean[68])
        computed = compute_tle_checksum(clean)
        return expected == computed
    except (ValueError, IndexError):
        return False


def validate_orbital_elements(elements: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Verify physical sanity of orbital parameters (eccentricity, inclination, altitude)."""
    errors = []

    ecc = elements.get("eccentricity", 0.0)
    if ecc < 0.0 or ecc >= 1.0:
        errors.append(f"Invalid eccentricity {ecc}: must be in [0.0, 1.0) for closed Earth orbits.")

    inc = elements.get("inclination_deg", 0.0)
    if inc < 0.0 or inc > 180.0:
        errors.append(f"Invalid inclination {inc} deg: must be in [0.0, 180.0].")

    mm = elements.get("mean_motion_rev_day", 0.0)
    if mm <= 0.0 or mm > 20.0:
        errors.append(f"Unrealistic mean motion {mm} rev/day for Earth satellites.")

    sma = elements.get("semi_major_axis_km", 0.0)
    if sma < 6400.0:
        errors.append(f"Semi-major axis {sma} km is below Earth radius.")

    return len(errors) == 0, errors
