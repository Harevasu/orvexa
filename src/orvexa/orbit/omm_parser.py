"""Orbit Mean-Elements Message (OMM) JSON/XML/Dict parser.

References:
- CCSDS 502.0-B-2: Orbit Data Messages standard.
- SGP4/OMM structure from reference_repo/python-sgp4 (sgp4/omm.py) and reference_repo/satkit (src/omm/).
"""

from typing import Any, Dict


def parse_omm(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse CCSDS Orbit Mean-Elements Message (OMM) structure into standard orbital state.
    
    Args:
        data: Dictionary representation of an OMM record.
        
    Returns:
        Standardized dictionary containing satellite ID, epoch, and orbital parameters.
    """
    # Support both flat and nested (e.g. CCSDS standard OMM JSON) structures
    obj_data = data.get("OMM", data)
    metadata = obj_data.get("metadata", obj_data.get("METADATA", {}))
    data_block = obj_data.get("data", obj_data.get("DATA", obj_data))
    mean_elem = data_block.get("mean_elements", data_block.get("MEAN_ELEMENTS", data_block))
    tle_params = data_block.get("tle_parameters", data_block.get("TLE_PARAMETERS", data_block))

    sat_name = str(metadata.get("OBJECT_NAME", obj_data.get("OBJECT_NAME", "UNKNOWN")))
    sat_id = str(metadata.get("OBJECT_ID", obj_data.get("NORAD_CAT_ID", "0")))
    center_name = str(metadata.get("CENTER_NAME", "EARTH"))
    ref_frame = str(metadata.get("REF_FRAME", "TEME"))

    epoch = str(data_block.get("EPOCH", mean_elem.get("EPOCH", "")))
    mean_motion = float(mean_elem.get("MEAN_MOTION", 0.0))
    eccentricity = float(mean_elem.get("ECCENTRICITY", 0.0))
    inclination = float(mean_elem.get("INCLINATION", 0.0))
    raan = float(mean_elem.get("RA_OF_ASC_NODE", 0.0))
    arg_perigee = float(mean_elem.get("ARG_OF_PERICENTER", 0.0))
    mean_anomaly = float(mean_elem.get("MEAN_ANOMALY", 0.0))

    bstar = float(tle_params.get("BSTAR", 0.0))
    mean_motion_dot = float(tle_params.get("MEAN_MOTION_DOT", 0.0))
    mean_motion_ddot = float(tle_params.get("MEAN_MOTION_DDOT", 0.0))

    # Approximate semi-major axis (km)
    n_rad_s = mean_motion * 2.0 * 3.141592653589793 / 86400.0
    if n_rad_s > 0:
        semi_major_axis_km = (398600.4418 / (n_rad_s ** 2)) ** (1.0 / 3.0)
    else:
        semi_major_axis_km = 0.0

    return {
        "object_name": sat_name,
        "object_id": sat_id,
        "center_name": center_name,
        "reference_frame": ref_frame,
        "epoch": epoch,
        "mean_motion_rev_day": mean_motion,
        "eccentricity": eccentricity,
        "inclination_deg": inclination,
        "raan_deg": raan,
        "arg_perigee_deg": arg_perigee,
        "mean_anomaly_deg": mean_anomaly,
        "bstar": bstar,
        "mean_motion_dot": mean_motion_dot,
        "mean_motion_ddot": mean_motion_ddot,
        "semi_major_axis_km": round(semi_major_axis_km, 3),
    }
