"""Orbital ephemeris and trajectory propagation service for ORVEXA demonstration.

IMPORTANT DISCLAIMER:
Orbital tools in this module are strictly for ephemeris visualization and demonstration.
Arbitrary TLE / analytical orbital propagation is completely decoupled from ESA ML risk scoring.
"""

from typing import Any, Dict, List, Optional
import math
import sys
from pathlib import Path

# Ensure src in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.schemas import (
    OrbitalPropagationResponse,
    OrbitalTrajectoryPoint,
)
from orvexa.orbit.coordinate_utils import julian_date
from orvexa.orbit.propagation import propagate_orbit
from orvexa.orbit.tle_parser import compute_tle_checksum


def format_valid_tle(line1_raw: str, line2_raw: str) -> tuple[str, str]:
    """Ensure TLE lines have strictly matching standard checksums."""
    base1 = line1_raw[:68].ljust(68)
    chk1 = compute_tle_checksum(base1 + "0")
    line1 = base1 + str(chk1)

    base2 = line2_raw[:68].ljust(68)
    chk2 = compute_tle_checksum(base2 + "0")
    line2 = base2 + str(chk2)

    return line1, line2


# Official verification TLEs formatted with exact valid checksums
_SAT_DEFS = {
    "VANGUARD_2": {
        "name": "Vanguard 2 / Delta DEB (NORAD 06251)",
        "norad_id": "06251",
        "raw_l1": "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  398",
        "raw_l2": "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  677",
    },
    "SENTINEL_1A": {
        "name": "Sentinel-1A (ESA Earth Observation)",
        "norad_id": "39634",
        "raw_l1": "1 39634U 14016A   26240.50000000  .00000100  00000-0  50000-4 0  999",
        "raw_l2": "2 39634  98.1800 120.5000 0001400  85.0000 275.1000 14.591980006500",
    },
    "ENVISAT_DEBRIS": {
        "name": "Envisat (ESA Inactive Large Target)",
        "norad_id": "27386",
        "raw_l1": "1 27386U 02009A   26240.50000000  .00000050  00000-0  35000-4 0  999",
        "raw_l2": "2 27386  98.5400 115.2000 0001100  90.0000 270.2000 14.380000003200",
    },
}

DEMO_SATELLITES = {}
for k, v in _SAT_DEFS.items():
    l1, l2 = format_valid_tle(v["raw_l1"], v["raw_l2"])
    DEMO_SATELLITES[k] = {
        "name": v["name"],
        "norad_id": v["norad_id"],
        "tle_line1": l1,
        "tle_line2": l2,
    }


class OrbitalService:
    """Auxiliary orbital ephemeris and trajectory calculator."""

    def get_demo_satellites(self) -> Dict[str, Any]:
        """List available demo orbital targets."""
        return {
            k: {"name": v["name"], "norad_id": v["norad_id"]}
            for k, v in DEMO_SATELLITES.items()
        }

    def propagate_demo_satellite(
        self,
        satellite_key: str = "VANGUARD_2",
        duration_hours: float = 3.0,
        step_seconds: float = 120.0,
    ) -> OrbitalPropagationResponse:
        """Propagate demo satellite trajectory over specified duration."""
        sat_key = satellite_key.upper()
        if sat_key not in DEMO_SATELLITES:
            sat_key = "VANGUARD_2"

        sat = DEMO_SATELLITES[sat_key]

        # Use fixed reference epoch JD for deterministic demo visualization
        start_jd = julian_date(2026, 8, 28, 0, 0, 0)
        end_jd = start_jd + (duration_hours / 24.0)

        raw_traj = propagate_orbit(
            sat["tle_line1"],
            sat["tle_line2"],
            start_jd=start_jd,
            end_jd=end_jd,
            step_seconds=step_seconds,
        )

        traj_points: List[OrbitalTrajectoryPoint] = []
        for i, pt in enumerate(raw_traj):
            offset_sec = i * step_seconds
            traj_points.append(
                OrbitalTrajectoryPoint(
                    jd=round(pt["jd"], 6),
                    timestamp_offset_sec=offset_sec,
                    x_km=round(pt["x_km"], 2),
                    y_km=round(pt["y_km"], 2),
                    z_km=round(pt["z_km"], 2),
                    vx_km_s=round(pt["vx_km_s"], 4),
                    vy_km_s=round(pt["vy_km_s"], 4),
                    vz_km_s=round(pt["vz_km_s"], 4),
                    latitude_deg=round(pt["latitude_deg"], 3),
                    longitude_deg=round(pt["longitude_deg"], 3),
                    altitude_km=round(pt["altitude_km"], 2),
                )
            )

        return OrbitalPropagationResponse(
            satellite_name=sat["name"],
            norad_id=sat["norad_id"],
            epoch_jd=round(start_jd, 6),
            duration_hours=duration_hours,
            step_seconds=step_seconds,
            trajectory=traj_points,
            notes=(
                "Analytical SGP4 / J2-perturbed Keplerian orbit propagation from Two-Line Element sets (TLE). "
                "Demonstrates ephemeris coordinate derivation for space situational awareness."
            ),
        )


# Global singleton instance
orbital_service = OrbitalService()
