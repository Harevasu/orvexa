"""Tests for TLE/OMM parsing, coordinate conversions, and orbital propagation."""

import math
import unittest

from orvexa.orbit.coordinate_utils import (
    eci_to_rtn_matrix,
    julian_date,
    rtn_to_eci_matrix,
    teme_to_geodetic,
)
from orvexa.orbit.omm_parser import parse_omm
from orvexa.orbit.propagation import propagate_orbit
from orvexa.orbit.tle_parser import compute_tle_checksum, parse_tle
from orvexa.orbit.validation import validate_orbital_elements, validate_tle_checksum


class TestOrbitModule(unittest.TestCase):
    """Test suite for orbital mechanics and coordinate routines."""

    def setUp(self):
        # Official Vallado SGP4 verification TLE (Vanguard 2 / Delta DEB, NORAD 06251)
        self.tle_line1 = "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985"
        self.tle_line2 = "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774"

    def test_tle_checksum_computation(self):
        self.assertEqual(compute_tle_checksum(self.tle_line1), 5)
        self.assertEqual(compute_tle_checksum(self.tle_line2), 4)
        self.assertTrue(validate_tle_checksum(self.tle_line1))
        self.assertTrue(validate_tle_checksum(self.tle_line2))

        # Corrupted line checksum
        corrupt_line1 = self.tle_line1[:68] + "9"
        self.assertFalse(validate_tle_checksum(corrupt_line1))

    def test_parse_tle_fields(self):
        elements = parse_tle(self.tle_line1, self.tle_line2)
        self.assertEqual(elements["satellite_number"], 6251)
        self.assertEqual(elements["classification"], "U")
        self.assertEqual(elements["epoch_year"], 2006)
        self.assertAlmostEqual(elements["epoch_day"], 176.82412014, places=4)
        self.assertAlmostEqual(elements["inclination_deg"], 58.0579, places=4)
        self.assertAlmostEqual(elements["eccentricity"], 0.0030035, places=7)
        self.assertAlmostEqual(elements["mean_motion_rev_day"], 15.56387291, places=5)
        # LEO satellite ~370-420 km altitude
        self.assertGreater(elements["perigee_altitude_km"], 350.0)
        self.assertLess(elements["apogee_altitude_km"], 450.0)

        is_valid, errors = validate_orbital_elements(elements)
        self.assertTrue(is_valid, f"Validation errors: {errors}")

    def test_julian_date_calculation(self):
        # J2000.0 epoch: 2000-01-01 12:00:00 UTC = JD 2451545.0
        jd_j2000 = julian_date(2000, 1, 1, 12, 0, 0.0)
        self.assertAlmostEqual(jd_j2000, 2451545.0, places=5)

    def test_teme_to_geodetic_bounds(self):
        # Subpoint at 7000 km along x-axis at J2000
        jd = 2451545.0
        lat, lon, alt = teme_to_geodetic(7000.0, 0.0, 0.0, jd)
        self.assertGreaterEqual(lat, -90.0)
        self.assertLessEqual(lat, 90.0)
        self.assertGreaterEqual(lon, -180.0)
        self.assertLessEqual(lon, 180.0)
        self.assertAlmostEqual(alt, 7000.0 - 6378.137, delta=10.0)

    def test_eci_rtn_transformation_orthogonality(self):
        pos = [7000.0, 0.0, 0.0]
        vel = [0.0, 7.5, 0.0]
        m_rtn = eci_to_rtn_matrix(pos, vel)
        m_eci = rtn_to_eci_matrix(pos, vel)

        # M_rtn * M_eci must equal Identity (3x3)
        for i in range(3):
            for j in range(3):
                dot = sum(m_rtn[i][k] * m_eci[k][j] for k in range(3))
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(dot, expected, places=5)

    def test_parse_omm_dictionary(self):
        omm_dict = {
            "OBJECT_NAME": "DELTA DEB",
            "NORAD_CAT_ID": "6251",
            "MEAN_MOTION": "15.5638",
            "ECCENTRICITY": "0.00300",
            "INCLINATION": "58.05",
            "RA_OF_ASC_NODE": "54.04",
            "ARG_OF_PERICENTER": "139.15",
            "MEAN_ANOMALY": "221.18",
        }
        parsed = parse_omm(omm_dict)
        self.assertEqual(parsed["object_name"], "DELTA DEB")
        self.assertEqual(parsed["object_id"], "6251")
        self.assertAlmostEqual(parsed["inclination_deg"], 58.05)

    def test_propagate_orbit_trajectory(self):
        start_jd = julian_date(2006, 6, 25, 12, 0, 0)
        end_jd = start_jd + (180.0 / 86400.0)  # 3 minutes
        traj = propagate_orbit(
            self.tle_line1, self.tle_line2, start_jd, end_jd, step_seconds=60.0
        )
        self.assertGreaterEqual(len(traj), 3)
        for point in traj:
            r_mag = math.sqrt(point["x_km"]**2 + point["y_km"]**2 + point["z_km"]**2)
            # Distance from Earth center for LEO satellite ~ 6700-6850 km
            self.assertGreater(r_mag, 6500.0)
            self.assertLess(r_mag, 7200.0)
            self.assertGreaterEqual(point["latitude_deg"], -90.0)
            self.assertLessEqual(point["latitude_deg"], 90.0)


if __name__ == "__main__":
    unittest.main()
