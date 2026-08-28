"""Unit and integration tests for ORVEXA FastAPI backend API."""

import json
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from backend.main import app
from orvexa.splitting import Phase5SplitManifest


class TestBackendAPI(unittest.TestCase):
    """Integration test suite for the ORVEXA backend API."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.workspace = Path(__file__).resolve().parent.parent
        split_manifest_path = cls.workspace / "artifacts" / "splits" / "phase5" / "phase5_split_manifest.json"
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            cls.split_manifest = json.load(f)

    def test_health_check(self):
        """Verify GET /api/health returns 200 and healthy status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("Candidate C", data["governance_candidate"])
        self.assertEqual(data["supported_horizons"], ["H2", "H3", "H5", "H6"])

    def test_freeze_manifest(self):
        """Verify GET /api/manifest returns valid Candidate C manifest."""
        response = self.client.get("/api/manifest")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["manifest_type"], "ORVEXA_PHASE5_CANDIDATE_FREEZE_MANIFEST")
        self.assertEqual(data["selected_candidate"]["candidate_id"], "Candidate_C_QuantileM4_CQR")

    def test_benchmarks_endpoint(self):
        """Verify GET /api/benchmarks returns exact Phase 5 internal test metrics."""
        response = self.client.get("/api/benchmarks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["horizons"]), 4)

        horizons_by_name = {h["horizon"]: h for h in data["horizons"]}
        self.assertIn("H2", horizons_by_name)
        self.assertIn("H3", horizons_by_name)
        self.assertIn("H5", horizons_by_name)
        self.assertIn("H6", horizons_by_name)

        # Check H2 values
        h2 = horizons_by_name["H2"]
        self.assertEqual(h2["n_test_samples"], 1528)
        self.assertAlmostEqual(h2["q50_r2"], 0.58497, places=4)
        self.assertAlmostEqual(h2["cqr_90pct_coverage"], 92.15, places=1)

        # Check H6 negative R² is present and explained
        h6 = horizons_by_name["H6"]
        self.assertEqual(h6["n_test_samples"], 1071)
        self.assertAlmostEqual(h6["q50_r2"], -0.16651, places=4)
        self.assertAlmostEqual(h6["cqr_90pct_coverage"], 90.20, places=1)
        self.assertIn("Negative R²", h6["point_prediction_notes"])

    def test_events_catalog_and_filtering(self):
        """Verify GET /api/events returns paginated allowed events."""
        response = self.client.get("/api/events?page=1&page_size=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["total_count"], 2000)
        self.assertEqual(len(data["events"]), 10)

        # Check search filter
        ev_id = data["events"][0]["event_id"]
        search_res = self.client.get(f"/api/events?search={ev_id}")
        self.assertEqual(search_res.status_code, 200)
        search_data = search_res.json()
        self.assertGreaterEqual(search_data["total_count"], 1)

    def test_event_detail_endpoint(self):
        """Verify GET /api/events/{event_id} returns detailed CDM sequence."""
        # Pick an allowed validation event
        val_id = self.split_manifest["val_event_ids"][0]
        response = self.client.get(f"/api/events/{val_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["event_id"], str(val_id))
        self.assertGreater(len(data["cdms"]), 0)
        self.assertIn("miss_distance", data["cdms"][0])
        self.assertIn("time_to_tca", data["cdms"][0])

    def test_quarantined_test_event_rejection(self):
        """Verify quarantined internal test event ID is strictly rejected with 403 Forbidden."""
        test_id = self.split_manifest["test_event_ids"][0]
        
        # Test event detail endpoint
        res_detail = self.client.get(f"/api/events/{test_id}")
        self.assertEqual(res_detail.status_code, 403)
        self.assertIn("Quarantine violation", res_detail.json()["detail"])

        # Test inference endpoint
        res_inf = self.client.post("/api/inference", json={"event_id": str(test_id), "horizon": "H2"})
        self.assertEqual(res_inf.status_code, 403)
        self.assertIn("Quarantine violation", res_inf.json()["detail"])

    def test_live_inference_execution(self):
        """Verify live Candidate C inference runs correctly on allowed validation event."""
        # Pick event 6709 (known to qualify across horizons)
        val_id = "6709"
        response = self.client.post(
            "/api/inference",
            json={"event_id": val_id, "horizon": "H2", "alpha": 0.10},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["event_id"], val_id)
        self.assertEqual(data["horizon"], "H2")
        self.assertEqual(data["lead_time_hours"], 48.0)
        self.assertGreater(data["qualifying_cdms_count"], 0)

        # Check quantiles monotonicity
        q = data["quantiles"]
        self.assertLessEqual(q["q05"], q["q10"])
        self.assertLessEqual(q["q10"], q["q25"])
        self.assertLessEqual(q["q25"], q["q50"])
        self.assertLessEqual(q["q50"], q["q75"])
        self.assertLessEqual(q["q75"], q["q90"])
        self.assertLessEqual(q["q90"], q["q95"])

        # Check CQR interval
        cqr = data["cqr_interval"]
        self.assertLess(cqr["lower"], cqr["upper"])
        self.assertEqual(cqr["confidence_level"], 0.90)
        self.assertGreater(cqr["conformal_shift_qhat"], 0.0)

        # Check risk classification
        self.assertIn(data["risk_assessment"]["level"], ["CRITICAL / HIGH RISK", "MODERATE RISK", "LOW RISK", "NEGLIGIBLE / SAFE"])

    def test_multi_horizon_inference(self):
        """Verify GET /api/inference/{event_id}/multi-horizon returns multi-horizon dictionary."""
        val_id = "6709"
        response = self.client.get(f"/api/inference/{val_id}/multi-horizon")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["event_id"], val_id)
        self.assertIn("H2", data["horizons"])
        self.assertIn("H3", data["horizons"])

    def test_ranked_alerts_endpoint(self):
        """Verify GET /api/ranked_alerts returns sorted queue."""
        response = self.client.get("/api/ranked_alerts?horizon=H2&limit=20")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["horizon"], "H2")
        self.assertEqual(len(data["alerts"]), 20)
        self.assertEqual(data["alerts"][0]["rank"], 1)

    def test_orbital_propagation_endpoint(self):
        """Verify GET /api/orbital/propagate returns calculated trajectory points."""
        response = self.client.get("/api/orbital/propagate?satellite=SENTINEL_1A&duration_hours=2.0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["satellite_name"], "Sentinel-1A (ESA Earth Observation)")
        self.assertGreater(len(data["trajectory"]), 10)
        pt0 = data["trajectory"][0]
        self.assertIn("x_km", pt0)
        self.assertIn("y_km", pt0)
        self.assertIn("z_km", pt0)
        self.assertIn("latitude_deg", pt0)
        self.assertIn("longitude_deg", pt0)
