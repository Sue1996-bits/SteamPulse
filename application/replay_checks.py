import csv
import unittest
from datetime import date

from fastapi.testclient import TestClient

from application.api import ROOT, app


class ReplayTests(unittest.TestCase):
    client = TestClient(app)

    def test_shared_oracle(self):
        with (ROOT / "expected_windows.csv").open() as stream:
            expected = list(csv.DictReader(stream))
        for method in ("persistence", "rolling28"):
            for game in ("888", "999"):
                payload = dict(app_id=game, as_of_date="2023-02-04", method=method)
                response = self.client.post("/forecast", json=payload)
                self.assertEqual(response.status_code, 200)
                by_cat = {r["category"]: r for r in response.json()}
                self.assertEqual(len(by_cat), 4)
                for row in (r for r in expected if r["app_id"] == game):
                    got = by_cat[row["category"]]
                    self.assertEqual(got["observed_status"], row["observed_status"])
                    self.assertEqual(got["eligibility_reason"], row["eligibility_reason"] or None)
                    value = row[method+"_pred"]
                    if value:
                        self.assertAlmostEqual(got["predicted_share"], float(value))
                    else:
                        self.assertIsNone(got["predicted_share"])

    def test_states(self):
        payload = dict(app_id="999", as_of_date="2023-02-04")
        for extra, status in [({}, 503), ({"method": "best_validation"}, 503),
                              ({"app_id": "missing"}, 404), ({"horizon_days": 3}, 422)]:
            self.assertEqual(self.client.post("/forecast", json=payload | extra).status_code, status)
        self.assertFalse(self.client.get("/health").json()["model_ready"])
        games = {g["app_id"]: g for g in self.client.get("/games").json()}
        self.assertEqual(games["999"]["eligible_dates"], ["2023-02-04"])
        self.assertEqual(games["888"]["eligible_dates"], [])

    def test_cutoff_and_gap(self):
        rows = self.client.get("/reviews", params=dict(app_id="999", category="crash",
                               as_of_date="2023-02-03")).json()
        self.assertTrue(rows)
        self.assertTrue(all(date.fromisoformat(r["available_at"][:10]) <= date(2023, 2, 3) for r in rows))
        rows = self.client.get("/history", params=dict(app_id="888", category="crash",
                               start_date="2023-01-20", end_date="2023-01-25")).json()
        self.assertIsNone(rows[0]["negative_total"])
        self.assertEqual(rows[-1]["negative_total"], 0)
