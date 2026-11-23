"""Exercise the production readers with an isolated synthetic Parquet publication."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


class PublicationChecks(unittest.TestCase):
    def test_published_replay(self):
        source = Path(__file__).resolve().parents[1] / "tests/fixtures"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, dest in (("expected_daily_category", "daily_category"),
                               ("reviews_clean", "reviews_clean"), ("review_labels", "review_labels")):
                df = pd.read_csv(source / (name + ".csv"))
                if dest == "daily_category":
                    df["label_version"], df["cohort_version"] = "fixture_v1", "v1"
                df.to_parquet(root / dest)
            windows = pd.read_csv(source / "expected_windows.csv").rename(columns={
                "recent_share": "share_7d", "rolling28_pred": "share_28d",
                "observed_lift_pp": "lift_pp", "C7": "c_7d"})
            windows["feature_max_date"] = windows.as_of_date
            windows["eligibility"] = windows.eligibility_reason.fillna("eligible")
            windows["source_snapshot"], windows["label_version"], windows["feature_version"] = "fixture", "fixture_v1", "v1"
            windows.to_parquet(root / "forecast_features")
            versions = dict(feature_version="v1", label_version="fixture_v1", snapshot_id="fixture")
            (root / "feature_manifest.json").write_text(json.dumps(dict(
                **versions, features=[{"name": "share_7d"}], label="target", target_columns=["target"])))
            (root / "model_manifest.json").write_text(json.dumps(dict(
                **versions, features=["share_7d"], format="linear", model_version="synthetic_test",
                training_label_end="2023-02-04", categories={
                    c: {"intercept": 0.1, "weights": [0.5]} for c in windows.category.unique()})))
            subprocess.run([sys.executable, "-c", """
from fastapi.testclient import TestClient
from application.api import app
c = TestClient(app)
def forecast(day, game="999"):
    return c.post("/forecast", json=dict(app_id=game, as_of_date=day))
assert c.get("/health").json()["model_ready"]
assert len(c.get("/games").json()) == 2
assert forecast("2023-02-03").status_code == 503
rows = forecast("2023-02-04").json()
assert len(rows) == 4
assert abs(rows[0]["predicted_share"] - 0.23) < 1e-12
assert rows[0]["model_version"] == "synthetic_test"
assert forecast("2023-02-04", "888").json()[0]["predicted_share"] is None
assert c.get("/reviews", params=dict(app_id="999", as_of_date="2023-02-03", category="crash")).json()
"""], env=dict(os.environ, STEAMPULSE_DATA_DIR=temp), check=True)
