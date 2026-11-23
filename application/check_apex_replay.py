"""Run against a live API and Wren's matching export: python -m application.check_apex_replay DATA_DIR"""
import argparse
import json
import math
from pathlib import Path

import httpx
import pandas as pd

GAME = "1172470"
DAY = "2023-08-27"
CATEGORIES = {"crash", "performance", "match_server", "monetization"}


def compare(rows, expected, manifest):
    if len(rows) != 4 or len(expected) != 4 or set(expected.category) != CATEGORIES:
        raise ValueError("Expected four unique offline and API categories")
    if {row["category"] for row in rows} != CATEGORIES:
        raise ValueError("Missing or duplicate API category")
    reference = expected.set_index("category")
    tolerance = float(manifest["parity"]["tolerance"])
    for row in rows:
        category = row["category"]
        wanted = dict(app_id=GAME, as_of_date=DAY, forecast_start="2023-08-28", forecast_end="2023-09-03",
                      label_version="rules_v2", prediction_method="trained_model", forecast_status="ok",
                      model_version=manifest["model_version"], source_snapshot=manifest["snapshot_id"],
                      training_label_end=manifest["training_label_end"], feature_version=manifest["feature_version"])
        for key, value in wanted.items():
            if row[key] != value:
                raise ValueError(f"{category}: unexpected {key}: {row[key]!r}")
        offline = reference.loc[category]
        prediction = row["predicted_share"]
        if offline.model_version != row["model_version"] or not 0 <= prediction <= 1 or not math.isclose(
            prediction, float(offline.predicted_share), rel_tol=0, abs_tol=tolerance
        ):
            raise ValueError(f"{category}: offline/API prediction mismatch")
        if category == "match_server" and row["observed_status"] != "emerging":
            raise ValueError("The expected match_server alert did not reproduce")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "model_manifest.json").read_text())
    if manifest["training_label_end"] > DAY or manifest["label_version"] != "rules_v2":
        raise ValueError("Model unavailable for this replay date or label version")
    frame = pd.read_parquet(args.data_dir / "model_predictions",
                            columns=["app_id", "as_of_date", "category", "model_version", "predicted_share"])
    expected = frame[(frame.app_id.astype(str) == GAME) & (frame.as_of_date.astype(str) == DAY)]
    response = httpx.post(args.url.rstrip("/") + "/forecast", json=dict(app_id=GAME, as_of_date=DAY,
                          method="trained_model"), timeout=180, trust_env=False)
    response.raise_for_status()
    rows = response.json()
    compare(rows, expected, manifest)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
