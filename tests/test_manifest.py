import json

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import manifest


def test_manifest_counts_rows_in_partitioned_output(tmp_path):
    table_dir = tmp_path / "daily_negative"
    for app_id, n in (("10", 3), ("20", 2)):
        part = table_dir / f"app_id={app_id}"
        part.mkdir(parents=True)
        pq.write_table(pa.table({"negative_total": list(range(n))}), part / "part-0.parquet")

    path = manifest.write_manifest({"daily_negative": table_dir}, out_dir=tmp_path, run_id="test")
    m = json.loads(path.read_text())

    assert m["run_id"] == "test"
    assert m["outputs"]["daily_negative"]["rows"] == 5
    assert m["outputs"]["daily_negative"]["files"] == 2
    assert m["source"]["dataset"] == "adde023/steam-reviews"
    assert set(m["versions"]) == {"cohort_policy", "cohort_version", "label_version", "feature_version"}
    assert m["runtime"]["pyspark"] == "3.5.8"
    assert m["runtime"]["uv_lock_sha256"]


def test_discover_outputs_handles_missing_dir(tmp_path):
    assert manifest.discover_outputs(tmp_path / "nope") == {}
