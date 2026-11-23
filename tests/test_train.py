"""Training on a synthetic year: splits, the exported bundle, evaluation tables, and offline/serving parity."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, _parse_datatype_string

from pipeline import checks, config, features, splits, train
from pipeline.config import get_spark
from tests.test_features import daily_frame

VERSION = "synthetic_v1"
TINY_GRID = [{"maxDepth": 2, "maxIter": 5, "stepSize": 0.3}, {"maxDepth": 3, "maxIter": 5, "stepSize": 0.3}]


def synthetic_daily(seed: int = 7, apps=("111", "222")) -> pd.DataFrame:
    """Games over the analysis window, ~60 negatives a day, shares that drift and spike."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(config.WINDOW_START, config.WINDOW_END, freq="D")
    rows = []
    for app in apps:
        n = rng.poisson(60, len(days))
        for k, cat in enumerate(config.CATEGORIES):
            share = 0.1 + 0.05 * np.sin(np.arange(len(days)) / (20 + 7 * k)) + 0.02 * k
            share[rng.random(len(days)) < 0.03] += 0.2  # occasional spikes
            c = rng.binomial(n, np.clip(share, 0, 1))
            rows += [dict(app_id=app, date=d.date().isoformat(), category=cat, category_count=int(ci),
                          negative_total=int(ni), share=ci / ni if ni else None, coverage_status="covered")
                     for d, ci, ni in zip(days, c, n)]
    return pd.DataFrame(rows)


def synthetic_features(apps=("111", "222")):
    return features.build(daily_frame(synthetic_daily(apps=apps)).withColumn("label_version", F.lit(VERSION)))


@pytest.fixture(scope="module")
def feats():
    return synthetic_features().cache()


@pytest.fixture(scope="module")
def trained(feats, tmp_path_factory):
    out = tmp_path_factory.mktemp("staging")
    (out / "feature_manifest.json").write_text(json.dumps(features.manifest(feats, VERSION)))
    manifest = train.run(feats, VERSION, out_dir=out, grid=TINY_GRID, score_test=True,
                         tracking_uri=(out / "mlruns").as_uri())
    return out, manifest


def test_split_boundaries():
    spark = get_spark("test")
    cases = {"2023-05-24": "train", "2023-05-25": "purged", "2023-05-31": "purged",
             "2023-06-01": "validation", "2023-08-08": "validation", "2023-08-09": "purged",
             "2023-08-15": "purged", "2023-08-16": "test", "2023-10-31": "test"}
    df = spark.createDataFrame([(d,) for d in cases], ["as_of_date"]).withColumn("as_of_date", F.to_date("as_of_date"))
    got = {r.as_of_date.isoformat(): r.split for r in df.withColumn("split", splits.split_col()).collect()}
    assert got == cases


def test_no_target_week_crosses_a_split(feats):
    rows = feats.withColumn("split", splits.split_col()).where(splits.usable())
    ends = {r.split: (r.first, r.last) for r in rows.groupBy("split").agg(
        F.min("as_of_date").alias("first"), F.max(F.date_add("as_of_date", 7)).alias("last")).collect()}
    assert ends["train"][1] <= date.fromisoformat(config.TRAIN_END) < ends["validation"][0]
    assert ends["validation"][1] <= date.fromisoformat(config.VAL_END) < ends["test"][0]


def test_manifest_matches_the_api_contract(trained, feats):
    out, m = trained
    fm = json.loads((out / "feature_manifest.json").read_text())
    assert m["format"] == "spark_pipeline"
    assert m["features"] == config.FEATURE_COLS == [f["name"] for f in fm["features"]]
    assert m["feature_types"] == {f["name"]: f["type"] for f in fm["features"]}
    assert (m["label_version"], m["feature_version"], m["snapshot_id"]) == \
        (fm["label_version"], fm["feature_version"], fm["snapshot_id"])
    assert m["training_label_end"] == m["replay_start"] == config.VAL_END  # refit on train + validation
    assert set(m["categories"]) == set(config.CATEGORIES)
    for c, v in m["categories"].items():
        assert (out / v["path"]).is_dir() and v["params"] in TINY_GRID
        assert v["validation_winner"] in {config.MODEL_VERSION, *train.BASELINES}
    assert json.loads((out / "model_manifest.json").read_text())["mlflow_run_id"] == m["mlflow_run_id"]


def test_scores_use_identical_rows(trained):
    scores = pd.read_csv(trained[0] / "model_scores.csv", dtype={"app_id": str})
    assert set(scores.split) == {"train", "validation", "test"}
    assert set(scores.method) == {config.MODEL_VERSION, *train.BASELINES}
    assert set(scores.app_id) == {"all", "111", "222"}
    per = scores[scores.category != "macro"].groupby(["split", "category", "app_id"]).rows.nunique()
    assert (per == 1).all()
    pooled = scores[(scores.app_id == "all") & (scores.category != "macro")]
    games = scores[(scores.app_id != "all")]
    assert pooled.groupby(["split", "method"]).rows.sum().equals(games.groupby(["split", "method"]).rows.sum())
    assert scores.mae_pp.between(0, 100).all() and scores.mae_pp_weighted.between(0, 100).all()
    macro = scores[scores.category == "macro"].set_index(["split", "method"]).mae_pp
    assert macro.loc[("validation", config.MODEL_VERSION)] == pytest.approx(
        pooled[(pooled.split == "validation") & (pooled.method == config.MODEL_VERSION)].mae_pp.mean())


def test_row_counts_explain_every_row(trained, feats):
    counts = pd.read_csv(trained[0] / "model_row_counts.csv")
    statuses = [c for c in counts.columns if c not in ("split", "category")]
    assert "scored" in statuses and "insufficient_history" in statuses
    assert counts.split.drop_duplicates().tolist() == ["train", "validation", "test", "purged"]
    assert all(pd.api.types.is_integer_dtype(counts[c]) for c in statuses)
    assert counts[statuses].to_numpy().sum() == feats.count()
    scores = pd.read_csv(trained[0] / "model_scores.csv", dtype={"app_id": str})
    val = scores[(scores.split == "validation") & (scores.app_id == "all") & (scores.category == "crash")]
    assert val.rows.iloc[0] == counts[(counts.split == "validation") & (counts.category == "crash")].scored.iloc[0]


def test_errors_and_model_card(trained):
    out, m = trained
    errors = pd.read_csv(out / "model_errors.csv")
    assert set(errors.split) == {"validation", "test"}
    assert (errors.groupby(["split", "category"]).size() <= train.TOP_ERRORS).all()
    assert (errors.abs_error_pp - 100 * (errors.predicted_share - errors.target_share).abs()).abs().max() < 1e-9
    card = (out / "model_card.md").read_text()
    for text in (m["mlflow_run_id"], VERSION, "Validation MAE", "Test MAE", "Limitations", config.VAL_END):
        assert text in card


def test_predictions_are_cutoff_safe_and_pass_checks(trained):
    out, m = trained
    spark = get_spark("test")
    preds = spark.read.parquet(str(out / "model_predictions"))
    assert preds.where(F.col("as_of_date") < F.col("training_label_end")).count() == 0
    assert preds.where(F.col("forecast_status") == "ok").count() > 0
    assert all(c.passed for c in checks.check_results(preds, "model_predictions"))
    assert checks.schema(preds, "model_predictions", config.FORECAST_RESULTS_COLS).passed
    fm = json.loads((out / "feature_manifest.json").read_text())
    bundle = checks.check_model_bundle(m, fm, preds, out)
    assert all(c.passed for c in bundle), [c for c in bundle if not c.passed]


def test_bundle_check_rejects_mismatches(trained):
    out, m = trained
    fm = json.loads((out / "feature_manifest.json").read_text())
    failed = {c.name for c in checks.check_model_bundle({**m, "label_version": "other"}, fm, None, out)
              if not c.passed}
    assert failed == {"matches_feature_manifest"}
    broken = {**m, "features": list(reversed(m["features"])),
              "categories": {**m["categories"], "crash": {"path": "model/nowhere"}}}
    failed = {c.name for c in checks.check_model_bundle(broken, fm, None, out) if not c.passed}
    assert failed == {"feature_order", "models_present"}


def test_parity_rows_match_a_row_built_like_the_api(trained, feats):
    """B -> C parity: the expected outputs equal a one-row prediction built the way the API builds it."""
    out, m = trained
    spark = get_spark("test")
    fm = json.loads((out / "feature_manifest.json").read_text())
    parity = pd.read_csv(out / "model_parity.csv", dtype={"app_id": str})
    assert set(parity.app_id) == {"111", "222"} and len(parity) == 2 * 3 * 4
    assert (parity.as_of_date >= m["training_label_end"]).all()
    source = feats.toPandas().set_index(["app_id", "as_of_date", "category"])
    for r in parity.itertuples():
        row = source.loc[(r.app_id, date.fromisoformat(r.as_of_date), r.category)]
        schema, values = StructType(), []
        for col in fm["features"]:
            schema.add(col["name"], _parse_datatype_string(col["type"]), True)
            v = row[col["name"]]
            values.append(None if pd.isna(v) else int(v) if col["type"] in {"int", "bigint"} else float(v))
        model = PipelineModel.load(str(out / m["categories"][r.category]["path"]))
        served = model.transform(spark.createDataFrame([tuple(values)], schema)).first().prediction
        assert min(1.0, max(0.0, served)) == pytest.approx(r.predicted_share, abs=m["parity"]["tolerance"])


def test_mlflow_has_the_run(trained):
    import mlflow

    out, m = trained
    client = mlflow.tracking.MlflowClient(tracking_uri=(out / "mlruns").as_uri())
    run = client.get_run(m["mlflow_run_id"])
    assert run.data.params["label_version"] == VERSION
    assert any(k.startswith("val_mae_pp.crash.") for k in run.data.metrics)
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"model_manifest.json", "model_card.md", "model_scores.csv", "model"} <= artifacts
    children = client.search_runs([run.info.experiment_id],
                                  f"tags.mlflow.parentRunId = '{run.info.run_id}'")
    assert len(children) == len(TINY_GRID) * len(config.CATEGORIES)


def test_c_api_serves_the_bundle(trained, feats):
    """Runs C's API (application/, once merged) on a publication holding this bundle: parity, cutoff, evidence."""
    import importlib.util
    import os
    import subprocess
    import sys

    if importlib.util.find_spec("application") is None:
        pytest.skip("application/ (C's API) isn't on this branch yet")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    out, m = trained
    feats.write.mode("overwrite").partitionBy("app_id").parquet(str(out / "forecast_features"))
    daily = daily_frame(synthetic_daily()).withColumn("label_version", F.lit(VERSION))
    daily.write.mode("overwrite").partitionBy("app_id").parquet(str(out / "daily_category"))
    ts = pd.Timestamp("2023-09-01T12:00:00Z")
    pd.DataFrame([dict(review_id="1", app_id="111", text_raw="crashes every match", text_normalized="crashes every match",
                       created_at=ts, updated_at=ts, available_at=ts, review_date=ts.date(), language="english",
                       voted_up=False, snapshot_id=config.SNAPSHOT_ID, cohort_policy=config.COHORT_POLICY)]
                 ).to_parquet(out / "reviews_clean")
    pd.DataFrame([dict(review_id="1", category="crash", is_match=True, rule_id="crash_word", evidence="crashes",
                       label_version=VERSION)]).to_parquet(out / "review_labels")
    script = """
import datetime, json, os, pandas as pd
from fastapi.testclient import TestClient
from application.api import app
root = os.environ["STEAMPULSE_DATA_DIR"]
m = json.load(open(os.path.join(root, "model_manifest.json")))
c = TestClient(app)
assert c.get("/health").json()["model_ready"]
parity = pd.read_csv(os.path.join(root, "model_parity.csv"), dtype={"app_id": str})
for (game, day), g in parity.groupby(["app_id", "as_of_date"]):
    r = c.post("/forecast", json=dict(app_id=game, as_of_date=day))
    assert r.status_code == 200, r.text
    got = {x["category"]: x for x in r.json()}
    for row in g.itertuples():
        assert abs(got[row.category]["predicted_share"] - row.predicted_share) <= m["parity"]["tolerance"]
        assert got[row.category]["model_version"] == m["model_version"]
before = (datetime.date.fromisoformat(m["training_label_end"]) - datetime.timedelta(days=1)).isoformat()
assert c.post("/forecast", json=dict(app_id="111", as_of_date=before)).status_code == 503
ev = c.get("/reviews", params=dict(app_id="111", category="crash", as_of_date="2023-09-02")).json()
assert [e["review_id"] for e in ev] == ["1"]
"""
    subprocess.run([sys.executable, "-c", script], cwd=config.ROOT, check=True,
                   env=dict(os.environ, STEAMPULSE_DATA_DIR=str(out)))
