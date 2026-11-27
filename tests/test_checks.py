"""Quality checks: the fixture pipeline passes, and each kind of broken table is caught."""
import json

import pytest
from pyspark.sql import functions as F

from pipeline import aggregate, baselines, checks, features
from pipeline.config import get_spark
from tests.test_aggregate import END, FIX, LABEL_VERSION, read


@pytest.fixture(scope="module")
def tables():
    spark = get_spark("test")
    clean = (spark.createDataFrame(read("reviews_clean.csv"))
             .withColumn("review_date", F.to_date("review_date")))
    labels = aggregate.read_labels(spark, FIX / "review_labels.csv")
    neg = aggregate.daily_negative(clean, END).cache()
    cat = aggregate.daily_category(clean, labels, neg, LABEL_VERSION).cache()
    feats = features.build(cat).cache()
    return {"reviews_clean": clean, "daily_negative": neg, "daily_category": cat,
            "forecast_features": feats, "forecast_results": baselines.build(feats)}


def summary_for(tables) -> dict:
    neg = tables["daily_negative"].groupBy("app_id").agg(F.sum("negative_total").alias("n")).collect()
    rows = {r.app_id: r["count"] for r in tables["reviews_clean"].groupBy("app_id").count().collect()}
    return {"games": {r.app_id: {"reviews_clean_rows": rows[r.app_id], "english_negatives": r.n} for r in neg}}


def failed(results) -> set[tuple[str, str]]:
    return {(c.table, c.name) for c in results if not c.passed}


def test_fixture_pipeline_passes_every_check(tables):
    results = checks.run_checks(tables, summary_for(tables))
    assert failed(results) == set()
    assert len({c.table for c in results}) == 5


def test_duplicate_key_is_caught(tables):
    neg = tables["daily_negative"]
    broken = {"daily_negative": neg.unionByName(neg.limit(1))}
    assert ("daily_negative", "unique_key") in failed(checks.run_checks(broken))


def test_zero_on_a_gap_day_is_caught(tables):
    neg = tables["daily_negative"].withColumn(
        "negative_total", F.coalesce("negative_total", F.lit(0)))  # the mistake: gap -> 0
    assert ("daily_negative", "gap_is_null") in failed(checks.run_checks({"daily_negative": neg}))


def test_missing_day_is_caught(tables):
    neg = tables["daily_negative"].where(F.col("date") != F.lit("2023-01-10").cast("date"))
    assert ("daily_negative", "full_grid") in failed(checks.run_checks({"daily_negative": neg}))


def test_bad_status_and_share_are_caught(tables):
    res = (tables["forecast_results"]
           .withColumn("observed_status", F.lit("spiking"))
           .withColumn("predicted_share", F.when(F.col("forecast_status") == "ok", F.lit(1.5))))
    got = failed(checks.run_checks({"forecast_results": res}))
    assert {("forecast_results", "observed_status_allowed"), ("forecast_results", "shares_in_0_1")} <= got


def test_counts_that_disagree_with_the_summary_are_caught(tables):
    summary = summary_for(tables)
    summary["games"]["999"]["english_negatives"] += 1
    got = failed(checks.run_checks(tables, summary))
    assert got == {("daily_negative", "matches_clean_summary")}


def test_publish_copies_tables_and_writes_manifest(tmp_path, monkeypatch, tables):
    staging, published = tmp_path / "staging", tmp_path / "published"
    tables["daily_negative"].write.partitionBy("app_id").parquet(str(staging / "daily_negative"))
    monkeypatch.setattr(checks.config, "STAGING_DIR", staging)
    monkeypatch.setattr(checks.config, "PUBLISH_DIR", published)
    checks.publish(["daily_negative"], "fixture_v1")
    assert not list((published / "daily_negative").rglob("*.crc"))
    m = json.loads((published / "manifest.json").read_text())
    assert m["versions"]["label_version"] == "fixture_v1"
    assert m["outputs"]["daily_negative"]["rows"] == 70


def test_publish_keeps_one_label_version_and_no_leftovers(tmp_path, monkeypatch):
    spark = get_spark("test")
    staging, published = tmp_path / "staging", tmp_path / "published"
    rows = [("1", "crash", True, "crash_word", "crash", "old_v"),
            ("1", "crash", True, "crash_word", "crash", "new_v"),
            ("2", "performance", True, "perf_fps", "low fps", "new_v")]
    labels = spark.createDataFrame(rows, ["review_id", "category", "is_match", "rule_id", "evidence", "label_version"])
    labels.write.partitionBy("label_version").parquet(str(staging / "review_labels"))
    (staging / "baseline_scores.csv").write_text("method_version,mae\n")
    (staging / "clean_summary.json").write_text("{}")
    published.mkdir()
    (published / "stale.csv").write_text("from an earlier run")
    monkeypatch.setattr(checks.config, "STAGING_DIR", staging)
    monkeypatch.setattr(checks.config, "PUBLISH_DIR", published)

    checks.publish(["review_labels"], "new_v")

    assert [p.name for p in (published / "review_labels").iterdir()] == ["label_version=new_v"]
    m = json.loads((published / "manifest.json").read_text())
    assert m["outputs"]["review_labels"]["rows"] == 2
    assert not (published / "stale.csv").exists()
    assert not (published / "baseline_scores.csv").exists()
    assert (published / "clean_summary.json").exists()
