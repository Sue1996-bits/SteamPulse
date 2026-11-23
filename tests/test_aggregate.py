"""Run the daily aggregation on the shared fixture and compare with its expected tables."""
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import functions as F

from pipeline import aggregate, config
from pipeline.config import get_spark

FIX = Path(__file__).parent / "fixtures"
END = "2023-02-04"  # the fixture's last day
LABEL_VERSION = "fixture_v1"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name, dtype={"app_id": str, "review_id": str})


@pytest.fixture(scope="module")
def tables():
    spark = get_spark("test")
    clean = (spark.createDataFrame(read("reviews_clean.csv"))
             .withColumn("review_date", F.to_date("review_date")))
    labels = aggregate.read_labels(spark, FIX / "review_labels.csv")
    neg = aggregate.daily_negative(clean, END)
    cat = aggregate.daily_category(clean, labels, neg, LABEL_VERSION)
    return neg.toPandas(), cat.toPandas()


def as_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["date"].astype(str)
    return df


def test_daily_negative_matches_fixture(tables):
    got = as_strings(tables[0])
    assert list(got.columns) == config.DAILY_NEGATIVE_COLS
    exp = read("expected_daily_negative.csv")
    m = exp.merge(got, on=["app_id", "date"], how="outer", suffixes=("_exp", "_got"), indicator=True)
    assert (m._merge == "both").all(), "grid rows differ"
    assert (m.coverage_status_exp == m.coverage_status_got).all()
    assert m.negative_total_exp.astype("Float64").equals(m.negative_total_got.astype("Float64"))


def test_daily_category_matches_fixture(tables):
    got = as_strings(tables[1])
    assert list(got.columns) == config.DAILY_CATEGORY_COLS
    assert (got.label_version == LABEL_VERSION).all()
    exp = read("expected_daily_category.csv")
    key = ["app_id", "date", "category"]
    m = exp.merge(got, on=key, how="outer", suffixes=("_exp", "_got"), indicator=True)
    assert (m._merge == "both").all(), "grid rows differ"
    for col in ["category_count", "negative_total"]:
        assert m[f"{col}_exp"].astype("Float64").equals(m[f"{col}_got"].astype("Float64")), col
    assert (m.coverage_status_exp == m.coverage_status_got).all()
    both_null = m.share_exp.isna() & m.share_got.isna()
    assert (both_null | ((m.share_exp - m.share_got).abs() < 1e-12)).all()


def test_other_label_versions_are_ignored(tables):
    spark = get_spark("test")
    clean = (spark.createDataFrame(read("reviews_clean.csv"))
             .withColumn("review_date", F.to_date("review_date")))
    labels = aggregate.read_labels(spark, FIX / "review_labels.csv")
    neg = aggregate.daily_negative(clean, END)
    cat = aggregate.daily_category(clean, labels, neg, "some_other_version").toPandas()
    covered = cat[cat.coverage_status == "covered"]
    assert (covered.category_count == 0).all()
