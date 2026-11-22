"""forecast_features on the shared fixture, checked against its hand-computed windows."""
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql import functions as F

from pipeline import config, features
from pipeline.config import get_spark

FIX = Path(__file__).parent / "fixtures"
LABEL_VERSION = "fixture_v1"


def daily_frame(pdf: pd.DataFrame):
    spark = get_spark("test")
    pdf = pdf.assign(label_version=LABEL_VERSION, cohort_version=config.COHORT_VERSION)
    pdf = pdf.astype({"category_count": "float64", "negative_total": "float64"})
    pdf = pdf.astype(object).where(pdf.notna(), None)  # NaN -> null in Spark
    return (spark.createDataFrame(pdf)
            .withColumn("date", F.to_date("date"))
            .withColumn("category_count", F.col("category_count").cast("long"))
            .withColumn("negative_total", F.col("negative_total").cast("long")))


@pytest.fixture(scope="module")
def daily_pdf():
    return pd.read_csv(FIX / "expected_daily_category.csv", dtype={"app_id": str})


@pytest.fixture(scope="module")
def feats(daily_pdf):
    out = features.build(daily_frame(daily_pdf)).toPandas()
    out["as_of_date"] = out["as_of_date"].astype(str)
    return out.set_index(["app_id", "as_of_date", "category"])


def test_columns_and_label_kept_apart(feats):
    assert list(feats.reset_index().columns) == config.FORECAST_FEATURES_COLS
    assert not set(config.TARGET_COLS) & set(config.FEATURE_COLS)
    assert set(config.FEATURE_COLS) == set(features.DESCRIPTIONS)


def test_matches_hand_computed_windows(feats):
    exp = pd.read_csv(FIX / "expected_windows.csv", dtype={"app_id": str})
    for w in exp.itertuples():
        f = feats.loc[(w.app_id, w.as_of_date, w.category)]
        assert (f.c_7d, f.n_7d) == (w.C7, w.N7)
        assert f.share_7d == pytest.approx(w.recent_share)
        if pd.isna(w.eligibility_reason):
            assert f.eligibility == "eligible"
            assert f.baseline_share == pytest.approx(w.baseline_share)
            assert f.lift_pp == pytest.approx(w.observed_lift_pp)
            assert f.share_28d == pytest.approx(w.rolling28_pred)
        else:
            assert f.eligibility == w.eligibility_reason
            assert pd.isna(f.baseline_share) and pd.isna(f.lift_pp)


def test_target_is_the_next_seven_days(feats):
    f = feats.loc[("999", "2023-01-28", "crash")]
    assert f.target_share_next7 == pytest.approx(26 / 100) and f.target_n_next7 == 100
    last = feats.loc[("999", "2023-02-04", "crash")]
    assert pd.isna(last.target_share_next7)  # no future data yet


def test_short_history_is_flagged(feats):
    assert feats.loc[("999", "2023-02-03", "crash")].eligibility == "insufficient_history"
    assert feats.loc[("999", "2023-02-04", "crash")].eligibility == "eligible"


def test_variation_and_weekday(feats):
    f = feats.loc[("999", "2023-02-04", "monetization")]
    assert (f.count_std_28d, f.zero_day_rate_28d) == (0.0, 1.0)
    assert f.weekday == 6  # 2023-02-04 was a Saturday
    crash = feats.loc[("999", "2023-02-04", "crash")]
    assert crash.zero_day_rate_28d == 0.0 and crash.count_std_28d > 0


def test_future_rows_do_not_change_features(daily_pdf):
    """Leakage check: rewriting every day after t must leave the features at t unchanged."""
    t = "2023-01-28"
    changed = daily_pdf.copy()
    later = changed["date"] > t
    changed.loc[later, "category_count"] = 0
    changed.loc[later, "negative_total"] = 999
    changed.loc[later, "share"] = 0.0

    def at_t(pdf):
        out = features.build(daily_frame(pdf)).where(F.col("as_of_date") == t).toPandas()
        return out.sort_values(["app_id", "category"]).reset_index(drop=True)

    before, after = at_t(daily_pdf), at_t(changed)
    pd.testing.assert_frame_equal(before[config.FEATURE_COLS], after[config.FEATURE_COLS])
    assert not before[config.LABEL_COL].equals(after[config.LABEL_COL])  # only the target moves
