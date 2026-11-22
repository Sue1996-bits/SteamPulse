"""Baselines on the shared fixture, checked against its expected windows."""
import pandas as pd
import pytest

from pipeline import baselines, config, features
from tests.test_features import FIX, daily_frame


@pytest.fixture(scope="module")
def frames():
    daily = pd.read_csv(FIX / "expected_daily_category.csv", dtype={"app_id": str})
    feats = features.build(daily_frame(daily)).cache()
    return feats, baselines.build(feats).cache()


@pytest.fixture(scope="module")
def results(frames):
    out = frames[1].toPandas()
    out["as_of_date"] = out["as_of_date"].astype(str)
    return out.set_index(["app_id", "as_of_date", "category", "method_version"])


def test_columns_and_methods(results):
    assert list(results.reset_index().columns) == config.FORECAST_RESULTS_COLS
    assert set(results.index.get_level_values("method_version")) == set(baselines.METHODS)
    assert set(results.observed_status) <= set(config.OBSERVED_STATUSES)
    assert set(results.forecast_status) <= set(config.FORECAST_STATUSES)
    assert not results.index.duplicated().any()


def test_matches_expected_windows(results):
    exp = pd.read_csv(FIX / "expected_windows.csv", dtype={"app_id": str})
    for w in exp.itertuples():
        for method, col in (("persistence_v1", "persistence_pred"), ("rolling28_v1", "rolling28_pred")):
            r = results.loc[(w.app_id, w.as_of_date, w.category, method)]
            assert r.observed_status == w.observed_status
            assert r.forecast_status == w.forecast_status
            expected = getattr(w, col)
            if pd.isna(expected):
                assert pd.isna(r.predicted_share)
                assert r.eligibility_reason == w.eligibility_reason
            else:
                assert r.predicted_share == pytest.approx(expected)
                assert pd.isna(r.eligibility_reason)


def test_change_is_relative_to_the_latest_week(results):
    r = results.loc[("999", "2023-02-04", "crash", "rolling28_v1")]
    assert r.predicted_change_pp == pytest.approx(100 * (47 / 310 - 0.26))
    p = results.loc[("999", "2023-02-04", "crash", "persistence_v1")]
    assert p.predicted_change_pp == pytest.approx(0.0)


def test_score_uses_only_rows_with_targets(frames):
    feats, res = frames
    scores = baselines.score(res, feats, train_end="2023-01-31")
    crash = scores[(scores.category == "crash") & (scores.method_version == "persistence_v1")]
    # Game 999 is eligible from day 35 only, and day 35 has no target week yet.
    assert crash.rows_scored.sum() == 0 and crash.rows_without_target.sum() == 1


def test_score_mae_and_periods():
    from datetime import date

    from pipeline.config import get_spark
    spark = get_spark("test")
    key = ["app_id", "as_of_date", "category"]
    feats = spark.createDataFrame([
        ("1", date(2023, 5, 1), "crash", 0.30),   # target week ends May 8: train
        ("1", date(2023, 5, 2), "crash", 0.10),   # train
        ("1", date(2023, 5, 30), "crash", 0.20),  # target week ends Jun 6: later
        ("1", date(2023, 5, 31), "crash", None),  # no target: counted, not scored
    ], [*key, config.LABEL_COL])
    res = spark.createDataFrame([
        ("1", date(2023, 5, 1), "crash", "m", 0.20, "ok"),
        ("1", date(2023, 5, 2), "crash", "m", 0.20, "ok"),
        ("1", date(2023, 5, 30), "crash", "m", 0.50, "ok"),
        ("1", date(2023, 5, 31), "crash", "m", 0.50, "ok"),
    ], [*key, "method_version", "predicted_share", "forecast_status"])
    s = baselines.score(res, feats, train_end="2023-05-31").set_index("period")
    assert s.loc["train", "mae"] == pytest.approx(0.10) and s.loc["train", "rows_scored"] == 2
    assert s.loc["later", "mae"] == pytest.approx(0.30)
    assert (s.loc["later", "rows_scored"], s.loc["later", "rows_without_target"]) == (1, 1)
