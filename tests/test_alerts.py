"""Alert analysis: same rule as the baselines' observed status, hand-checked measures, and the sensitivity grid."""
import pandas as pd
import pytest
from pyspark.sql import functions as F

from pipeline import alerts, baselines, config
from pipeline.splits import split_col, usable
from tests.test_train import synthetic_features


@pytest.fixture(scope="module")
def feats():
    return synthetic_features().cache()


@pytest.fixture(scope="module")
def df(feats):
    return alerts.followed(feats)


def test_alerts_match_observed_status(feats, df):
    """The analysis counts exactly the rows the baselines mark emerging."""
    status = (baselines.build(feats).where(F.col("method_version") == "persistence_v1")
              .join(feats.select("app_id", "as_of_date", "category", "eligibility", config.LABEL_COL),
                    ["app_id", "as_of_date", "category"])
              .withColumn("split", split_col()).where(usable()))
    expected = {(r.split, r.category): r["count"] for r in
                status.where(F.col("observed_status") == "emerging").groupBy("split", "category").count().collect()}
    summ = alerts.summary(df, ["train", "validation", "test"])
    got = {(r.split, r.category): r.alerts for r in summ[summ.app_id == "all"].itertuples() if r.alerts}
    assert got == expected and sum(got.values()) > 0


def test_measures_by_hand():
    df = pd.DataFrame(dict(c_7d=[20, 20, 3, 20], n_7d=[100] * 4, lift_pp=[10.0, 6.0, 9.0, 1.0],
                           next_lift=[0.07, 0.01, 0.06, 0.0], next_change_pp=[2.0, -4.0, 1.0, 0.0]))
    alert = alerts.is_alert(df)               # rows 0 and 1 (row 2 has c_7d < 10, row 3 a small lift)
    assert alert.tolist() == [True, True, False, False]
    m = alerts.measures(df, alert, 0.05)
    assert (m["alerts"], m["alert_rate"]) == (2, 0.5)
    assert m["stays_elevated_after_alert"] == 0.5 and m["stays_elevated_without_alert"] == 0.5
    assert m["next_change_pp_after_alert"] == pytest.approx(-1.0)


def test_sensitivity_grid_contains_the_configured_rule(df):
    sens = alerts.sensitivity(df)
    n_settings = len(alerts.GRID["min_lift"]) * len(alerts.GRID["min_c7"]) * len(alerts.GRID["min_n7"])
    assert len(sens) == n_settings * (len(config.CATEGORIES) + 1)
    assert set(sens.split) == {"validation"}
    configured = sens[sens.configured & (sens.category == "all")]
    summ = alerts.summary(df, ["validation"])
    assert configured.alerts.iloc[0] == summ[summ.app_id == "all"].alerts.sum()
    loose = sens[(sens.category == "all") & (sens.min_lift == 0.03) & (sens.min_c7 == 5) & (sens.min_n7 == config.MIN_N7)]
    strict = sens[(sens.category == "all") & (sens.min_lift == 0.10) & (sens.min_c7 == 20) & (sens.min_n7 == 100)]
    assert loose.alerts.iloc[0] >= configured.alerts.iloc[0] >= strict.alerts.iloc[0]


def test_summary_leaves_test_out_by_default(df):
    assert set(alerts.summary(df, ["train", "validation"]).split) == {"train", "validation"}
