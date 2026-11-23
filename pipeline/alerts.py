"""Emerging-issue alerts: how often they fire, how they depend on the thresholds, and what follows them.

    python -m pipeline.alerts --label-version rules_v1                 # train + validation
    python -m pipeline.alerts --label-version rules_v1 --include-test  # once, thresholds frozen

The alert is the observed status from baselines.observed_status: an eligible row
is `emerging` when c_7d >= MIN_C7 and lift_pp >= 100 x MIN_LIFT (the latest 7 days
against the 28 days before them). It describes the past week only. To check
whether alerts point at something real, each eligible row with a target week is
followed one week ahead:

  next_change_pp   100 x (target_share_next7 - share_7d): does the share keep rising?
  stays_elevated   target_share_next7 - baseline_share >= MIN_LIFT: is next week still
                   at least MIN_LIFT above the old baseline?

Rows that didn't alert give the comparison rate, so "alerts are followed by a
raised share" can be told apart from "shares are often raised".

Writes to data/staging/:
  alert_summary.csv      per split, game and category at the configured thresholds
  alert_sensitivity.csv  validation only: the same measures over a grid of MIN_LIFT,
                         MIN_C7 and a stricter MIN_N7. Tune on this file, then freeze
                         the thresholds in config.py before looking at test.
"""
import argparse
from itertools import product

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pipeline import config
from pipeline.config import get_spark
from pipeline.features import OUT_DIR as FEATURES_DIR
from pipeline.splits import split_col, usable

SUMMARY_PATH = config.STAGING_DIR / "alert_summary.csv"
SENSITIVITY_PATH = config.STAGING_DIR / "alert_sensitivity.csv"
GRID = {"min_lift": [0.03, 0.05, 0.08, 0.10], "min_c7": [5, 10, 20], "min_n7": [config.MIN_N7, 100]}
COLS = ["app_id", "as_of_date", "category", "split", "c_7d", "n_7d", "lift_pp", "share_7d",
        "baseline_share", config.LABEL_COL]


def followed(feats: DataFrame) -> pd.DataFrame:
    """Eligible rows with a target week, with the one-week-ahead outcomes."""
    df = feats.withColumn("split", split_col()).where(usable()).select(*COLS).toPandas()
    df["next_change_pp"] = 100 * (df[config.LABEL_COL] - df.share_7d)
    df["next_lift"] = df[config.LABEL_COL] - df.baseline_share
    return df


def is_alert(df: pd.DataFrame, min_lift: float = config.MIN_LIFT, min_c7: int = config.MIN_C7,
             min_n7: int = config.MIN_N7) -> pd.Series:
    """Same rule as baselines.observed_status (eligibility is already applied), plus an optional stricter N7."""
    return (df.c_7d >= min_c7) & (df.lift_pp >= 100 * min_lift) & (df.n_7d >= min_n7)


def measures(df: pd.DataFrame, alert: pd.Series, min_lift: float) -> dict:
    elevated = df.next_lift >= min_lift
    a, rest = alert, ~alert
    return dict(
        rows=len(df), alerts=int(a.sum()), alert_rate=a.mean() if len(df) else None,
        stays_elevated_after_alert=elevated[a].mean() if a.any() else None,
        stays_elevated_without_alert=elevated[rest].mean() if rest.any() else None,
        next_change_pp_after_alert=df.next_change_pp[a].mean() if a.any() else None,
        next_change_pp_without_alert=df.next_change_pp[rest].mean() if rest.any() else None,
    )


def summary(df: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    rows = []
    df = df[df.split.isin(splits)]
    for (split, category), g in df.groupby(["split", "category"], sort=False):
        for app_id, h in [("all", g), *g.groupby("app_id")]:
            rows.append(dict(split=split, category=category, app_id=app_id,
                             **measures(h, is_alert(h), config.MIN_LIFT)))
    order = {s: i for i, s in enumerate(splits)}
    out = pd.DataFrame(rows)
    return out.sort_values(["split", "category", "app_id"], key=lambda s: s.map(order) if s.name == "split" else s)


def sensitivity(df: pd.DataFrame, split: str = "validation") -> pd.DataFrame:
    df = df[df.split == split]
    rows = []
    for min_lift, min_c7, min_n7 in product(GRID["min_lift"], GRID["min_c7"], GRID["min_n7"]):
        for category, g in [("all", df), *df.groupby("category")]:
            rows.append(dict(split=split, min_lift=min_lift, min_c7=min_c7, min_n7=min_n7, category=category,
                             configured=(min_lift, min_c7, min_n7) == (config.MIN_LIFT, config.MIN_C7, config.MIN_N7),
                             **measures(g, is_alert(g, min_lift, min_c7, min_n7), min_lift)))
    return pd.DataFrame(rows)


def main(label_version: str, include_test: bool) -> None:
    spark = get_spark("steampulse-alerts")
    feats = spark.read.parquet(str(FEATURES_DIR)).where(F.col("label_version") == label_version)
    df = followed(feats)
    if df.empty:
        raise SystemExit(f"No eligible forecast_features rows with label_version {label_version!r}.")
    splits = ["train", "validation"] + (["test"] if include_test else [])
    summ = summary(df, splits)
    summ.assign(label_version=label_version).to_csv(SUMMARY_PATH, index=False)
    sens = sensitivity(df)
    sens.assign(label_version=label_version).to_csv(SENSITIVITY_PATH, index=False)

    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(f"Alerts at MIN_LIFT={config.MIN_LIFT}, MIN_C7={config.MIN_C7}, MIN_N7={config.MIN_N7} (all games):")
        print(summ[summ.app_id == "all"].drop(columns="app_id").to_string(index=False))
        print("\nValidation sensitivity, all categories:")
        print(sens[sens.category == "all"].drop(columns=["split", "category"]).to_string(index=False))
    print(f"\nWrote {SUMMARY_PATH.relative_to(config.ROOT)} and {SENSITIVITY_PATH.relative_to(config.ROOT)}")
    if not include_test:
        print("Test period left out. Freeze the thresholds, then run once with --include-test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-version", default=config.LABEL_VERSION)
    parser.add_argument("--include-test", action="store_true", help="add the test period (do this once)")
    args = parser.parse_args()
    main(args.label_version, args.include_test)
