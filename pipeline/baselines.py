"""Persistence and rolling-28 baselines, written as forecast_results rows.

    python -m pipeline.baselines

Reads data/staging/forecast_features and writes, for every game, day and category:

  persistence_v1  predicts next week's share = share_7d (the latest seven days)
  rolling28_v1    predicts next week's share = share_28d (t-27..t)

Both rows also carry the observed (detection) fields, which don't depend on the method:
  emerging           eligible, c_7d >= MIN_C7 and lift_pp >= 100 x MIN_LIFT
  not_flagged        eligible otherwise
  insufficient_data  not eligible (short history, gap, or low volume)

Forecasts exist only for eligible rows (forecast_status "ok"). Unlike an alert,
a forecast doesn't need c_7d >= MIN_C7. Predictions are clipped to [0, 1].

Also writes data/staging/baseline_scores.csv: MAE per method, category and period
on eligible rows with a target. A row is "train" when its whole target week ends
by TRAIN_END, and "later" otherwise. Rows without a target are counted, not scored.
"""
import pandas as pd
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from pipeline import config
from pipeline.config import get_spark
from pipeline.features import OUT_DIR as FEATURES_DIR

OUT_DIR = config.STAGING_DIR / "forecast_results"
SCORES_PATH = config.STAGING_DIR / "baseline_scores.csv"
METHODS = {"persistence_v1": "share_7d", "rolling28_v1": "share_28d"}


def _clip(c: Column) -> Column:
    return F.least(F.greatest(c, F.lit(0.0)), F.lit(1.0))


def observed_status() -> Column:
    eligible = F.col("eligibility") == "eligible"
    rising = (F.col("c_7d") >= config.MIN_C7) & (F.col("lift_pp") >= 100 * config.MIN_LIFT)
    return (F.when(~eligible, "insufficient_data")
            .when(rising, "emerging")
            .otherwise("not_flagged"))


def build(feats: DataFrame) -> DataFrame:
    """forecast_features -> forecast_results, one row per method."""
    eligible = F.col("eligibility") == "eligible"
    parts = []
    for method, source in METHODS.items():
        pred = F.when(eligible, _clip(F.col(source)))
        parts.append(feats.select(
            "app_id", "as_of_date", "category",
            F.lit(method).alias("method_version"),
            F.col("share_7d").alias("recent_share"),
            "baseline_share",
            F.col("lift_pp").alias("observed_lift_pp"),
            observed_status().alias("observed_status"),
            pred.alias("predicted_share"),
            (100 * (pred - F.col("share_7d"))).alias("predicted_change_pp"),
            F.when(eligible, "ok").otherwise("insufficient_data").alias("forecast_status"),
            F.when(~eligible, F.col("eligibility")).alias("eligibility_reason"),
            F.lit(method).alias("model_version"),  # no training: the method is the model
            F.lit(None).cast("date").alias("training_label_end"),
            "feature_version",
        ))
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out.select(*config.FORECAST_RESULTS_COLS)


def score(results: DataFrame, feats: DataFrame, train_end: str = config.TRAIN_END) -> pd.DataFrame:
    """MAE per method, category and period. Only rows with forecast_status ok are used."""
    target_end = F.date_add("as_of_date", 7)
    joined = (results.where(F.col("forecast_status") == "ok")
              .join(feats.select("app_id", "as_of_date", "category", config.LABEL_COL),
                    ["app_id", "as_of_date", "category"])
              .withColumn("period", F.when(target_end <= F.lit(train_end).cast("date"), "train")
                          .otherwise("later")))
    has_target = F.col(config.LABEL_COL).isNotNull()
    err = F.abs(F.col("predicted_share") - F.col(config.LABEL_COL))
    return (joined.groupBy("method_version", "category", "period")
            .agg(F.sum(has_target.cast("int")).alias("rows_scored"),
                 F.sum((~has_target).cast("int")).alias("rows_without_target"),
                 F.avg(F.when(has_target, err)).alias("mae"))
            .orderBy("period", "category", "method_version")
            .toPandas())


def main() -> None:
    spark = get_spark("steampulse-baselines")
    feats = spark.read.parquet(str(FEATURES_DIR))
    results = build(feats).cache()
    results.repartition("app_id").write.mode("overwrite").partitionBy("app_id").parquet(str(OUT_DIR))
    scores = score(results, feats)
    scores.to_csv(SCORES_PATH, index=False)
    with pd.option_context("display.width", 200):
        print(scores.round(4).to_string(index=False))
    print(f"Wrote {OUT_DIR.relative_to(config.ROOT)} and {SCORES_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
