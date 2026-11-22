"""Build forecast_features from daily_category: one row per game, day and category.

    python -m pipeline.features --label-version V

Every feature at as_of_date t uses only days up to t. The target uses t+1..t+7
and lives in its own column (target_share_next7), never in FEATURE_COLS.

Windows are date ranges (rangeBetween on a day number), not row counts, so a
missing row can never stretch a window past its intended dates. Gap days are rows
with null counts: sums skip nulls, so each window also counts its covered days,
and a window's values are null unless every day in it is covered.

eligibility (first rule that applies):
  insufficient_history  fewer than 35 days of grid up to t
  gap_in_window         a gap day in t-34..t
  low_volume            n_7d < MIN_N7 or the t-34..t-7 baseline has < MIN_N28 negatives
  eligible              usable for training and forecasting

Writes data/staging/forecast_features/ (partitioned by app_id) and
data/staging/feature_manifest.json.
"""
import argparse
import json
from datetime import datetime, timezone

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from pipeline import config
from pipeline.aggregate import DAILY_CATEGORY_DIR
from pipeline.config import get_spark

OUT_DIR = config.STAGING_DIR / "forecast_features"
MANIFEST_PATH = config.STAGING_DIR / "feature_manifest.json"

# name: (first offset, last offset) in days relative to t
WINDOWS = {
    "1d": (0, 0), "7d": (-6, 0), "14d": (-13, 0), "28d": (-27, 0),
    "prior7d": (-13, -7), "baseline": (-34, -7), "hist": (-34, 0), "next7": (1, 7),
}

DESCRIPTIONS = {
    "c_1d": "category count on t", "c_7d": "category count t-6..t",
    "c_14d": "category count t-13..t", "c_28d": "category count t-27..t",
    "n_1d": "English negatives on t", "n_7d": "English negatives t-6..t",
    "n_14d": "English negatives t-13..t", "n_28d": "English negatives t-27..t",
    "share_7d": "c_7d / n_7d", "share_14d": "c_14d / n_14d", "share_28d": "c_28d / n_28d",
    "share_prior7d": "category share t-13..t-7 (the week before the latest)",
    "baseline_share": "category share t-34..t-7 (detection baseline)",
    "week_change_pp": "100 x (share_7d - share_prior7d)",
    "lift_pp": "100 x (share_7d - baseline_share)",
    "count_std_28d": "sample std of daily category counts t-27..t",
    "zero_day_rate_28d": "fraction of days t-27..t with no reviews in the category",
    "weekday": "weekday of t, 1 = Monday .. 7 = Sunday",
    "prior7d_missing": "1 when share_prior7d is null (no English negatives t-13..t-7)",
}


def _ratio(num: Column, den: Column) -> Column:
    return F.when(den > 0, num / den)


def build(daily_cat: DataFrame) -> DataFrame:
    """daily_category rows for one label_version -> forecast_features rows."""
    df = daily_cat.withColumn("_day", F.datediff("date", F.lit("1970-01-01")))
    base = Window.partitionBy("app_id", "category").orderBy("_day")
    covered = (F.col("coverage_status") == "covered").cast("int")

    for name, (lo, hi) in WINDOWS.items():
        w = base.rangeBetween(lo, hi)
        complete = F.sum(covered).over(w) == (hi - lo + 1)
        df = (df.withColumn(f"_days_{name}", F.count("*").over(w))
                .withColumn(f"_ok_{name}", complete)
                .withColumn(f"_c_{name}", F.when(complete, F.sum("category_count").over(w)))
                .withColumn(f"_n_{name}", F.when(complete, F.sum("negative_total").over(w))))

    def share(name: str) -> Column:
        return _ratio(F.col(f"_c_{name}"), F.col(f"_n_{name}"))

    w28 = base.rangeBetween(-27, 0)
    df = df.select(
        "*",
        *[F.col(f"_c_{w}").cast("long").alias(f"c_{w}") for w in ("1d", "7d", "14d", "28d")],
        *[F.col(f"_n_{w}").cast("long").alias(f"n_{w}") for w in ("1d", "7d", "14d", "28d")],
        share("7d").alias("share_7d"), share("14d").alias("share_14d"), share("28d").alias("share_28d"),
        share("prior7d").alias("share_prior7d"), share("baseline").alias("baseline_share"),
        F.when(F.col("_ok_28d"), F.stddev_samp("category_count").over(w28)).alias("count_std_28d"),
        F.when(F.col("_ok_28d"), F.avg((F.col("category_count") == 0).cast("double")).over(w28))
         .alias("zero_day_rate_28d"),
        (F.expr("weekday(date)") + 1).cast("int").alias("weekday"),
        share("next7").alias(config.LABEL_COL),
        F.col("_n_next7").cast("long").alias("target_n_next7"),
    )
    df = (df.withColumn("week_change_pp", 100 * (F.col("share_7d") - F.col("share_prior7d")))
            .withColumn("lift_pp", 100 * (F.col("share_7d") - F.col("baseline_share")))
            .withColumn("prior7d_missing", F.col("share_prior7d").isNull().cast("int")))

    eligibility = (
        F.when(F.col("_days_hist") < config.HISTORY_DAYS, "insufficient_history")
        .when(~F.col("_ok_hist"), "gap_in_window")
        .when((F.col("n_7d") < config.MIN_N7) | (F.col("_n_baseline") < config.MIN_N28), "low_volume")
        .otherwise("eligible")
    )
    return (df.withColumn("eligibility", eligibility)
              .withColumn("as_of_date", F.col("date"))
              .withColumn("feature_version", F.lit(config.FEATURE_VERSION))
              .withColumn("feature_max_date", F.col("date"))
              .withColumn("source_snapshot", F.lit(config.SNAPSHOT_ID))
              .select(*config.FORECAST_FEATURES_COLS))


def manifest(features: DataFrame, label_version: str) -> dict:
    types = {f.name: f.dataType.simpleString() for f in features.schema.fields}
    return {
        "feature_version": config.FEATURE_VERSION,
        "label_version": label_version,
        "snapshot_id": config.SNAPSHOT_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "key": config.FORECAST_FEATURES_KEY,
        "features": [{"name": c, "type": types[c], "description": DESCRIPTIONS[c]}
                     for c in config.FEATURE_COLS],
        "label": config.LABEL_COL,
        "target_columns": config.TARGET_COLS,
        "meta_columns": config.FORECAST_FEATURES_META,
        "eligibility": {
            "values": config.ELIGIBILITY_VALUES,
            "history_days": config.HISTORY_DAYS, "min_n7": config.MIN_N7, "min_n28_baseline": config.MIN_N28,
        },
        "windows": {k: list(v) for k, v in WINDOWS.items()},
    }


def main(label_version: str) -> None:
    spark = get_spark("steampulse-features")
    daily = spark.read.parquet(str(DAILY_CATEGORY_DIR)).where(F.col("label_version") == label_version)
    feats = build(daily).cache()
    if feats.limit(1).count() == 0:
        raise SystemExit(f"No daily_category rows with label_version {label_version!r}. Run `make daily` first.")
    feats.repartition("app_id").write.mode("overwrite").partitionBy("app_id").parquet(str(OUT_DIR))
    MANIFEST_PATH.write_text(json.dumps(manifest(feats, label_version), indent=2) + "\n")

    counts = feats.groupBy("eligibility").count().collect()
    print(", ".join(f"{r.eligibility} {r['count']:,}" for r in sorted(counts, key=lambda r: r.eligibility)))
    print(f"Wrote {OUT_DIR.relative_to(config.ROOT)} and {MANIFEST_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-version", required=True)
    main(parser.parse_args().label_version)
