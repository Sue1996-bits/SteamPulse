"""Build daily_negative and daily_category from reviews_clean and review_labels.

    python -m pipeline.aggregate                                   # daily_negative only
    python -m pipeline.aggregate --labels PATH --label-version V   # both tables

Both tables have one row per game per day (four in daily_category, one per
category), from the game's first review date to WINDOW_END:

  covered  the game has at least one review that day, in any language.
           negative_total counts English negatives; a day without any is a real 0.
  gap      no reviews at all. Counts and shares are null, never 0.

A review counts once in negative_total however many labels it has, and once per
category it's labeled with. Negatives with no label stay in the denominator.
share is category_count / negative_total, and null when negative_total is 0.

Outputs go to data/staging/, partitioned by app_id. Labels are read from a
Parquet directory or a CSV file in the review_labels schema.
"""
import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pipeline import config
from pipeline.clean import OUT_DIR as CLEAN_DIR
from pipeline.config import get_spark

DAILY_NEGATIVE_DIR = config.STAGING_DIR / "daily_negative"
DAILY_CATEGORY_DIR = config.STAGING_DIR / "daily_category"


def date_grid(clean: DataFrame, end: str = config.WINDOW_END) -> DataFrame:
    """(app_id, date) for every day from each game's first review date to `end`."""
    first = clean.groupBy("app_id").agg(F.min("review_date").alias("first_date"))
    return first.select(
        "app_id",
        F.explode(F.sequence("first_date", F.lit(end).cast("date"))).alias("date"),
    )


def daily_negative(clean: DataFrame, end: str = config.WINDOW_END) -> DataFrame:
    is_en_neg = (F.col("language") == config.LANGUAGE) & ~F.col("voted_up")
    per_day = (clean.groupBy("app_id", F.col("review_date").alias("date"))
               .agg(F.sum(is_en_neg.cast("long")).alias("negative_total")))
    return (date_grid(clean, end).join(per_day, ["app_id", "date"], "left")
            .withColumn("coverage_status",
                        F.when(F.col("negative_total").isNull(), "gap").otherwise("covered"))
            .withColumn("cohort_version", F.lit(config.COHORT_VERSION))
            .withColumn("snapshot_id", F.lit(config.SNAPSHOT_ID))
            .select(*config.DAILY_NEGATIVE_COLS))


def daily_category(clean: DataFrame, labels: DataFrame, daily_neg: DataFrame,
                   label_version: str) -> DataFrame:
    en_neg = (clean.where((F.col("language") == config.LANGUAGE) & ~F.col("voted_up"))
              .select("review_id", "app_id", F.col("review_date").alias("date")))
    # Duplicate label rows count once; labels on positives or other languages count nowhere.
    assigned = (labels.where((F.col("label_version") == label_version) & F.col("is_match"))
                .select("review_id", "category").distinct())
    counts = (en_neg.join(assigned, "review_id")
              .groupBy("app_id", "date", "category").agg(F.count("*").alias("n")))

    cats = daily_neg.sparkSession.createDataFrame([(c,) for c in config.CATEGORIES], ["category"])
    covered = F.col("coverage_status") == "covered"
    return (daily_neg.select("app_id", "date", "negative_total", "coverage_status").crossJoin(cats)
            .join(counts, ["app_id", "date", "category"], "left")
            .withColumn("category_count", F.when(covered, F.coalesce("n", F.lit(0)).cast("long")))
            .withColumn("share", F.when(F.col("negative_total") > 0,
                                        F.col("category_count") / F.col("negative_total")))
            .withColumn("label_version", F.lit(label_version))
            .withColumn("cohort_version", F.lit(config.COHORT_VERSION))
            .select(*config.DAILY_CATEGORY_COLS))


def read_labels(spark: SparkSession, path: Path) -> DataFrame:
    if path.suffix == ".csv":
        df = spark.read.csv(str(path), header=True, inferSchema=False)
        return df.withColumn("is_match", F.lower("is_match").isin("true", "1"))
    return spark.read.parquet(str(path))


def main(labels_path: Path | None, label_version: str | None) -> None:
    spark = get_spark("steampulse-aggregate")
    clean = spark.read.parquet(str(CLEAN_DIR))
    neg = daily_negative(clean).cache()
    neg.repartition("app_id").write.mode("overwrite").partitionBy("app_id").parquet(str(DAILY_NEGATIVE_DIR))
    summary = neg.groupBy("app_id", "coverage_status").count().orderBy("app_id").collect()
    print("daily_negative:", ", ".join(f"{r.app_id} {r.coverage_status} {r['count']}" for r in summary))

    if labels_path is None:
        print("No --labels given, so daily_category was not built.")
        return
    cat = daily_category(clean, read_labels(spark, labels_path), neg, label_version)
    cat.repartition("app_id").write.mode("overwrite").partitionBy("app_id").parquet(str(DAILY_CATEGORY_DIR))
    print(f"daily_category ({label_version}) -> {DAILY_CATEGORY_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, help="review_labels as Parquet dir or CSV")
    parser.add_argument("--label-version", help="which label_version to use (required with --labels)")
    args = parser.parse_args()
    if args.labels and not args.label_version:
        parser.error("--label-version is required with --labels")
    main(args.labels, args.label_version)
