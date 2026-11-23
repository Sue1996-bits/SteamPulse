"""Build reviews_clean for the chosen games.

    python -m pipeline.clean                  # all games in config.GAMES
    python -m pipeline.clean --apps 1172470   # just Apex

Every row of the chosen games ends up in exactly one place:

  quarantine     missing ID or timestamp, non-integer ID, missing vote, or empty text
                 -> data/quarantine/reviews_clean/ with a `reason` column
  out_of_window  created outside WINDOW_START..WINDOW_END (dropped, counted)
  edited         updated after it was created, so not in the strict cohort (dropped, counted)
  kept           -> data/staging/reviews_clean/, partitioned by app_id

Kept rows are deduplicated on review_id. Every language and both vote directions
are kept: the daily step needs all reviews to tell a gap day from a real zero.
Per-game counts for each stage go to data/staging/clean_summary.json.

The raw files aren't split by game, so every run scans all of them. Cleaning
all five games costs about the same as cleaning one.
"""
import argparse
import json
import time
from datetime import datetime, timezone

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from pipeline import config
from pipeline.config import get_spark

OUT_DIR = config.STAGING_DIR / "reviews_clean"
QUARANTINE_OUT = config.QUARANTINE_DIR / "reviews_clean"
SUMMARY_PATH = config.STAGING_DIR / "clean_summary.json"

# Checked in this order; a row gets the first reason that applies.
QUARANTINE_REASONS = ["missing_key", "bad_id", "missing_vote", "empty_text"]
STAGES = [*QUARANTINE_REASONS, "out_of_window", "edited", "kept"]


def with_stage(raw: DataFrame, start: str = config.WINDOW_START, end: str = config.WINDOW_END) -> DataFrame:
    """Add a `stage` column saying where each raw row goes (see module docstring)."""
    rid, created = F.col("recommendationid"), F.col("timestamp_created")
    day = F.to_date(F.timestamp_seconds(created))  # UTC session time zone
    stage = (
        F.when(rid.isNull() | F.col("appid").isNull() | created.isNull()
               | F.col("timestamp_updated").isNull(), "missing_key")
        .when((rid != F.floor(rid)) | (rid <= 0), "bad_id")
        .when(F.col("voted_up").isNull(), "missing_vote")
        .when(F.col("review").isNull() | (F.trim("review") == ""), "empty_text")
        .when((day < F.lit(start).cast("date")) | (day > F.lit(end).cast("date")), "out_of_window")
        .when(created != F.col("timestamp_updated"), "edited")
        .otherwise("kept")
    )
    return raw.withColumn("stage", stage)


def normalize_text(text: Column) -> Column:
    """Lowercase, drop Steam BBCode tags and URLs, collapse whitespace.

    Punctuation and emoji are kept so labeling rules can still match on them.
    """
    t = F.lower(text)
    t = F.regexp_replace(t, r"\[/?[a-z0-9*]+(=[^\]]*)?\]", " ")  # [b], [/spoiler], [url=...]
    t = F.regexp_replace(t, r"https?://\S+", " ")
    t = F.regexp_replace(t, r"\s+", " ")
    return F.trim(t)


def to_clean(kept: DataFrame) -> DataFrame:
    """Rename and cast kept raw rows into the reviews_clean schema, one row per review_id."""
    created = F.timestamp_seconds("timestamp_created")
    updated = F.timestamp_seconds("timestamp_updated")
    typed = kept.select(
        F.col("recommendationid").cast("long").cast("string").alias("review_id"),
        F.col("appid").cast("long").cast("string").alias("app_id"),
        F.col("review").alias("text_raw"),
        normalize_text(F.col("review")).alias("text_normalized"),
        created.alias("created_at"),
        updated.alias("updated_at"),
        updated.alias("available_at"),  # the text we have is the version as of its last update
        F.to_date(created).alias("review_date"),
        F.col("language"),
        (F.col("voted_up") != 0).alias("voted_up"),
        F.lit(config.SNAPSHOT_ID).alias("snapshot_id"),
        F.lit(config.COHORT_POLICY).alias("cohort_policy"),
    )
    # The profile found no duplicate IDs, but keep one row per ID regardless:
    # the latest update wins, and text breaks ties so reruns pick the same row.
    w = Window.partitionBy("review_id").orderBy(F.col("updated_at").desc(), F.col("text_raw"))
    return (typed.withColumn("_rn", F.row_number().over(w))
            .where("_rn = 1").drop("_rn")
            .select(*config.REVIEWS_CLEAN_COLS))


def stage_counts(staged: DataFrame) -> dict[str, dict[str, int]]:
    """{app_id: {stage: rows}} with every stage present."""
    rows = staged.groupBy("appid", "stage").count().collect()
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        app = str(int(r.appid))
        out.setdefault(app, dict.fromkeys(STAGES, 0))[r.stage] = r["count"]
    return out


def write_summary(counts: dict, clean_counts: dict, seconds: float) -> dict:
    """Merge this run's per-game counts into clean_summary.json (other games are kept)."""
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {"games": {}}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for app, c in counts.items():
        kept_unique = clean_counts.get(app, {}).get("rows", 0)
        summary["games"][app] = {
            "game": config.GAMES.get(app),
            "run_at": now,
            "raw_rows": sum(c.values()),
            **c,
            "duplicates_dropped": c["kept"] - kept_unique,
            "reviews_clean_rows": kept_unique,
            "english_negatives": clean_counts.get(app, {}).get("english_negatives", 0),
        }
    summary.update({
        "snapshot_id": config.SNAPSHOT_ID,
        "window": [config.WINDOW_START, config.WINDOW_END],
        "cohort_policy": config.COHORT_POLICY,
        "last_run_seconds": round(seconds),
    })
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(apps: list[str]) -> None:
    t0 = time.perf_counter()
    spark = get_spark("steampulse-clean")
    raw = (spark.read.parquet(str(config.RAW_DIR / "part.*.parquet"))
           .select(*config.RAW_COLUMNS)
           .where(F.col("appid").isin([float(a) for a in apps])))
    # Persist the chosen games' rows so the counts, quarantine and output
    # don't each rescan the full dataset.
    staged = with_stage(raw).persist(StorageLevel.MEMORY_AND_DISK)

    counts = stage_counts(staged)
    missing = sorted(set(apps) - set(counts))
    if missing:
        raise SystemExit(f"No rows found for app IDs {missing}. Check data/raw/ and the IDs.")

    (staged.where(F.col("stage").isin(QUARANTINE_REASONS))
     .withColumnRenamed("stage", "reason")
     .withColumn("app_id", F.col("appid").cast("long").cast("string"))
     .write.mode("overwrite").partitionBy("app_id").parquet(str(QUARANTINE_OUT)))

    to_clean(staged.where("stage = 'kept'")).write.mode("overwrite").partitionBy("app_id").parquet(str(OUT_DIR))
    staged.unpersist()

    written = spark.read.parquet(str(OUT_DIR)).where(F.col("app_id").isin(apps))
    clean_counts = {
        r.app_id: {"rows": r.rows, "english_negatives": r.en_neg}
        for r in written.groupBy("app_id").agg(
            F.count("*").alias("rows"),
            F.sum(((F.col("language") == config.LANGUAGE) & ~F.col("voted_up")).cast("int")).alias("en_neg"),
        ).collect()
    }
    seconds = time.perf_counter() - t0
    summary = write_summary(counts, clean_counts, seconds)

    for app in apps:
        g = summary["games"][app]
        dropped = ", ".join(f"{s} {g[s]:,}" for s in STAGES if s != "kept" and g[s])
        print(f"{app} {g['game']}: {g['raw_rows']:,} raw -> {g['reviews_clean_rows']:,} kept "
              f"({g['english_negatives']:,} English negatives). Dropped: {dropped or 'none'}")
    print(f"Wrote {OUT_DIR.relative_to(config.ROOT)} and {SUMMARY_PATH.relative_to(config.ROOT)} in {seconds:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apps", nargs="+", default=list(config.GAMES), help="app IDs (default: config.GAMES)")
    main(parser.parse_args().apps)
