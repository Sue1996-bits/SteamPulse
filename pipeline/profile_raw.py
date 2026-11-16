"""Per-game profile of the raw snapshot (rows, unique IDs, English negatives, edit rate, date range, coverage).

    python -m pipeline.profile_raw

Writes to data/profile/:
  game_profile.csv      one row per game, sorted by strict English negatives
  daily_counts/         one row per game per UTC day with any review (Parquet)
  scale_run.json        files, bytes, rows and runtime for the scale log

Each run also appends a row to docs/scale_log.md. Pass --no-log for a
throwaway rerun so duplicate runs don't end up in the scale totals.

Profiling covers the whole corpus. Game selection must later be judged on the
training period only (docs/temporal_policy.md).
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pipeline import config
from pipeline.config import get_spark


def with_flags(raw: DataFrame) -> DataFrame:
    """Add the derived columns that every profile metric is built from."""
    is_en = F.col("language") == config.LANGUAGE
    is_neg = F.col("voted_up") == 0
    unedited = F.col("timestamp_created") == F.col("timestamp_updated")
    return raw.select(
        "recommendationid", "appid", "game",
        F.to_date(F.timestamp_seconds("timestamp_created")).alias("date"),  # UTC session tz
        is_en.alias("is_en"),
        (is_en & is_neg).alias("is_en_neg"),
        (is_en & is_neg & unedited).alias("is_en_neg_strict"),
        unedited.alias("unedited"),
        (F.col("recommendationid") != F.floor("recommendationid")).alias("bad_id"),
        (F.col("recommendationid").isNull() | F.col("appid").isNull()
         | F.col("timestamp_created").isNull()).alias("missing_key"),
        (F.col("review").isNull() | (F.trim("review") == "")).alias("empty_text"),
    )


def daily_counts(flags: DataFrame) -> DataFrame:
    """Reviews per game per UTC day. Days with no reviews at all are absent."""
    return (
        flags.where(F.col("date").isNotNull())
        .groupBy("appid", "date")
        .agg(
            F.count("*").alias("all_reviews"),
            F.sum(F.col("is_en").cast("int")).alias("en_reviews"),
            F.sum(F.col("is_en_neg").cast("int")).alias("en_neg"),
            F.sum(F.col("is_en_neg_strict").cast("int")).alias("en_neg_strict"),
        )
    )


def game_profile(flags: DataFrame, daily: DataFrame) -> DataFrame:
    def total(col: str) -> F.Column:
        return F.sum(F.col(col).cast("int")).alias(col)

    per_game = flags.groupBy("appid").agg(
        F.first("game", ignorenulls=True).alias("game"),
        F.count("*").alias("rows"),
        F.countDistinct("recommendationid").alias("unique_ids"),
        total("missing_key"), total("bad_id"), total("empty_text"),
        F.sum(F.col("is_en").cast("int")).alias("en_reviews"),
        F.sum(F.col("is_en_neg").cast("int")).alias("en_neg"),
        F.sum(F.col("is_en_neg_strict").cast("int")).alias("en_neg_strict"),
        F.avg((~F.col("unedited")).cast("double")).alias("edit_rate"),
    )
    span = daily.groupBy("appid").agg(
        F.min("date").alias("first_date"),
        F.max("date").alias("last_date"),
        F.count("*").alias("active_days"),
    ).withColumn("span_days", F.datediff("last_date", "first_date") + 1)

    return (
        per_game.join(span, "appid", "left")
        .withColumn("duplicate_rows", F.col("rows") - F.col("unique_ids"))
        .withColumn("days_without_reviews", F.col("span_days") - F.col("active_days"))
        .withColumn("en_neg_edit_share",
                    F.when(F.col("en_neg") > 0, 1 - F.col("en_neg_strict") / F.col("en_neg")))
        .withColumn("en_neg_strict_per_day",
                    F.when(F.col("span_days") > 0, F.col("en_neg_strict") / F.col("span_days")))
        .orderBy(F.desc("en_neg_strict"))
    )


def parquet_bytes(files: list[Path], columns: list[str]) -> dict:
    """Compressed and uncompressed bytes from Parquet footers, total and for `columns`."""
    out = {"compressed_all": 0, "uncompressed_all": 0,
           "compressed_selected": 0, "uncompressed_selected": 0, "rows": 0}
    for f in files:
        meta = pq.ParquetFile(f).metadata
        out["rows"] += meta.num_rows
        for rg in range(meta.num_row_groups):
            group = meta.row_group(rg)
            for c in range(group.num_columns):
                col = group.column(c)
                selected = col.path_in_schema in columns
                for kind, size in (("compressed", col.total_compressed_size),
                                   ("uncompressed", col.total_uncompressed_size)):
                    out[f"{kind}_all"] += size
                    if selected:
                        out[f"{kind}_selected"] += size
    return out


def main(log: bool = True) -> None:
    files = sorted(config.RAW_DIR.glob("part.*.parquet"))
    if not files:
        raise SystemExit(f"No part files in {config.RAW_DIR}. Run `make download-sample` first.")

    start = time.perf_counter()
    spark = get_spark("steampulse-profile")
    raw = spark.read.parquet(*map(str, files)).select(*config.RAW_COLUMNS)
    flags = with_flags(raw).cache()
    daily = daily_counts(flags).cache()
    profile = game_profile(flags, daily)

    config.PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    daily_dir = config.PROFILE_DIR / "daily_counts"
    daily.write.mode("overwrite").parquet(str(daily_dir))
    profile_pd = profile.toPandas()
    profile_pd.to_csv(config.PROFILE_DIR / "game_profile.csv", index=False)
    elapsed = time.perf_counter() - start

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_id": config.SNAPSHOT_ID,
        "files": len(files),
        "columns": config.RAW_COLUMNS,
        **parquet_bytes(files, config.RAW_COLUMNS),
        "unique_ids": int(profile_pd["unique_ids"].sum()),
        "games": len(profile_pd),
        "game_days": daily.count(),
        "output_files": len(list(daily_dir.glob("*.parquet"))) + 1,  # + game_profile.csv
        "output_bytes": sum(f.stat().st_size for f in config.PROFILE_DIR.rglob("*") if f.is_file()),
        "driver_memory": spark.conf.get("spark.driver.memory"),
        "runtime_s": round(elapsed, 1),
    }
    (config.PROFILE_DIR / "scale_run.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    row = scale_log_row(stats)
    if log:
        append_scale_log(row)
        print(f"\nAppended to {SCALE_LOG.relative_to(config.ROOT)}:\n{row}")
    else:
        print(f"\nNot logged (--no-log):\n{row}")


SCALE_LOG = config.ROOT / "docs" / "scale_log.md"


def _bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def scale_log_row(stats: dict) -> str:
    """One Markdown table row matching the columns in docs/scale_log.md."""
    total_files = 663
    scope = "full snapshot" if stats["files"] >= total_files else f"benchmark ({stats['files']}/{total_files} files)"
    cells = [
        stats["generated_at"][:10],
        "profile",
        f"{stats['files']} part files",
        f"{len(stats['columns'])} of 25",
        _bytes(stats["compressed_selected"]),
        _bytes(stats["uncompressed_selected"]),
        f"{stats['rows']:,}",
        f"{stats['unique_ids']:,}",
        "none",
        f"{stats['output_files']} files",
        f"{stats['runtime_s']}s",
        _bytes(stats["output_bytes"]),
        f"local[*], driver {stats['driver_memory']}",
        f"{scope}; {stats['games']:,} games; all columns {_bytes(stats['uncompressed_all'])} uncompressed; "
        f"snapshot {stats['snapshot_id'][:7]}",
    ]
    return "| " + " | ".join(cells) + " |"


def append_scale_log(row: str, path: Path = SCALE_LOG) -> None:
    text = path.read_text()
    path.write_text(text + ("" if text.endswith("\n") else "\n") + row + "\n")


if __name__ == "__main__":
    import sys

    main(log="--no-log" not in sys.argv[1:])
