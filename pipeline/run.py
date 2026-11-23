"""Run the whole pipeline in order, time each step, and optionally log it.

    python -m pipeline.run                                         # clean -> daily -> checks
    python -m pipeline.run --label-version V                       # + rule labels, features and baselines
    python -m pipeline.run --labels PATH --label-version V         # same, with labels from PATH
    python -m pipeline.run ... --publish --log                     # publish, add rows to docs/scale_log.md

Steps run in one Spark session, so the first step's time includes Spark startup.
With --label-version and no --labels, pipeline.labels writes the rule labels
first and the later steps read them. Without either, the steps that need labels
(daily_category, features, baselines) are skipped. Use --log only for deliberate measured runs, so test reruns don't
end up in the scale log.
"""
import argparse
import os
import time
from datetime import date
from pathlib import Path

from pipeline import aggregate, baselines, checks, clean, config, features
from pipeline import labels as rule_labels
from pipeline.config import get_spark
from pipeline.manifest import describe_output
from pipeline.profile_raw import _bytes as _gb_mb, append_scale_log, parquet_bytes

TOTAL_RAW_FILES = 663


def _bytes(n: int) -> str:
    return f"{n / 1e3:.0f} KB" if n < 1e6 else _gb_mb(n)


def _out(path: Path) -> dict:
    d = describe_output(path)
    parts = len([p for p in path.iterdir() if p.is_dir()]) if path.is_dir() else 0
    d["partitions"] = f"{parts} app_id, {d['files']} files" if parts else f"{d['files']} files"
    return d


def log_row(stage: str, source: str, columns: str, compressed: str, uncompressed: str,
            rows_read: int, out: dict, filters: str, runtime: float, notes: str) -> str:
    """One row in the docs/scale_log.md column order."""
    spark = get_spark()
    cells = [
        date.today().isoformat(), stage, source, columns, compressed, uncompressed,
        f"{rows_read:,}", f"{out['rows']:,}", filters, out["partitions"], f"{runtime:.1f}s",
        _bytes(out["bytes"]),
        f"local[{os.cpu_count()}], driver {spark.conf.get('spark.driver.memory')}",
        f"{notes}; snapshot {config.SNAPSHOT_ID[:7]}",
    ]
    return "| " + " | ".join(cells) + " |"


def timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def main(labels: Path | None, label_version: str | None, publish: bool, log: bool) -> None:
    rows: list[str] = []
    apps = list(config.GAMES)
    staging = config.STAGING_DIR

    raw_files = sorted(config.RAW_DIR.glob("part.*.parquet"))
    if len(raw_files) < TOTAL_RAW_FILES:
        print(f"Note: {len(raw_files)} of {TOTAL_RAW_FILES} raw files present.")
    sec = timed(clean.main, apps)
    raw = parquet_bytes(raw_files, config.RAW_COLUMNS)
    out = _out(staging / "reviews_clean")
    rows.append(log_row("clean", f"{len(raw_files)} part files", f"{len(config.RAW_COLUMNS)} of 25",
                        _bytes(raw["compressed_selected"]), _bytes(raw["uncompressed_selected"]),
                        raw["rows"], out, "5 games, window, unedited, valid rows", sec,
                        f"includes Spark startup; quarantine in data/quarantine/"))

    if label_version and labels is None:
        src = _out(staging / "reviews_clean")
        sec = timed(rule_labels.main, label_version)
        labels = rule_labels.OUT_DIR
        rows.append(log_row("labels", "reviews_clean", "review_id, text_normalized", _bytes(src["bytes"]), "-",
                            src["rows"], _out(labels / f"label_version={label_version}"),
                            "English negatives", sec, f"{label_version} rules"))

    sec = timed(aggregate.main, labels, label_version)
    src = _out(staging / "reviews_clean")
    out = _out(staging / "daily_negative")
    rows.append(log_row("daily", "reviews_clean", "all", _bytes(src["bytes"]), "-", src["rows"], out,
                        "none", sec, "daily_negative" + (f" + daily_category ({label_version})" if labels else "")))

    if labels:
        src = _out(staging / "daily_category")
        sec = timed(features.main, label_version)
        rows.append(log_row("features", "daily_category", "all", _bytes(src["bytes"]), "-", src["rows"],
                            _out(staging / "forecast_features"), f"label_version {label_version}", sec,
                            f"{len(config.FEATURE_COLS)} features"))
        src = _out(staging / "forecast_features")
        sec = timed(baselines.main)
        rows.append(log_row("baselines", "forecast_features", "all", _bytes(src["bytes"]), "-", src["rows"],
                            _out(staging / "forecast_results"), "none", sec, "persistence + rolling-28"))
    else:
        print("No --labels: skipped daily_category, features and baselines.")

    sec = timed(checks.main, publish)
    print(f"checks{' + publish' if publish else ''}: {sec:.1f}s")

    print("\nScale log rows:")
    for r in rows:
        print(r)
    if log:
        for r in rows:
            append_scale_log(r)
        print(f"Appended {len(rows)} rows to docs/scale_log.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-version")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--log", action="store_true", help="append the rows to docs/scale_log.md")
    args = parser.parse_args()
    if args.labels and not args.label_version:
        parser.error("--label-version is required with --labels")
    main(args.labels, args.label_version, args.publish, args.log)
