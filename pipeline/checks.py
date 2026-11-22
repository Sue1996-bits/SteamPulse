"""Data quality checks on staging output, and the publish step.

    python -m pipeline.checks             # check whatever tables exist in data/staging/
    python -m pipeline.checks --publish   # check, then copy to data/published/ if all pass

Each table that exists gets its checks: schema, unique key, allowed values, nulls
and ranges, plus cross-table checks (the daily grid, and counts reconciled with
clean_summary.json). Results go to data/staging/check_report.json.

Publishing copies every checked table to data/published/ and writes
manifest.json there. If any check fails, nothing is copied.
"""
import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pipeline import config, manifest
from pipeline.config import get_spark

REPORT_PATH = config.STAGING_DIR / "check_report.json"
TABLES = {
    "reviews_clean": (config.REVIEWS_CLEAN_COLS, config.REVIEWS_CLEAN_KEY),
    "daily_negative": (config.DAILY_NEGATIVE_COLS, config.DAILY_NEGATIVE_KEY),
    "daily_category": (config.DAILY_CATEGORY_COLS, config.DAILY_CATEGORY_KEY),
    "forecast_features": (config.FORECAST_FEATURES_COLS, config.FORECAST_FEATURES_KEY),
    "forecast_results": (config.FORECAST_RESULTS_COLS, config.FORECAST_RESULTS_KEY),
}
EXTRA_FILES = ["feature_manifest.json", "clean_summary.json", "baseline_scores.csv"]


@dataclass
class Check:
    table: str
    name: str
    passed: bool
    detail: str = ""


def _bad(df: DataFrame, condition) -> int:
    return df.where(condition).count()


def count_check(table: str, name: str, bad: int, what: str) -> Check:
    return Check(table, name, bad == 0, f"{bad:,} {what}" if bad else "")


# ---- generic checks ---------------------------------------------------------------

def schema(df: DataFrame, table: str, cols: list[str]) -> Check:
    missing, extra = set(cols) - set(df.columns), set(df.columns) - set(cols)
    detail = "; ".join(s for s in (f"missing {sorted(missing)}" if missing else "",
                                   f"unexpected {sorted(extra)}" if extra else "") if s)
    return Check(table, "schema", not detail, detail)


def unique_key(df: DataFrame, table: str, key: list[str]) -> Check:
    dupes = df.groupBy(*key).count().where("count > 1").count()
    return count_check(table, "unique_key", dupes, f"keys appear more than once ({'+'.join(key)})")


def not_null(df: DataFrame, table: str, cols: list[str]) -> Check:
    cond = F.lit(False)
    for c in cols:
        cond = cond | F.col(c).isNull()
    return count_check(table, "key_not_null", _bad(df, cond), f"rows with a null in {cols}")


def allowed(df: DataFrame, table: str, col: str, values: list[str]) -> Check:
    bad = _bad(df, ~F.col(col).isin(values) | F.col(col).isNull())
    return count_check(table, f"{col}_allowed", bad, f"rows outside {values}")


def in_unit_range(df: DataFrame, table: str, cols: list[str]) -> Check:
    cond = F.lit(False)
    for c in cols:
        cond = cond | (F.col(c) < 0) | (F.col(c) > 1)
    return count_check(table, "shares_in_0_1", _bad(df, cond), f"values outside [0, 1] in {cols}")


# ---- table-specific checks --------------------------------------------------------

def check_reviews_clean(df: DataFrame) -> list[Check]:
    t = "reviews_clean"
    window = (F.col("review_date") < F.lit(config.WINDOW_START).cast("date")) | \
             (F.col("review_date") > F.lit(config.WINDOW_END).cast("date"))
    return [
        count_check(t, "in_window", _bad(df, window), "reviews outside the analysis window"),
        count_check(t, "strict_cohort", _bad(df, F.col("created_at") != F.col("updated_at")), "edited reviews"),
        count_check(t, "has_text", _bad(df, F.trim("text_raw") == ""), "empty reviews"),
    ]


def check_daily(neg: DataFrame, cat: DataFrame | None) -> list[Check]:
    out = [allowed(neg, "daily_negative", "coverage_status", config.COVERAGE_STATUSES)]
    gap, covered = F.col("coverage_status") == "gap", F.col("coverage_status") == "covered"
    out.append(count_check("daily_negative", "gap_is_null",
                           _bad(neg, (gap & F.col("negative_total").isNotNull())
                                | (covered & F.col("negative_total").isNull())),
                           "gap days with counts or covered days without"))
    span = neg.groupBy("app_id").agg(F.count("*").alias("n"),
                                     (F.datediff(F.max("date"), F.min("date")) + 1).alias("days"))
    out.append(count_check("daily_negative", "full_grid", _bad(span, F.col("n") != F.col("days")),
                           "games with missing or repeated days"))
    if cat is None:
        return out

    t = "daily_category"
    out.append(allowed(cat, t, "coverage_status", config.COVERAGE_STATUSES))
    out.append(allowed(cat, t, "category", config.CATEGORIES))
    out.append(in_unit_range(cat, t, ["share"]))
    out.append(count_check(t, "count_le_total",
                           _bad(cat, F.col("category_count") > F.col("negative_total")),
                           "rows where category_count > negative_total"))
    per_day = cat.groupBy("app_id", "date", "label_version").count()
    out.append(count_check(t, "four_categories", _bad(per_day, F.col("count") != len(config.CATEGORIES)),
                           "game-days without exactly one row per category"))
    mismatch = (cat.select("app_id", "date", F.col("negative_total").alias("cat_n"),
                           F.col("coverage_status").alias("cat_status"))
                .join(neg, ["app_id", "date"], "full")
                .where(~F.col("cat_n").eqNullSafe(F.col("negative_total"))
                       | ~F.col("cat_status").eqNullSafe(F.col("coverage_status"))))
    out.append(count_check(t, "matches_daily_negative", mismatch.count(),
                           "rows whose total or coverage differs from daily_negative"))
    return out


def check_features(df: DataFrame) -> list[Check]:
    t = "forecast_features"
    eligible = F.col("eligibility") == "eligible"
    required = ["c_7d", "n_7d", "share_7d", "share_28d", "baseline_share", "lift_pp"]
    missing = F.lit(False)
    for c in required:
        missing = missing | F.col(c).isNull()
    return [
        allowed(df, t, "eligibility", config.ELIGIBILITY_VALUES),
        Check(t, "label_not_a_feature", not set(config.TARGET_COLS) & set(config.FEATURE_COLS)),
        count_check(t, "no_future_dates", _bad(df, F.col("feature_max_date") > F.col("as_of_date")),
                    "rows using dates after as_of_date"),
        count_check(t, "eligible_rows_complete", _bad(df, eligible & missing),
                    f"eligible rows with a null in {required}"),
        in_unit_range(df, t, ["share_7d", "share_14d", "share_28d", "share_prior7d",
                              "baseline_share", config.LABEL_COL]),
    ]


def check_results(df: DataFrame) -> list[Check]:
    t = "forecast_results"
    ok = F.col("forecast_status") == "ok"
    return [
        allowed(df, t, "observed_status", config.OBSERVED_STATUSES),
        allowed(df, t, "forecast_status", config.FORECAST_STATUSES),
        in_unit_range(df, t, ["predicted_share", "recent_share", "baseline_share"]),
        count_check(t, "prediction_iff_ok",
                    _bad(df, (ok & F.col("predicted_share").isNull())
                         | (~ok & F.col("predicted_share").isNotNull())),
                    "rows where a prediction and forecast_status disagree"),
    ]


def reconcile(clean: DataFrame, neg: DataFrame | None, summary: dict) -> list[Check]:
    """reviews_clean and daily_negative agree with each other and with clean_summary.json."""
    games = summary.get("games", {})
    rows = {r.app_id: r["count"] for r in clean.groupBy("app_id").count().collect()}
    bad = [a for a, g in games.items() if a in rows and rows[a] != g["reviews_clean_rows"]]
    out = [Check("reviews_clean", "matches_clean_summary", not bad, f"row counts differ for {bad}" if bad else "")]
    if neg is not None:
        totals = {r.app_id: r.n for r in neg.groupBy("app_id").agg(F.sum("negative_total").alias("n")).collect()}
        bad = [a for a, g in games.items() if a in totals and totals[a] != g["english_negatives"]]
        out.append(Check("daily_negative", "matches_clean_summary", not bad,
                         f"negative totals differ for {bad}" if bad else ""))
    return out


# ---- runner ---------------------------------------------------------------------

def load(spark: SparkSession) -> dict[str, DataFrame]:
    return {name: spark.read.parquet(str(config.STAGING_DIR / name))
            for name in TABLES if (config.STAGING_DIR / name).exists()}


def run_checks(tables: dict[str, DataFrame], summary: dict | None = None) -> list[Check]:
    results: list[Check] = []
    for name, df in tables.items():
        cols, key = TABLES[name]
        results += [schema(df, name, cols), not_null(df, name, key), unique_key(df, name, key)]
    if "reviews_clean" in tables:
        results += check_reviews_clean(tables["reviews_clean"])
        if summary:
            results += reconcile(tables["reviews_clean"], tables.get("daily_negative"), summary)
    if "daily_negative" in tables:
        results += check_daily(tables["daily_negative"], tables.get("daily_category"))
    if "forecast_features" in tables:
        results += check_features(tables["forecast_features"])
    if "forecast_results" in tables:
        results += check_results(tables["forecast_results"])
    return results


def publish(table_names: list[str], label_version: str | None) -> None:
    config.PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in table_names:
        dest = config.PUBLISH_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(config.STAGING_DIR / name, dest, ignore=shutil.ignore_patterns("*.crc", "_SUCCESS"))
        outputs[name] = dest
    for f in EXTRA_FILES:
        if (config.STAGING_DIR / f).exists():
            shutil.copy2(config.STAGING_DIR / f, config.PUBLISH_DIR / f)
    path = manifest.write_manifest(outputs, out_dir=config.PUBLISH_DIR, label_version=label_version)
    print(f"Published {', '.join(table_names)} -> {config.PUBLISH_DIR}; wrote {path.name}")


def main(do_publish: bool) -> None:
    spark = get_spark("steampulse-checks")
    tables = load(spark)
    if not tables:
        raise SystemExit("Nothing in data/staging/ to check.")
    summary_path = config.STAGING_DIR / "clean_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    results = run_checks(tables, summary)

    failed = [c for c in results if not c.passed]
    REPORT_PATH.write_text(json.dumps({
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tables": sorted(tables), "passed": not failed,
        "checks": [asdict(c) for c in results],
    }, indent=2) + "\n")
    for c in results:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.table:<18} {c.name:<24} {c.detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed -> {REPORT_PATH.relative_to(config.ROOT)}")

    if failed:
        raise SystemExit("Checks failed; nothing published." if do_publish else 1)
    if do_publish:
        versions = {r.label_version for r in tables["daily_category"].select("label_version").distinct().collect()} \
            if "daily_category" in tables else set()
        publish(sorted(tables), ",".join(sorted(versions)) or None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publish", action="store_true", help="copy to data/published/ if all checks pass")
    main(parser.parse_args().publish)
