"""Data quality checks on staging output, and the publish step.

    python -m pipeline.checks             # check whatever tables exist in data/staging/
    python -m pipeline.checks --publish   # check, then copy to data/published/ if all pass

Each table that exists gets its checks: schema, unique key, allowed values, nulls
and ranges, plus cross-table checks (the daily grid, and counts reconciled with
clean_summary.json). Results go to data/staging/check_report.json.

Publishing empties data/published/, then copies every checked table there and
writes manifest.json, so the folder only ever holds one run. review_labels keeps
only the label_version being published (staging can hold several side by side).
If any check fails, nothing is copied.
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
    "review_labels": (config.REVIEW_LABELS_COLS, config.REVIEW_LABELS_KEY),
    "daily_negative": (config.DAILY_NEGATIVE_COLS, config.DAILY_NEGATIVE_KEY),
    "daily_category": (config.DAILY_CATEGORY_COLS, config.DAILY_CATEGORY_KEY),
    "forecast_features": (config.FORECAST_FEATURES_COLS, config.FORECAST_FEATURES_KEY),
    "forecast_results": (config.FORECAST_RESULTS_COLS, config.FORECAST_RESULTS_KEY),
    "model_predictions": (config.FORECAST_RESULTS_COLS, config.FORECAST_RESULTS_KEY),
}
# baseline_scores.csv stays in staging: its "later" period mixes validation and test.
# model_scores.csv has the same baselines on the train/validation/test splits.
EXTRA_FILES = ["feature_manifest.json", "clean_summary.json",
               "model_manifest.json", "model_parity.csv", "model_scores.csv", "model_row_counts.csv",
               "model_errors.csv", "model_feature_importance.csv", "model_card.md",
               "alert_summary.csv", "alert_sensitivity.csv"]
EXTRA_DIRS = ["model"]  # saved Spark models, loaded by the API through model_manifest.json


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


def check_labels(labels: DataFrame, clean: DataFrame | None) -> list[Check]:
    t = "review_labels"
    out = [
        allowed(labels, t, "category", config.CATEGORIES),
        count_check(t, "is_match_true", _bad(labels, ~F.col("is_match") | F.col("is_match").isNull()),
                    "rows with is_match not true (absence of a row means no label)"),
        count_check(t, "has_evidence", _bad(labels, F.col("rule_id").isNull() | (F.col("evidence") == "")),
                    "rows without a rule_id or evidence"),
    ]
    if clean is not None:
        en_neg = clean.where((F.col("language") == config.LANGUAGE) & ~F.col("voted_up")).select("review_id")
        stray = labels.select("review_id").distinct().join(en_neg, "review_id", "left_anti").count()
        out.append(count_check(t, "on_english_negatives", stray,
                               "labeled review_ids that aren't English negatives in reviews_clean"))
    return out


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


def check_results(df: DataFrame, t: str = "forecast_results") -> list[Check]:
    ok = F.col("forecast_status") == "ok"
    out = []
    if t == "model_predictions":
        out.append(count_check(t, "trained_before_as_of",
                               _bad(df, F.col("training_label_end").isNull()
                                    | (F.col("training_label_end") > F.col("as_of_date"))),
                               "rows predicted by a model trained on outcomes after as_of_date"))
    return out + [
        allowed(df, t, "observed_status", config.OBSERVED_STATUSES),
        allowed(df, t, "forecast_status", config.FORECAST_STATUSES),
        in_unit_range(df, t, ["predicted_share", "recent_share", "baseline_share"]),
        count_check(t, "prediction_iff_ok",
                    _bad(df, (ok & F.col("predicted_share").isNull())
                         | (~ok & F.col("predicted_share").isNotNull())),
                    "rows where a prediction and forecast_status disagree"),
    ]


def check_model_bundle(model: dict, features: dict | None, preds: DataFrame | None, root) -> list[Check]:
    """The model bundle matches the feature manifest, its files exist, and predictions come from it."""
    t = "model_bundle"
    fmt_ok, order_ok = model.get("format") == "spark_pipeline", model.get("features") == config.FEATURE_COLS
    out = [Check(t, "format", fmt_ok, "" if fmt_ok else f"format {model.get('format')!r}, expected 'spark_pipeline'"),
           Check(t, "feature_order", order_ok, "" if order_ok else "features differ from FEATURE_COLS")]
    if features is not None:
        diff = [k for k in ("feature_version", "label_version", "snapshot_id") if model.get(k) != features.get(k)]
        out.append(Check(t, "matches_feature_manifest", not diff, f"differs in {diff}" if diff else ""))
    cats = model.get("categories", {})
    missing = [c for c in config.CATEGORIES if not (root / cats.get(c, {}).get("path", "?")).is_dir()]
    out.append(Check(t, "models_present", not missing, f"missing models for {missing}" if missing else ""))
    if preds is not None:
        other = _bad(preds, (F.col("model_version") != model.get("model_version"))
                     | (F.col("training_label_end") != F.lit(model.get("training_label_end")).cast("date")))
        out.append(count_check(t, "predictions_from_this_model", other, "model_predictions rows from another model"))
    return out


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
    if "review_labels" in tables:
        results += check_labels(tables["review_labels"], tables.get("reviews_clean"))
    if "daily_negative" in tables:
        results += check_daily(tables["daily_negative"], tables.get("daily_category"))
    if "forecast_features" in tables:
        results += check_features(tables["forecast_features"])
    if "forecast_results" in tables:
        results += check_results(tables["forecast_results"])
    if "model_predictions" in tables:
        results += check_results(tables["model_predictions"], "model_predictions")
    return results


SKIP = shutil.ignore_patterns("*.crc", "_SUCCESS")


def _copy_labels(src, dest, label_version: str | None) -> None:
    """Copy only the published label_version folder(s) of review_labels."""
    wanted = [src / f"label_version={v}" for v in (label_version or "").split(",") if v]
    if not wanted or not all(w.is_dir() for w in wanted):
        shutil.copytree(src, dest, ignore=SKIP)  # not partitioned by version, or no version given
        return
    for w in wanted:
        shutil.copytree(w, dest / w.name, ignore=SKIP)


def publish(table_names: list[str], label_version: str | None) -> None:
    if config.PUBLISH_DIR.exists():
        shutil.rmtree(config.PUBLISH_DIR)  # no files left over from an earlier run
    config.PUBLISH_DIR.mkdir(parents=True)
    outputs = {}
    for name in table_names:
        src, dest = config.STAGING_DIR / name, config.PUBLISH_DIR / name
        if name == "review_labels":
            _copy_labels(src, dest, label_version)
        else:
            shutil.copytree(src, dest, ignore=SKIP)
        outputs[name] = dest
    for f in EXTRA_FILES:
        if (config.STAGING_DIR / f).exists():
            shutil.copy2(config.STAGING_DIR / f, config.PUBLISH_DIR / f)
    for d in EXTRA_DIRS:
        if (config.STAGING_DIR / d).exists():
            shutil.copytree(config.STAGING_DIR / d, config.PUBLISH_DIR / d, ignore=SKIP)
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
    bundle = config.STAGING_DIR / "model_manifest.json"
    if bundle.exists():
        fm = config.STAGING_DIR / "feature_manifest.json"
        results += check_model_bundle(json.loads(bundle.read_text()),
                                      json.loads(fm.read_text()) if fm.exists() else None,
                                      tables.get("model_predictions"), config.STAGING_DIR)

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
