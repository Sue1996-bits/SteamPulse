"""Shared settings, paths and schema v1 for the SteamPulse pipeline.

Schema follows section 9 of the project plan (v1.1). Changes to these column
lists are contract changes: open an issue and get B and C to review.
"""
import os
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                 # git-ignored
RAW_DIR = DATA / "raw"
QUARANTINE_DIR = DATA / "quarantine"
STAGING_DIR = DATA / "staging"       # outputs land here until checks pass
PUBLISH_DIR = DATA / "published"

HF_DATASET = "adde023/steam-reviews"
SNAPSHOT_ID = None  # TODO (profiling): dataset commit hash recorded at download time

LANGUAGE = "english"
# Section 4 taxonomy. "unclassified" is a coverage status, not a modeled category.
CATEGORIES = ["crash", "performance", "match_server", "monetization"]

# Versions are tracked independently (section 9).
COHORT_POLICY = "strict_unedited"  # created_at == updated_at
COHORT_VERSION = "v1"
LABEL_VERSION = None  # set by B's labeling step
FEATURE_VERSION = "v1"

# Sections 5-6 thresholds. Provisional: tune on validation only, freeze by Nov 22.
HISTORY_DAYS = 35
MIN_N7 = 50
MIN_N28 = 100
MIN_C7 = 10        # observed alerts only; forecasting does not require it
MIN_LIFT = 0.05    # S7 - S28 as a share (= 5 percentage points)

# --- Schema v1 (section 9) ---------------------------------------------------
# reviews_clean, key: review_id
REVIEWS_CLEAN_COLS = [
    "review_id", "app_id", "text_raw", "text_normalized",
    "created_at", "updated_at", "available_at", "review_date",
    "language", "voted_up", "snapshot_id", "cohort_policy",
]
# daily_negative, key: app_id + date + cohort_version.
# One row per game per day, from the game's first review date to the window end.
# A day with no reviews in any language is a gap (null counts), not a zero.
DAILY_NEGATIVE_COLS = [
    "app_id", "date", "cohort_version", "negative_total", "coverage_status", "snapshot_id",
]
# daily_category, key: app_id + date + category + versions; 4 rows per game-day.
# Gap days keep their rows with null counts, and coverage_status repeats the
# daily_negative value so window code doesn't need a join.
DAILY_CATEGORY_COLS = [
    "app_id", "date", "category", "category_count", "negative_total", "share",
    "coverage_status", "label_version", "cohort_version",
]
# forecast_features, key: app_id + as_of_date + category + feature_version
FORECAST_FEATURES_KEY = ["app_id", "as_of_date", "category", "feature_version"]
FORECAST_FEATURES_META = ["eligibility", "feature_max_date", "source_snapshot", "label_version"]
FEATURE_COLS: list[str] = []  # TODO (features): fixed order, exported as feature_manifest.json
LABEL_COL = "target_share_next7"  # never a feature

# review_labels (labeling step), key: review_id + category + label_version.
# One row per assigned label. No row means the category was not assigned, and the
# review still counts in negative_total (unclassified).
REVIEW_LABELS_KEY = ["review_id", "category", "label_version"]
REVIEW_LABELS_COLS = ["review_id", "category", "is_match", "rule_id", "evidence", "label_version"]

# forecast_results (trained model and baselines), key: app_id + as_of_date + category + method_version
FORECAST_RESULTS_KEY = ["app_id", "as_of_date", "category", "method_version"]
FORECAST_RESULTS_COLS = [
    *FORECAST_RESULTS_KEY,
    "recent_share", "baseline_share", "observed_lift_pp", "observed_status",
    "predicted_share", "predicted_change_pp", "forecast_status", "eligibility_reason",
    "model_version", "training_label_end", "feature_version",
]

# Keys for the other tables (used by quality checks)
REVIEWS_CLEAN_KEY = ["review_id"]
DAILY_NEGATIVE_KEY = ["app_id", "date", "cohort_version"]
DAILY_CATEGORY_KEY = ["app_id", "date", "category", "label_version", "cohort_version"]

# Allowed values
COVERAGE_STATUSES = ["covered", "gap"]
OBSERVED_STATUSES = ["emerging", "not_flagged", "insufficient_data"]
FORECAST_STATUSES = ["ok", "insufficient_data"]


def get_spark(app_name: str = "steampulse") -> SparkSession:
    """Spark session with UTC day boundaries (see docs/temporal_policy.md)."""
    os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
