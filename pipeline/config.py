"""Shared settings, paths and schema v1 for the SteamPulse pipeline.

The column lists are the shared data contract for the pipeline, model and app.
Changing one is a contract change: open an issue and get it reviewed.
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
PROFILE_DIR = DATA / "profile"

HF_DATASET = "adde023/steam-reviews"
# Hugging Face commit hash; every download is pinned to it.
SNAPSHOT_ID = "d0ec12b33dffb8109b683b2e07780e2fc6b97187"

# Raw columns the pipeline reads. Author/user columns are deliberately excluded.
RAW_COLUMNS = [
    "recommendationid", "appid", "game", "language", "review",
    "timestamp_created", "timestamp_updated", "voted_up",
]

# Analysis window (docs/temporal_policy.md). The snapshot ends 2023-11-03 and its
# last days are incomplete, so the window stops at 2023-10-31.
WINDOW_START = "2022-11-01"
WINDOW_END = "2023-10-31"
# End of the training period (~60% of the window). Game selection only looks
# at data up to here.
TRAIN_END = "2023-05-31"

# Games to model (see docs/temporal_policy.md, "Chosen games"). Apex is built first.
GAMES = {
    "1172470": "Apex Legends",
    "1517290": "Battlefield 2042",
    "1811260": "FIFA 23",
    "1085660": "Destiny 2",
    "359550": "Rainbow Six Siege",
}

LANGUAGE = "english"
# Complaint taxonomy. "unclassified" is a coverage status, not a modeled category.
CATEGORIES = ["crash", "performance", "match_server", "monetization"]

# Versions are tracked independently.
COHORT_POLICY = "strict_unedited"  # created_at == updated_at
COHORT_VERSION = "v1"
LABEL_VERSION = None  # set by the labeling step
FEATURE_VERSION = "v1"

# Detection and forecast thresholds. Provisional: tune on validation only, freeze by Nov 22.
HISTORY_DAYS = 35
MIN_N7 = 50
MIN_N28 = 100
MIN_C7 = 10        # observed alerts only; forecasting does not require it
MIN_LIFT = 0.05    # S7 - S28 as a share (= 5 percentage points)

# --- Schema v1 ----------------------------------------------------------------
# reviews_clean, key: review_id
REVIEWS_CLEAN_COLS = [
    "review_id", "app_id", "text_raw", "text_normalized",
    "created_at", "updated_at", "available_at", "review_date",
    "language", "voted_up", "snapshot_id", "cohort_policy",
]
# daily_negative, key: app_id + date + cohort_version
DAILY_NEGATIVE_COLS = [
    "app_id", "date", "cohort_version", "negative_total", "coverage_status", "snapshot_id",
]
# daily_category, key: app_id + date + category + versions; 4 rows per covered date
DAILY_CATEGORY_COLS = [
    "app_id", "date", "category", "category_count", "negative_total", "share",
    "label_version", "cohort_version",
]
# forecast_features, key: app_id + as_of_date + category + feature_version
FORECAST_FEATURES_KEY = ["app_id", "as_of_date", "category", "feature_version"]
FORECAST_FEATURES_META = ["eligibility", "feature_max_date", "source_snapshot", "label_version"]
FEATURE_COLS: list[str] = []  # TODO (features): fixed order, exported as feature_manifest.json
LABEL_COL = "target_share_next7"  # never a feature


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
