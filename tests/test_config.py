from pipeline import config


def test_four_categories():
    assert config.CATEGORIES == ["crash", "performance", "match_server", "monetization"]


def test_schema_columns_unique():
    for cols in (config.REVIEWS_CLEAN_COLS, config.DAILY_NEGATIVE_COLS, config.DAILY_CATEGORY_COLS,
                 config.REVIEW_LABELS_COLS, config.FORECAST_RESULTS_COLS):
        assert len(cols) == len(set(cols))


def test_keys_are_in_their_tables():
    pairs = [
        (config.REVIEWS_CLEAN_KEY, config.REVIEWS_CLEAN_COLS),
        (config.REVIEW_LABELS_KEY, config.REVIEW_LABELS_COLS),
        (config.DAILY_NEGATIVE_KEY, config.DAILY_NEGATIVE_COLS),
        (config.DAILY_CATEGORY_KEY, config.DAILY_CATEGORY_COLS),
        (config.FORECAST_RESULTS_KEY, config.FORECAST_RESULTS_COLS),
    ]
    for key, cols in pairs:
        assert set(key) <= set(cols)


def test_results_keep_observed_and_forecast_status_separate():
    assert "observed_status" in config.FORECAST_RESULTS_COLS
    assert "forecast_status" in config.FORECAST_RESULTS_COLS
    assert "status" not in config.FORECAST_RESULTS_COLS


def test_label_is_not_a_feature():
    assert config.LABEL_COL not in config.FEATURE_COLS


def test_spark_uses_utc():
    spark = config.get_spark("test")
    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"
