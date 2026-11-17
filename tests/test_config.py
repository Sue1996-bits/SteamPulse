from pipeline import config


def test_four_categories():
    assert config.CATEGORIES == ["crash", "performance", "match_server", "monetization"]


def test_schema_columns_unique():
    for cols in (config.REVIEWS_CLEAN_COLS, config.DAILY_NEGATIVE_COLS, config.DAILY_CATEGORY_COLS):
        assert len(cols) == len(set(cols))


def test_label_is_not_a_feature():
    assert config.LABEL_COL not in config.FEATURE_COLS


def test_spark_uses_utc():
    spark = config.get_spark("test")
    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"
