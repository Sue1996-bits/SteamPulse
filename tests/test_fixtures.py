"""Checks on the shared test fixture (tests/fixtures/).

These confirm the fixture files agree with the schemas in config.py and with
each other. Tests of the pipeline itself (aggregate, features, baselines) use
the same files once those steps are implemented.
"""
from pathlib import Path

import pandas as pd
import pytest

from pipeline import config

FIX = Path(__file__).parent / "fixtures"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name, dtype={"app_id": str, "review_id": str})


@pytest.fixture(scope="module")
def reviews():
    return read("reviews_clean.csv")


@pytest.fixture(scope="module")
def labels():
    return read("review_labels.csv")


def test_columns_match_schema(reviews, labels):
    assert list(reviews.columns) == config.REVIEWS_CLEAN_COLS
    assert list(labels.columns) == config.REVIEW_LABELS_COLS
    daily_cat = read("expected_daily_category.csv")
    assert set(daily_cat.columns) <= set(config.DAILY_CATEGORY_COLS)
    daily_neg = read("expected_daily_negative.csv")
    assert set(daily_neg.columns) <= set(config.DAILY_NEGATIVE_COLS)


def test_fixture_covers_edge_cases(reviews, labels):
    keyed = labels.groupby(config.REVIEW_LABELS_KEY).size()
    assert (keyed > 1).any(), "needs a duplicate label row"
    per_review = labels.drop_duplicates(config.REVIEW_LABELS_KEY).groupby("review_id").size()
    assert (per_review > 1).any(), "needs a multi-label review"
    en_neg = reviews[(reviews.language == "english") & (~reviews.voted_up)]
    assert (~en_neg.review_id.isin(labels.review_id)).any(), "needs unclassified negatives"
    assert (reviews.language != "english").any() and reviews.voted_up.any()
    daily_neg = read("expected_daily_negative.csv")
    assert (daily_neg.coverage_status == "gap").any()
    assert ((daily_neg.coverage_status == "covered") & (daily_neg.negative_total == 0)).any()
    late = reviews[reviews.created_at.str.contains("T23:59:59")]
    assert len(late) and (late.created_at.str[:10] == late.review_date).all()


def test_expected_daily_matches_inputs(reviews, labels):
    """Recompute daily counts from the raw fixture rows in a second, simple way."""
    en_neg = reviews[(reviews.language == "english") & (~reviews.voted_up)]
    n = en_neg.groupby(["app_id", "review_date"]).size()
    lab = labels.drop_duplicates(config.REVIEW_LABELS_KEY).merge(
        en_neg[["review_id", "app_id", "review_date"]], on="review_id")
    c = lab.groupby(["app_id", "review_date", "category"]).size()

    expected = read("expected_daily_category.csv")
    daily_neg = read("expected_daily_negative.csv")
    status = dict(zip(zip(daily_neg.app_id, daily_neg.date), daily_neg.coverage_status))
    for r in expected.itertuples():
        assert r.coverage_status == status[(r.app_id, r.date)]
        if r.coverage_status == "gap":
            assert pd.isna(r.category_count) and pd.isna(r.negative_total) and pd.isna(r.share)
            assert not ((reviews.app_id == r.app_id) & (reviews.review_date == r.date)).any()
            continue
        assert r.negative_total == n.get((r.app_id, r.date), 0)
        assert r.category_count == c.get((r.app_id, r.date, r.category), 0)
        assert r.category_count <= r.negative_total
        if r.negative_total == 0:
            assert pd.isna(r.share)
        else:
            assert r.share == pytest.approx(r.category_count / r.negative_total)


def test_daily_tables_have_a_full_grid():
    """Every game-day has one daily_negative row and four daily_category rows, gaps included."""
    daily_neg = read("expected_daily_negative.csv")
    daily_cat = read("expected_daily_category.csv")
    assert not daily_neg.duplicated(["app_id", "date"]).any()
    for app, g in daily_neg.groupby("app_id"):
        dates = pd.to_datetime(g.date)
        assert len(dates) == (dates.max() - dates.min()).days + 1, f"missing days for {app}"
    per_day = daily_cat.groupby(["app_id", "date"]).category.agg(sorted)
    assert len(per_day) == len(daily_neg)
    assert all(cats == sorted(config.CATEGORIES) for cats in per_day)


def test_expected_windows_match_daily():
    """Windows are ratios of sums over the daily table, never averages of daily shares."""
    daily = read("expected_daily_category.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    windows = read("expected_windows.csv")
    for w in windows.itertuples():
        t = pd.Timestamp(w.as_of_date)
        g = daily[(daily.app_id == w.app_id) & (daily.category == w.category)].set_index("date")
        recent = g.loc[t - pd.Timedelta(days=6): t]
        base = g.loc[t - pd.Timedelta(days=34): t - pd.Timedelta(days=7)]
        assert recent.category_count.sum() == w.C7 and recent.negative_total.sum() == w.N7
        assert w.recent_share == pytest.approx(w.C7 / w.N7)
        if base.negative_total.isna().any():
            assert pd.isna(w.baseline_share) and w.observed_status == "insufficient_data"
            assert w.forecast_status == "insufficient_data" and pd.isna(w.persistence_pred)
            continue
        assert base.category_count.sum() == w.C28 and base.negative_total.sum() == w.N28
        assert w.baseline_share == pytest.approx(w.C28 / w.N28)
        assert w.observed_lift_pp == pytest.approx(100 * (w.recent_share - w.baseline_share))
        roll = g.loc[t - pd.Timedelta(days=27): t]
        assert w.rolling28_pred == pytest.approx(roll.category_count.sum() / roll.negative_total.sum())
        assert w.persistence_pred == pytest.approx(w.recent_share)
        assert w.observed_status in config.OBSERVED_STATUSES
        assert w.forecast_status in config.FORECAST_STATUSES


def test_averaging_daily_shares_gives_the_wrong_answer():
    """The crash spike is where a mean of daily shares and a ratio of sums disagree."""
    daily = read("expected_daily_category.csv").query("app_id == '999' and category == 'crash'")
    last7 = daily.tail(7)
    assert last7.share.mean() == pytest.approx((6 * 0.10 + 0.50) / 7)  # 0.157, wrong
    assert last7.category_count.sum() / last7.negative_total.sum() == pytest.approx(0.26)  # right
