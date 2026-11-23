from datetime import date, timedelta

import pytest

from pipeline import label_sample
from pipeline.config import get_spark


@pytest.fixture(scope="module")
def clean():
    spark = get_spark("test")
    rows = []
    for i in range(200):
        d = date(2023, 1, 1) + timedelta(days=i % 150)  # Jan 1 to May 30
        rows.append((f"{i}", "1172470", d, "english", False))           # train-period negatives
        rows.append((f"L{i}", "1172470", date(2023, 8, 1), "english", False))  # later-period negatives
    rows += [
        ("pos", "1172470", date(2023, 1, 5), "english", True),      # positive: never sampled
        ("de", "1172470", date(2023, 1, 5), "german", False),       # not English: never sampled
        ("old", "1172470", date(2022, 6, 1), "english", False),     # before the window
        ("edge", "1172470", date(2023, 5, 31), "english", False),   # last training day
    ]
    return spark.createDataFrame(rows, ["review_id", "app_id", "review_date", "language", "voted_up"])


def test_exact_sizes_and_periods(clean):
    s = label_sample.draw(clean, 30, 15, seed=1).toPandas()
    assert s.period.value_counts().to_dict() == {"train": 30, "later": 15}
    assert not s.review_id.isin(["pos", "de", "old"]).any()
    assert (s[s.period == "train"].review_date <= date(2023, 5, 31)).all()
    assert (s[s.period == "later"].review_date > date(2023, 5, 31)).all()


def test_same_seed_same_sample(clean):
    a = set(label_sample.draw(clean, 30, 15, seed=1).toPandas().review_id)
    b = set(label_sample.draw(clean, 30, 15, seed=1).toPandas().review_id)
    c = set(label_sample.draw(clean, 30, 15, seed=2).toPandas().review_id)
    assert a == b and a != c


def test_small_pool_returns_everything(clean):
    s = label_sample.draw(clean, 10_000, 10_000, seed=1).toPandas()
    assert (s.period == "later").sum() == 200
    assert "edge" in set(s[s.period == "train"].review_id)
