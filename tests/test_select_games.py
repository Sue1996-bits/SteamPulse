import pandas as pd
import pytest

from pipeline import select_games

START, END = "2023-01-01", "2023-01-31"


def daily_rows(appid, per_day, start="2022-11-01", end=END, skip=()):
    days = pd.date_range(start, end)
    return [
        {"appid": appid, "date": d.date(), "all_reviews": per_day * 3,
         "en_neg": per_day + 2, "en_neg_strict": per_day}
        for d in days if d.strftime("%Y-%m-%d") not in skip
    ]


@pytest.fixture
def daily(monkeypatch):
    monkeypatch.setattr(select_games, "MIN_TRAIN_STRICT", 10)
    gap_days = {f"2023-01-{d:02d}" for d in range(10, 13)}  # 3-day source gap
    return pd.DataFrame(
        daily_rows(1.0, 40)                    # steady and busy: always eligible
        + daily_rows(2.0, 5)                   # N7 = 35 < 50: never eligible
        + daily_rows(3.0, 10, skip=gap_days)   # N7 = 70 normally, 40 across the gap
        + daily_rows(4.0, 40, start="2023-01-28")  # released too late for a 28-day baseline
    )


def test_eligibility(daily):
    e = select_games.train_eligibility(daily, START, END).set_index("appid")
    assert e.loc[1.0, "pct_eligible"] == 100
    assert e.loc[1.0, "per_day"] == 40
    assert e.loc[1.0, "edit_share"] == pytest.approx(1 - 40 / 42)
    assert e.loc[2.0, "pct_eligible"] == 0
    assert e.loc[3.0, "zero_days"] == 3
    assert 0 < e.loc[3.0, "pct_eligible"] < 100  # gap days drop N7 below 50
    assert e.loc[4.0, "pct_eligible"] == 0       # never has 28 days of baseline


def test_uses_history_before_start(daily):
    """Origins at the start of training count history from before it."""
    e = select_games.train_eligibility(daily, START, END).set_index("appid")
    assert e.loc[1.0, "pct_eligible"] == 100  # Jan 1 already has 34 prior days


def test_tiers():
    df = pd.DataFrame({"pct_eligible": [100, 60, 100, 10], "zero_days": [0, 0, 20, 0]})
    assert list(select_games.assign_tier(df)) == ["core", "near_miss", "near_miss", "excluded"]


def test_keywords_match_expected_phrases():
    import re
    kw = select_games.KEYWORDS
    assert re.search(kw["crash"], "game keeps crashing on startup")
    assert re.search(kw["performance"], "terrible fps and stuttering")
    assert re.search(kw["match_server"], "servers are down, can't find a match in matchmaking")
    assert re.search(kw["monetization"], "pay to win battle pass garbage")
    assert not re.search(kw["performance"], "great story, bad ending")
