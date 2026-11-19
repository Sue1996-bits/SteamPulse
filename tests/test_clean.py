from datetime import date

import pytest

from pipeline import clean, config
from pipeline.config import RAW_COLUMNS, get_spark

T0 = 1_672_531_200.0  # 2023-01-01 00:00:00 UTC
DAY = 86_400.0
APEX = 1172470.0


@pytest.fixture(scope="module")
def staged():
    spark = get_spark("test")
    rows = [
        # id, app, game, language, review, created, updated, voted_up
        (1.0, APEX, "Apex", "english", "[b]Crashes[/b] every   match http://x.co/a", T0, T0, 0.0),  # kept
        (2.0, APEX, "Apex", "german", "schlecht", T0 + DAY - 1, T0 + DAY - 1, 0.0),  # kept, last second of Jan 1
        (3.0, APEX, "Apex", "english", "fun", T0, T0, 1.0),                         # kept, positive
        (4.0, APEX, "Apex", "english", "bad", T0, T0 + 500, 0.0),                   # edited
        (5.0, APEX, "Apex", "english", "old", 1_600_000_000.0, 1_600_000_000.0, 0.0),  # 2020: out of window
        (6.0, APEX, "Apex", "english", "   ", T0, T0, 0.0),                         # empty_text
        (7.5, APEX, "Apex", "english", "x", T0, T0, 0.0),                           # bad_id
        (None, APEX, "Apex", "english", "x", T0, T0, 0.0),                          # missing_key
        (8.0, APEX, "Apex", "english", "x", None, None, 0.0),                       # missing_key
        (9.0, APEX, "Apex", "english", "x", T0, T0, None),                          # missing_vote
        (10.0, APEX, "Apex", "english", "lag", T0 + 10, T0 + 10, 0.0),              # kept
        (10.0, APEX, "Apex", "english", "lag", T0 + 10, T0 + 10, 0.0),              # exact duplicate
    ]
    return clean.with_stage(spark.createDataFrame(rows, RAW_COLUMNS))


def test_every_row_gets_the_right_stage(staged):
    got = {(r.recommendationid, r.timestamp_created, r.voted_up): r.stage for r in staged.collect()}
    assert got[(1.0, T0, 0.0)] == "kept"
    assert got[(4.0, T0, 0.0)] == "edited"
    assert got[(5.0, 1_600_000_000.0, 0.0)] == "out_of_window"
    assert got[(6.0, T0, 0.0)] == "empty_text"
    assert got[(7.5, T0, 0.0)] == "bad_id"
    assert got[(None, T0, 0.0)] == "missing_key"
    assert got[(8.0, None, 0.0)] == "missing_key"
    assert got[(9.0, T0, None)] == "missing_vote"


def test_stage_counts_add_up(staged):
    counts = clean.stage_counts(staged)["1172470"]
    assert set(counts) == set(clean.STAGES)
    assert sum(counts.values()) == staged.count()
    assert counts["kept"] == 5 and counts["missing_key"] == 2


def test_clean_output_matches_schema(staged):
    out = clean.to_clean(staged.where("stage = 'kept'"))
    assert out.columns == config.REVIEWS_CLEAN_COLS
    rows = {r.review_id: r for r in out.collect()}
    assert sorted(rows) == ["1", "10", "2", "3"]  # duplicate of 10 dropped; IDs are strings, not "1.0"
    r1 = rows["1"]
    assert r1.app_id == "1172470"
    assert r1.voted_up is False and rows["3"].voted_up is True
    assert r1.review_date == date(2023, 1, 1)
    assert rows["2"].review_date == date(2023, 1, 1)  # 23:59:59 UTC stays on its day
    assert r1.created_at == r1.updated_at == r1.available_at
    assert (r1.snapshot_id, r1.cohort_policy) == (config.SNAPSHOT_ID, config.COHORT_POLICY)
    assert rows["2"].language == "german"  # every language is kept for coverage


def test_normalized_text(staged):
    out = {r.review_id: r for r in clean.to_clean(staged.where("stage = 'kept'")).collect()}
    assert out["1"].text_normalized == "crashes every match"
    assert out["1"].text_raw.startswith("[b]Crashes[/b]")  # raw text is untouched
