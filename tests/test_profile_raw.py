from datetime import date

import pytest

from pipeline import profile_raw
from pipeline.config import RAW_COLUMNS, get_spark

DAY = 86_400
T0 = 1_700_006_400  # 2023-11-15 00:00:00 UTC


@pytest.fixture(scope="module")
def raw():
    spark = get_spark("test")
    rows = [
        # id, app, game, language, review, created, updated, voted_up
        (1.0, 10.0, "A", "english", "crashes", T0, T0, 0.0),                # strict negative
        (2.0, 10.0, "A", "english", "bad", T0 + 100, T0 + 999, 0.0),        # edited negative
        (2.0, 10.0, "A", "english", "bad", T0 + 100, T0 + 999, 0.0),        # duplicate id
        (3.0, 10.0, "A", "russian", "плохо", T0 + DAY - 1, T0 + DAY - 1, 0.0),  # last second of day 1
        (4.0, 10.0, "A", "english", "  ", T0 + 3 * DAY, T0 + 3 * DAY, 1.0),  # empty text, day 4
        (5.0, 20.0, "B", "english", "great", T0, T0, 1.0),
    ]
    return spark.createDataFrame(rows, RAW_COLUMNS)


def test_daily_counts_use_utc_days(raw):
    daily = profile_raw.daily_counts(profile_raw.with_flags(raw))
    got = {(r.appid, r.date): r for r in daily.collect()}
    day1 = got[(10.0, date(2023, 11, 15))]
    assert (day1.all_reviews, day1.en_neg, day1.en_neg_strict) == (4, 3, 1)
    assert (10.0, date(2023, 11, 18)) in got
    assert (10.0, date(2023, 11, 16)) not in got  # no reviews that day


def test_game_profile(raw):
    flags = profile_raw.with_flags(raw)
    prof = {r.appid: r for r in profile_raw.game_profile(flags, profile_raw.daily_counts(flags)).collect()}
    a = prof[10.0]
    assert (a.rows, a.unique_ids, a.duplicate_rows) == (5, 4, 1)
    assert (a.en_neg, a.en_neg_strict) == (3, 1)
    assert a.edit_rate == pytest.approx(2 / 5)
    assert a.en_neg_edit_share == pytest.approx(2 / 3)
    assert (a.span_days, a.active_days, a.days_without_reviews) == (4, 2, 2)
    assert a.empty_text == 1 and a.bad_id == 0 and a.missing_key == 0
    assert prof[20.0].en_neg_edit_share is None  # no negatives: undefined, not zero


def test_scale_log_row_matches_header_and_appends(tmp_path):
    stats = {
        "generated_at": "2026-11-16T07:35:24+00:00", "snapshot_id": "d0ec12b33dff",
        "files": 2, "columns": RAW_COLUMNS, "compressed_selected": 51_425_243,
        "uncompressed_selected": 87_384_450, "uncompressed_all": 104_189_325,
        "rows": 384_128, "unique_ids": 384_128, "games": 476, "output_files": 9,
        "output_bytes": 900_000, "driver_memory": "8g", "runtime_s": 9.7,
    }
    row = profile_raw.scale_log_row(stats)
    header = next(line for line in profile_raw.SCALE_LOG.read_text().splitlines()
                  if line.startswith("| Date"))
    assert row.count("|") == header.count("|")
    assert "benchmark (2/663 files)" in row and "87.4 MB" in row

    log = tmp_path / "scale_log.md"
    log.write_text("| a |\n|---|")
    profile_raw.append_scale_log(row, log)
    assert log.read_text().splitlines()[-1] == row
