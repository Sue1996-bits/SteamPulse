# Test fixture

Two fake games over 35 days (2023-01-01 to 2023-02-04), small enough to check by hand. The pipeline, the model and the API can all be tested against the same numbers. To rebuild the files after changing the spec:

```bash
uv run python tests/fixtures/build_fixture.py
```

| File | What it is |
|---|---|
| `reviews_clean.csv` | Input reviews, in the `reviews_clean` schema |
| `review_labels.csv` | Input labels, in the `review_labels` schema |
| `expected_daily_negative.csv` | Expected `daily_negative` rows |
| `expected_daily_category.csv` | Expected `daily_category` rows |
| `expected_windows.csv` | Expected 7/28-day windows, statuses and baseline predictions on 2023-02-04 |

## Game 999: a crash spike

Every day has 10 English negative reviews, except day 35 (2023-02-04), which has 40. Crash labels: 1 a day, then 20 on day 35. Five of those 20 crash reviews are also labeled performance, and every day has 2 match_server reviews. The other negatives have no label and stay in the denominator as unclassified. Each day also has 3 positive reviews and 1 German negative, and neither counts.

Windows on 2023-02-04:

| | Days | Crash | Negatives | Share |
|---|---|---|---|---|
| Recent (t-6 to t) | Jan 29 to Feb 4 | 6×1 + 20 = 26 | 6×10 + 40 = 100 | 0.26 |
| Baseline (t-34 to t-7) | Jan 1 to Jan 28 | 28 | 280 | 0.10 |
| Rolling 28 (t-27 to t) | Jan 8 to Feb 4 | 27 + 20 = 47 | 270 + 40 = 310 | 0.152 |

- Lift is 100 × (0.26 − 0.10) = +16 pp, with N7 = 100, N28 = 280 and C7 = 26, so crash is `emerging`.
- Performance: C7 = 5, a 5 pp lift, but C7 < 10, so it's `not_flagged`.
- Match_server: share drops from 0.20 to 0.14 (−6 pp), so it's `not_flagged`.
- The persistence prediction is the recent share (0.26 for crash). The rolling-28 prediction is 47/310.

Averaging the seven daily crash shares instead gives (6 × 0.10 + 0.50) / 7 = 0.157, not 0.26. The test for this makes sure window shares are computed as a ratio of sums.

## Game 888: a gap and a real zero

10 English negatives a day with 2 crash labels each, except:

- Day 20 (2023-01-20) has no reviews at all. It's a gap, so `coverage_status = "gap"` and its counts and shares are null.
- Day 25 (2023-01-25) has positive reviews but no English negatives. It's covered, so `negative_total = 0` and its shares are null (0/0).

The recent window is complete (crash share 14/70 = 0.20), but the baseline window includes the gap. Baseline values are null and both statuses are `insufficient_data`, with reason `gap_in_window`.

## Gap days in the daily tables

Every game-day gets a row, including gaps. A gap day has one `daily_negative` row and four `daily_category` rows, all with null counts and `coverage_status = "gap"`. The grid for each game starts on its first review date, so days before release aren't rows at all. Both fixture games start on day 1.

Sums skip nulls, so a gap doesn't show up in N7 or N28 on its own. Window code should count covered days and require 7 of 7 and 28 of 28. Game 888 checks this.

## Other cases covered

- One review is labeled both crash and performance. It counts once in each category but only once in `negative_total`.
- One label row is duplicated (same `review_id`, `category`, `label_version`). It still counts once.
- One review on 2023-02-03 is created at 23:59:59 UTC and must stay on that day.
