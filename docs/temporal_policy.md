# Temporal policy

How dates, review filtering, missing days and game selection are handled, so a count means the same thing in the pipeline, the model and the app. Numbers come from the full profile run on snapshot `d0ec12b` (`make profile` and `make select-games`).

## Day boundaries

All days are UTC (`get_spark()` sets `spark.sql.session.timeZone=UTC`). A review belongs to the UTC day of its `timestamp_created`.

## Analysis window

| | Dates | Where it's set |
|---|---|---|
| Snapshot range | 2010-10-15 to 2023-11-03 | profile |
| Analysis window | 2022-11-01 to 2023-10-31 (12 months) | `WINDOW_START`, `WINDOW_END` |
| Training period (tentative) | 2022-11-01 to 2023-05-31 (~60%) | `TRAIN_END`, pending the train/validation/test split |

The window ends on 2023-10-31 because the last days of the snapshot are incomplete. Collection ended in early November 2023, and games stop on different days (Nov 1-3). The whole dataset averages about 55,000 reviews a day, but Nov 2 has 30,223 and Nov 3 has 5,132. Those are partial days rather than a real drop, so they're left out.

Each forecast date also needs 35 days of history (t-34 to t) and 7 days after it to score against, so the first and last usable dates sit a little inside the window.

## Strict cohort (`strict_unedited`)

Only reviews with `timestamp_created == timestamp_updated` are used, for both the category counts and the denominators. An edited review may have text or a recommendation written after its creation date, which would leak later information into a historical replay.

Share of English negatives removed as edited (training period):

| Game | Removed as edited |
|---|---|
| Apex Legends | 14% |
| Battlefield 2042 | 16% |
| FIFA 23 | 15% |
| Destiny 2 | 19% |
| Rainbow Six Siege | 15% |
| Call of Duty | 27% |
| All 17 core candidates | 10-27% |

This rule has a bias to note in the report: older reviews have had more time to be edited. For the 20 games with the most English negatives, 20-39% were edited over each game's full history, compared with 10-27% in the recent window. So the rule keeps a larger share of recent reviews than older ones, and later periods look slightly busier than they really were.

If the strict cohort turns out too sparse, the fallback is to use each review's latest version, dated by `timestamp_updated`. That measures what was visible when, not when reviews were written, and should be described that way.

## Missing days

A game-day is covered if the game has at least one review that day in any language. A day with zero reviews in every language, between a game's first and last review, is treated as a suspected source gap:

- `coverage_status = "gap"`, and its counts and shares are null, not 0.
- A covered day with no English negatives is a real 0.
- Forecast and alert windows that include a gap day return `insufficient_data`.

Gaps are checked per game. No date in the last year is missing for the whole dataset, but some games have their own gaps (War Thunder has 12 zero-review days in the training period). None of the core candidates have any.

## Data quality

The full profile found 0 duplicate review IDs, 0 non-integer IDs and 0 missing keys. 46,718 reviews have empty text; the clean step moves them to quarantine and logs the count rather than dropping them silently.

## Game selection

Only the training period is used here, so validation and test data don't influence which games are modeled. A game is a candidate if it:

1. was released by 2022-11-01 and has data through 2023-10-31,
2. has at least 1,000 strict English negatives in training, and
3. passes the forecast checks (N7 >= 50 and N28 >= 100) on most training days. Core means at least 80% of days and at most 7 gap days; near-miss means at least 40% and at most 40 gap days.

This gives 17 core games and 11 near-misses. The full table is in `data/profile/candidates.csv` after running `make select-games`.

To check that each complaint category has enough reviews, `pipeline/select_games.py` scans strict English negatives for keywords. This is a rough check, not the labeling rules. A category needs about 1.4 matches a day to reach C7 >= 10 in a typical week.

| Category | Core games with at least 1.4 keyword matches a day |
|---|---|
| crash | Call of Duty (55), FIFA 23 (4.4), Battlefield 2042 (1.8), Apex (1.4), RDR2 (1.4) |
| performance | Call of Duty (9.5), Battlefield 2042 (2.4), FIFA 23 (2.0) |
| match_server | Call of Duty (20.7), Apex (6.0), ARK (5.0), Battlefield 2042 (2.2), Rainbow Six Siege (2.0), CS2 (1.8), Lost Ark (1.7), Rust (1.6), GTA V (1.6) |
| monetization | Call of Duty (10.0), Destiny 2 (4.0), ARK (3.3), Lost Ark (1.6) |

Performance has the fewest matches, so at least one of Call of Duty, Battlefield 2042 or FIFA 23 is needed to evaluate it.

Counter-Strike 2 isn't a good fit: 94% of its negative reviews match no category, and the switch from CS:GO to CS2 happens during the test period.

### Proposed games (to discuss)

Apex Legends (1172470) is proposed as the first game for the end-to-end prototype. It passes the forecast checks on every training day, has no gap days and averages about 39 strict English negatives a day. It came out in 2020, so there's no launch spike in the window.

Two options for the other four:

| Option | Games | Tradeoff |
|---|---|---|
| 1 | Battlefield 2042, FIFA 23, Destiny 2, Rainbow Six Siege | Covers every category, but performance depends on two games |
| 2 | Call of Duty, Battlefield 2042, FIFA 23, Destiny 2 | More reviews in every category, but Call of Duty launched on 2022-10-28, so its launch period is in the training data |

The final games and the date they were chosen will be added here once decided.
