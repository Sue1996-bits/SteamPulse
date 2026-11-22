"""Build the hand-checkable test fixture: two fake games over 35 days.

    uv run python tests/fixtures/build_fixture.py

Writes the CSVs next to this file. The expected outputs come from the per-day
counts written out below and the hand-computed window values in WINDOWS, not
from pipeline code, so they can be used to test it. See README.md for the math.
"""
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent
DAY1 = date(2023, 1, 1)          # day 1; day 35 = 2023-02-04 is the as-of date
DAYS = range(1, 36)
LABEL_VERSION = "fixture_v1"
SNAPSHOT = "fixture"


def day(d: int) -> date:
    return DAY1 + timedelta(days=d - 1)


# ---- Per-day spec ---------------------------------------------------------------
# Game 999: steady game with a spike on day 35.
#   English negatives: 10 a day, 40 on day 35
#   crash: 1 a day, 20 on day 35
#   performance: 5 on day 35 only, all on reviews that are also crash (multi-label)
#   match_server: 2 a day, on reviews that are not crash
#   monetization: none
# Game 888: steady game with a source gap.
#   English negatives: 10 a day, crash: 2 a day
#   day 20: no reviews at all (gap -> null)
#   day 25: no English negatives, but positive reviews exist (covered -> real 0)
def spec(app: str, d: int) -> dict:
    if app == "999":
        n = 40 if d == 35 else 10
        return {"neg": n, "crash": 20 if d == 35 else 1, "perf_on_crash": 5 if d == 35 else 0,
                "match_server": 2, "pos": 3, "other_lang_neg": 1, "gap": False}
    if d == 20:
        return {"neg": 0, "crash": 0, "perf_on_crash": 0, "match_server": 0, "pos": 0,
                "other_lang_neg": 0, "gap": True}
    if d == 25:
        return {"neg": 0, "crash": 0, "perf_on_crash": 0, "match_server": 0, "pos": 3,
                "other_lang_neg": 0, "gap": False}
    return {"neg": 10, "crash": 2, "perf_on_crash": 0, "match_server": 0, "pos": 3,
            "other_lang_neg": 0, "gap": False}


def build_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews, labels = [], []

    def add_review(app, d, i, *, voted_up, language="english", last_second=False):
        rid = f"{app}{d:02d}{i:03d}"
        t = time(23, 59, 59) if last_second else time(12, 0, i % 60)
        ts = datetime.combine(day(d), t, tzinfo=timezone.utc).isoformat()
        text = f"fixture review {rid}"
        reviews.append({
            "review_id": rid, "app_id": app, "text_raw": text, "text_normalized": text,
            "created_at": ts, "updated_at": ts, "available_at": ts, "review_date": day(d).isoformat(),
            "language": language, "voted_up": voted_up, "snapshot_id": SNAPSHOT,
            "cohort_policy": "strict_unedited",
        })
        return rid

    def add_label(rid, category):
        labels.append({"review_id": rid, "category": category, "is_match": True,
                       "rule_id": f"fixture_{category}", "evidence": category,
                       "label_version": LABEL_VERSION})

    for app in ("999", "888"):
        for d in DAYS:
            s = spec(app, d)
            i = 0
            for k in range(s["neg"]):
                i += 1
                # Day 34's last negative is created at 23:59:59 UTC: still day 34.
                rid = add_review(app, d, i, voted_up=False, last_second=(app == "999" and d == 34 and k == 0))
                if k < s["crash"]:
                    add_label(rid, "crash")
                    if k < s["perf_on_crash"]:
                        add_label(rid, "performance")
                elif k < s["crash"] + s["match_server"]:
                    add_label(rid, "match_server")
                # everything else stays unclassified
            for _ in range(s["pos"]):
                i += 1
                add_review(app, d, i, voted_up=True)
            for _ in range(s["other_lang_neg"]):
                i += 1
                rid = add_review(app, d, i, voted_up=False, language="german")
                add_label(rid, "crash")  # labeled, but not English: must not count anywhere

    # Duplicate label row (same key) on game 999, day 35: must still count once.
    labels.append(dict(next(l for l in labels if l["review_id"] == "99935001" and l["category"] == "crash")))
    return pd.DataFrame(reviews), pd.DataFrame(labels)


def expected_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    neg_rows, cat_rows = [], []
    for app in ("999", "888"):
        for d in DAYS:
            s = spec(app, d)
            n = None if s["gap"] else s["neg"]
            neg_rows.append({"app_id": app, "date": day(d).isoformat(), "negative_total": n,
                             "coverage_status": "gap" if s["gap"] else "covered"})
            counts = {"crash": s["crash"], "performance": s["perf_on_crash"],
                      "match_server": s["match_server"], "monetization": 0}
            for cat, c in counts.items():
                c = None if s["gap"] else c
                share = None if (n is None or n == 0) else c / n
                cat_rows.append({"app_id": app, "date": day(d).isoformat(), "category": cat,
                                 "category_count": c, "negative_total": n, "share": share,
                                 "coverage_status": "gap" if s["gap"] else "covered"})
    neg, cat = pd.DataFrame(neg_rows), pd.DataFrame(cat_rows)
    neg["negative_total"] = neg["negative_total"].astype("Int64")
    cat[["category_count", "negative_total"]] = cat[["category_count", "negative_total"]].astype("Int64")
    return neg, cat


# ---- Hand-computed windows at as_of_date = day 35 (2023-02-04) ------------------
# Recent window: days 29-35 (t-6..t). Baseline: days 1-28 (t-34..t-7).
# rolling28 baseline prediction: days 8-35 (t-27..t).
AS_OF = day(35).isoformat()
WINDOWS = [
    # game 999: N7 = 6*10 + 40 = 100, N28 = 28*10 = 280, rolling N = 27*10 + 40 = 310
    {"app_id": "999", "category": "crash", "C7": 26, "N7": 100, "recent_share": 0.26,
     "C28": 28, "N28": 280, "baseline_share": 0.10, "observed_lift_pp": 16.0,
     "observed_status": "emerging", "forecast_status": "ok", "eligibility_reason": None,
     "persistence_pred": 0.26, "rolling28_pred": 47 / 310},
    {"app_id": "999", "category": "performance", "C7": 5, "N7": 100, "recent_share": 0.05,
     "C28": 0, "N28": 280, "baseline_share": 0.0, "observed_lift_pp": 5.0,
     "observed_status": "not_flagged", "forecast_status": "ok", "eligibility_reason": None,
     "persistence_pred": 0.05, "rolling28_pred": 5 / 310},
    {"app_id": "999", "category": "match_server", "C7": 14, "N7": 100, "recent_share": 0.14,
     "C28": 56, "N28": 280, "baseline_share": 0.20, "observed_lift_pp": -6.0,
     "observed_status": "not_flagged", "forecast_status": "ok", "eligibility_reason": None,
     "persistence_pred": 0.14, "rolling28_pred": 56 / 310},
    {"app_id": "999", "category": "monetization", "C7": 0, "N7": 100, "recent_share": 0.0,
     "C28": 0, "N28": 280, "baseline_share": 0.0, "observed_lift_pp": 0.0,
     "observed_status": "not_flagged", "forecast_status": "ok", "eligibility_reason": None,
     "persistence_pred": 0.0, "rolling28_pred": 0.0},
]
# game 888: the recent window (days 29-35) is complete, but the baseline window
# contains the day-20 gap, so baseline values are null and both statuses are
# insufficient_data.
for cat, c7 in (("crash", 14), ("performance", 0), ("match_server", 0), ("monetization", 0)):
    WINDOWS.append({"app_id": "888", "category": cat, "C7": c7, "N7": 70, "recent_share": c7 / 70,
                    "C28": None, "N28": None, "baseline_share": None, "observed_lift_pp": None,
                    "observed_status": "insufficient_data", "forecast_status": "insufficient_data",
                    "eligibility_reason": "gap_in_window", "persistence_pred": None,
                    "rolling28_pred": None})


def main() -> None:
    reviews, labels = build_inputs()
    daily_neg, daily_cat = expected_daily()
    windows = pd.DataFrame(WINDOWS)
    windows.insert(1, "as_of_date", AS_OF)
    windows[["C7", "N7", "C28", "N28"]] = windows[["C7", "N7", "C28", "N28"]].astype("Int64")

    reviews.to_csv(OUT / "reviews_clean.csv", index=False)
    labels.to_csv(OUT / "review_labels.csv", index=False)
    daily_neg.to_csv(OUT / "expected_daily_negative.csv", index=False)
    daily_cat.to_csv(OUT / "expected_daily_category.csv", index=False)
    windows.to_csv(OUT / "expected_windows.csv", index=False)
    print(f"{len(reviews)} reviews, {len(labels)} label rows, {len(daily_neg)} game-days, "
          f"{len(windows)} window rows -> {OUT}")


if __name__ == "__main__":
    main()
