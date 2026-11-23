"""Historical replay against the team's checked-in integration fixture."""
import csv
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures"
Category = Literal["crash", "performance", "match_server", "monetization"]
Method = Literal["trained_model", "persistence", "rolling28", "best_validation"]
CATEGORIES = ("crash", "performance", "match_server", "monetization")


def load(name):
    with (ROOT / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


daily = {}
for row in load("expected_daily_category"):
    key = row["app_id"], row["category"]
    daily.setdefault(key, {})[date.fromisoformat(row["date"])] = {
        "date": row["date"], "coverage_status": row["coverage_status"],
        "category_count": int(row["category_count"]) if row["category_count"] else None,
        "negative_total": int(row["negative_total"]) if row["negative_total"] else None,
        "share": float(row["share"]) if row["share"] else None,
    }

labels = {(r["review_id"], r["category"]) for r in load("review_labels")
          if r["is_match"].lower() == "true" and r["label_version"] == "fixture_v1"}
evidence = sorted((r for r in load("reviews_clean") if r["language"] == "english"
                   and r["voted_up"].lower() == "false" and r["created_at"] == r["updated_at"]),
                  key=lambda r: r["available_at"], reverse=True)

publication = None
titles = {}
if os.environ.get("STEAMPULSE_DATA_DIR"):
    from application.published import Publication
    import pandas as pd

    publication = Publication(os.environ["STEAMPULSE_DATA_DIR"])
    titles = {"1172470": "Apex Legends", "1517290": "Battlefield 2042", "1811260": "FIFA 23",
              "1085660": "Destiny 2", "359550": "Rainbow Six Siege"}
    table = publication.table("daily_category")
    table = table[(table.label_version == publication.meta["label_version"]) & (table.cohort_version == "v1")]
    daily = {}
    for item in table.to_dict("records"):
        day = pd.Timestamp(item["date"]).date()
        group = daily.setdefault((item["app_id"], item["category"]), {})
        if day in group:
            raise ValueError("Duplicate daily category")
        group[day] = {k: None if pd.isna(item[k]) else item[k] for k in
                      ("category_count", "negative_total", "coverage_status", "share")}
        group[day]["date"] = day.isoformat()
    tagged = publication.table("review_labels")
    labels = set(zip(*[tagged.loc[tagged.is_match & (tagged.label_version == publication.meta["label_version"]), col].astype(str)
                       for col in ("review_id", "category")]))
    clean = publication.table("reviews_clean")
    clean = clean[(clean.language == "english") & ~clean.voted_up & (clean.created_at == clean.updated_at)
                  & (clean.snapshot_id == publication.meta["snapshot_id"]) & (clean.cohort_policy == "strict_unedited")].copy()
    clean["available_at"] = pd.to_datetime(clean.available_at, utc=True).map(lambda d: d.isoformat())
    clean["review_id"] = clean.review_id.astype(str)
    evidence = clean.sort_values("available_at", ascending=False).to_dict("records")

app = FastAPI(title="SteamPulse historical replay")


def series(app_id, category):
    if (app_id, category) not in daily:
        raise HTTPException(404, "Unsupported game")
    return daily[app_id, category]


def window(rows, start, length):
    days = [rows.get(start + timedelta(days=i)) for i in range(length)]
    if any(day is None or day["coverage_status"] != "covered" for day in days):
        return None, None, None
    count = sum(day["category_count"] for day in days)
    total = sum(day["negative_total"] for day in days)
    return count, total, count / total if total else None


@app.get("/health")
def health():
    if publication:
        meta = publication.meta
        return dict(data_ready=True, source_snapshot=meta["snapshot_id"], feature_version=meta["feature_version"],
                    label_version=meta["label_version"], model_ready=bool(publication.model),
                    model_version=publication.model["model_version"] if publication.model else None)
    return dict(data_ready=True, source_snapshot="fixture", feature_version="fixture_v1",
                label_version="fixture_v1", model_ready=False, model_version=None)


@app.get("/games")
def games():
    result = []
    for game in sorted({key[0] for key in daily}):
        rows = series(game, "crash")
        eligible = [d for d in sorted(rows) if (window(rows, d-timedelta(days=6), 7)[1] or 0) >= 50
                    and (window(rows, d-timedelta(days=34), 28)[1] or 0) >= 100]
        if publication:
            frame = publication.features.xs(game, level="app_id")
            eligible = [d for d, group in frame.groupby(level="as_of_date") if
                        set(group.index.get_level_values("category")) == set(CATEGORIES) and
                        (group.eligibility == "eligible").all()]
        result.append(dict(app_id=game, title=titles.get(game, game if publication else f"Fixture game {game}"), start_date=min(rows),
                           end_date=max(rows), eligible_dates=eligible))
    return result


@app.get("/history")
def history(app_id: str, category: Category, start_date: date, end_date: date):
    if start_date > end_date:
        raise HTTPException(422, "Start must precede or equal end")
    return [v for d, v in sorted(series(app_id, category).items()) if start_date <= d <= end_date]


@app.get("/reviews")
def reviews(app_id: str, category: Category, as_of_date: date, limit: Annotated[int, Query(ge=1, le=50)] = 5):
    series(app_id, category)
    result = []
    for r in evidence:
        available = datetime.fromisoformat(r["available_at"]).astimezone(timezone.utc).date()
        if r["app_id"] == app_id and available <= as_of_date and (r["review_id"], category) in labels:
            result.append(dict(review_id=r["review_id"], available_at=r["available_at"], excerpt=r["text_raw"][:300]))
            if len(result) == limit:
                break
    return result


class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")
    app_id: str
    as_of_date: date
    horizon_days: Literal[7] = 7
    method: Method = "trained_model"


@app.post("/forecast")
def forecast(request: Forecast):
    t = request.as_of_date
    if t not in series(request.app_id, "crash"):
        raise HTTPException(422, "Replay date outside fixture coverage")
    if request.method == "best_validation" or (request.method == "trained_model" and
            (not publication or not publication.model or publication.cutoff > t)):
        raise HTTPException(503, f"{request.method} artifact unavailable for this replay")
    results = []
    for cat in CATEGORIES:
        rows = series(request.app_id, cat)
        c7, n7, recent = window(rows, t-timedelta(days=6), 7)
        c28, n28, prior = window(rows, t-timedelta(days=34), 28)
        reason = None
        if t-timedelta(days=34) < min(rows):
            reason = "insufficient_history"
        elif n7 is None or n28 is None:
            reason = "gap_in_window"
        elif n7 < 50 or n28 < 100:
            reason = "low_volume"
        lift = 100*(recent-prior) if recent is not None and prior is not None else None
        status = "insufficient_data"
        prediction = None
        if reason is None:
            status = "emerging" if c7 >= 10 and lift >= 5 else "not_flagged"
            prediction = recent if request.method == "persistence" else window(rows, t-timedelta(days=27), 28)[2]
        if publication:
            try:
                feature = publication.features.loc[(request.app_id, t, cat)]
            except KeyError:
                raise HTTPException(503, "Incomplete canonical feature publication")
            recent, prior, lift = (None if pd.isna(feature[k]) else float(feature[k])
                                   for k in ("share_7d", "baseline_share", "lift_pp"))
            reason = None if feature.eligibility == "eligible" else str(feature.eligibility)
            status, prediction = "insufficient_data", None
            if reason is None:
                status = "emerging" if feature.c_7d >= 10 and lift >= 5 else "not_flagged"
                if request.method == "trained_model":
                    try:
                        prediction = publication.prediction(feature, cat)
                    except ValueError:
                        raise HTTPException(503, "Model inputs incompatible with saved coefficients")
                else:
                    prediction = recent if request.method == "persistence" else float(feature.share_28d)
        results.append(dict(app_id=request.app_id, category=cat, as_of_date=t,
                            forecast_start=t+timedelta(days=1), forecast_end=t+timedelta(days=7),
                            recent_share=recent, baseline_share=prior, predicted_share=prediction,
                            observed_lift_pp=lift, observed_status=status, prediction_method=request.method,
                            predicted_change_pp=100*(prediction-recent) if prediction is not None else None,
                            forecast_status="insufficient_data" if reason else "ok", eligibility_reason=reason,
                            method_version=request.method+"_v1", training_label_end=None, model_version=None,
                            source_snapshot="fixture", feature_version="fixture_v1", label_version="fixture_v1",
                            sample_counts=dict(C7=c7, N7=n7, C28=c28, N28=n28)))
        if publication:
            version = health()
            fields = ("source_snapshot", "feature_version", "label_version")
            results[-1].update({k: version[k] for k in fields})
            if request.method == "trained_model":
                results[-1].update(model_version=version["model_version"], method_version=version["model_version"],
                                   training_label_end=publication.cutoff)
    return results
