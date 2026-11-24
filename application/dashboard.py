import os
from datetime import date, timedelta

import httpx
import pandas as pd
import streamlit as st

URL = os.environ.get("STEAMPULSE_API_URL", "http://127.0.0.1:8000")


def fetch(path, payload=None, **params):
    r = (httpx.post(URL + path, json=payload, timeout=60, trust_env=False) if payload is not None
         else httpx.get(URL + path, params=params, trust_env=False))
    r.raise_for_status()
    return r.json()


def dashboard():
    st.title("SteamPulse")
    meta = fetch("/health")
    st.caption(f"Historical replay · snapshot {meta['source_snapshot']} · UTC")
    games = {g["title"]: g for g in fetch("/games")}
    titles = sorted(games, reverse=True)
    default_game = "Apex Legends" if "Apex Legends" in games else titles[0]
    game = games[st.selectbox("Game", titles, index=titles.index(default_game))]
    model_start = date.fromisoformat(meta["replay_start"]) if meta["replay_start"] else None
    start = date.fromisoformat(game["start_date"])
    end = date.fromisoformat(game["end_date"])
    methods = ["persistence", "rolling28"]
    if model_start is not None and model_start <= end:
        methods.insert(0, "trained_model")
        if meta["best_validation_ready"]:
            methods.append("best_validation")
    method = st.selectbox("Forecast method", methods)
    if method in {"trained_model", "best_validation"}:
        start = max(start, model_start)
    demo_day = date(2023, 8, 27)
    default_day = (demo_day if game["app_id"] == "1172470" and start <= demo_day <= end
                   and demo_day.isoformat() in game["eligible_dates"] else end)
    day = st.date_input("As of", default_day, min_value=start, max_value=end,
                        key=f"replay_date_{game['app_id']}_{method}")
    params = dict(app_id=game["app_id"], as_of_date=day.isoformat())
    try:
        rows = fetch("/forecast", dict(**params, method=method))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 503:
            raise
        st.warning(exc.response.json()["detail"] + ". Select persistence or rolling28 to replay a baseline.")
    else:
        frame = pd.DataFrame(rows).sort_values("observed_lift_pp", ascending=False)
        st.subheader(f"Forecast · {rows[0]['forecast_start']} to {rows[0]['forecast_end']}")
        st.caption(f"Negative reviews in last 7 days: {rows[0]['sample_counts']['N7']} · "
                   f"Prior 28-day baseline: {rows[0]['sample_counts']['N_base']}")
        for column in ("recent_share", "baseline_share", "predicted_share"):
            frame[column] = frame[column] * 100
        st.caption("Observed alert status uses the latest 7 days versus the prior 28-day baseline; the forecast estimates the following 7 days.")
        st.dataframe(
            frame[["category", "observed_status", "recent_share", "baseline_share",
                   "predicted_share", "resolved_method", "forecast_status", "eligibility_reason"]],
            column_config={
                "recent_share": st.column_config.NumberColumn("Recent share", format="%.2f%%"),
                "baseline_share": st.column_config.NumberColumn("Prior 28-day share", format="%.2f%%"),
                "predicted_share": st.column_config.NumberColumn("Next 7-day forecast", format="%.2f%%"),
            },
            hide_index=True,
        )
    categories = ["crash", "performance", "match_server", "monetization"]
    cat = st.selectbox("Complaint history and evidence", categories,
                       index=2 if game["app_id"] == "1172470" else 0)
    history = fetch("/history", app_id=game["app_id"], category=cat,
                    start_date=(day-timedelta(days=34)).isoformat(), end_date=day.isoformat())
    if history:
        st.line_chart(pd.DataFrame(history).set_index("date")["share"], y_label="Observed share")
    if st.checkbox("Show realized share over the next 7 days"):
        actual = fetch("/history", app_id=game["app_id"], category=cat,
                       start_date=(day + timedelta(days=1)).isoformat(),
                       end_date=(day + timedelta(days=7)).isoformat())
        if len(actual) == 7 and all(row["coverage_status"] == "covered" for row in actual):
            negative_total = sum(row["negative_total"] for row in actual)
            if negative_total:
                realized = sum(row["category_count"] for row in actual) / negative_total
                st.metric("Realized next 7 days", f"{realized:.2%}")
            else:
                st.info("No negative reviews in the outcome window.")
        else:
            st.info("The full outcome window is unavailable.")
    for review in fetch("/reviews", **params, category=cat):
        with st.expander(f"{review['available_at'][:10]} · {review['matched_phrase'] or 'Review excerpt'}"):
            st.text(review["excerpt"])


try:
    dashboard()
except httpx.HTTPError as exc:
    st.error(f"API request failed: {exc}")
