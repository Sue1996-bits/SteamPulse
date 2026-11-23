import os
from datetime import date, timedelta

import httpx
import pandas as pd
import streamlit as st

URL = os.environ.get("STEAMPULSE_API_URL", "http://127.0.0.1:8000")


def fetch(path, payload=None, **params):
    r = (httpx.post(URL + path, json=payload, trust_env=False) if payload is not None
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
    end = date.fromisoformat(game["end_date"])
    demo_day = date(2023, 8, 27)
    default_day = demo_day if game["app_id"] == "1172470" and demo_day.isoformat() in game["eligible_dates"] else end
    day = st.date_input("As of", default_day, min_value=date.fromisoformat(game["start_date"]), max_value=end)
    method = st.selectbox("Forecast method", ["trained_model", "persistence", "rolling28", "best_validation"])
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
        st.caption(f"Recent negative reviews: {rows[0]['sample_counts']['N7']} · Shares are fractions from 0 to 1.")
        st.caption("Observed alert status uses the latest 7 days versus the prior 28-day baseline; the forecast estimates the following 7 days.")
        st.dataframe(frame[["category", "observed_lift_pp", "observed_status", "recent_share",
                            "predicted_share", "predicted_change_pp", "forecast_status", "eligibility_reason"]], hide_index=True)
    cat = st.selectbox("Complaint history and evidence", ["crash", "performance", "match_server", "monetization"])
    history = fetch("/history", app_id=game["app_id"], category=cat,
                    start_date=(day-timedelta(days=34)).isoformat(), end_date=day.isoformat())
    if history:
        st.line_chart(pd.DataFrame(history).set_index("date")["share"], y_label="Observed share")
    for review in fetch("/reviews", **params, category=cat):
        st.text(f"{review['available_at']}  {review['excerpt']}")


try:
    dashboard()
except httpx.HTTPError as exc:
    st.error(f"API request failed: {exc}")
