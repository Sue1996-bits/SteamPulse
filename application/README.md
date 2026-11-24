# Historical replay demo

From the repository root, start the API and dashboard in separate terminals:

```sh
uv run --with-requirements application/requirements.txt python -m uvicorn application.api:app
uv run --with-requirements application/requirements.txt streamlit run application/dashboard.py
```

Open http://localhost:8501. API docs: http://localhost:8000/docs.
Run contract checks with
`uv run --with-requirements application/requirements.txt python -m unittest application.replay_checks`.

## Demo and offline backup

### Team handoff replay (Apex Legends)

Unzip Wren's `steampulse_published_rules_v2_gbt_v2.zip` into `data/staging/` and point the API
at the `published/` directory inside it:

```sh
unzip steampulse_published_rules_v2_gbt_v2.zip -d data/staging/gbt_v2
export STEAMPULSE_DATA_DIR="$PWD/data/staging/gbt_v2/published"
uv run --with-requirements application/requirements.txt python -m uvicorn application.api:app
```

In a second terminal, run `uv run --with-requirements application/requirements.txt streamlit run application/dashboard.py`.
The dashboard opens on Apex Legends and 2023-08-27, the agreed replay date. The API's trained-model
forecast for 2023-08-28 through 2023-09-03 flags `match_server` as emerging: its observed recent
share is 17.54%, against a 12.22% baseline. The saved GBT model predicts 13.09%; persistence
predicts 17.54%. Select “Show realized share over the next 7 days” to reveal the 19.30% outcome separately
from the forecast. The model underestimates this emerging issue
at the alert's start. This is a useful limitation to explain: the alert detects the observed rise,
while the forecasting model adjusts more slowly. In validation, the model's macro MAE was 1.61
percentage points versus 1.87 for persistence, but overall validation performance does not imply
that the model wins on every replay day or category.

If the live app is unavailable, show this snapshot from the checked Apex replay. The
forecast window is August 28 to September 3, 2023.

| Complaint | Recent 7 days | Prior 28 days | Model forecast | Observed alert |
| --- | ---: | ---: | ---: | --- |
| Crash | 3.22% | 3.06% | 3.16% | Not flagged |
| Performance | 1.46% | 0.82% | 0.74% | Not flagged |
| Match / server | 17.54% | 12.22% | 13.09% | Emerging |
| Monetization | 2.05% | 3.88% | 2.39% | Not flagged |

In the development workspace, the real bundle took about 14 seconds to load, the
first trained forecast took 2.84 seconds, and a repeat took 1.54 seconds. Start the
services and warm up the forecast before presenting.

Verify the real handoff after starting the API:

```sh
uv run --with-requirements application/requirements.txt python -m application.check_apex_replay "$STEAMPULSE_DATA_DIR"
```

This checks all four API predictions against the handoff's saved offline predictions and checks
the model cutoff, feature and label versions, and the `match_server` alert. Keep the extracted
bundle under `data/staging/`; do not commit the review text or saved model files.

### Synthetic offline backup

Select fixture game 999, persistence, and February 4, 2023. Crash reached 26 of 100 recent negative reviews (26%), compared with
28 of 280 (10%) in the preceding 28 days. The observed increase is 16 percentage points.
Persistence forecasts 26% over February 5–11. Switching to rolling28 forecasts 47/310 (15.16%).
The two estimates are alternative baselines. This synthetic case demonstrates the application,
and supplies no evidence about actual games or model accuracy. Show the crash history and dated
excerpts. Switch to game 888 to demonstrate a gap and a null forecast.

If the UI is unavailable, run the same request from the FastAPI docs. If the API is unavailable,
show tests/fixtures/expected_windows.csv and the hand calculation above as a static backup.

## Integration handoffs

* A: the versioned publication reader now serves Wren's `rules_v2` Parquet tables and canonical
  features. The actual bundle was loaded and the five game/date ranges were checked.
* B: the app loads the saved `gbt_v2` Spark pipelines. The real-data replay check matched all four
  API predictions to Wren's offline predictions within the manifest tolerance (1e-9), and confirmed
  the 2023-08-15 training cutoff and the `match_server` alert. The model and validation modes use the
  published replay cutoff in the date picker. Baselines are available for earlier dates.
* C: the real API replay, evidence endpoint, baselines and cutoff failure state were checked. Before
  presenting, restart both services on the presentation machine, measure cold and warm latency,
  and capture a real replay for the report/slides backup.

## Serving a team publication

Set STEAMPULSE_DATA_DIR to a compact export for the selected games before starting the API.
The directory contains A's feature_manifest.json and Parquet tables named daily_category,
forecast_features, reviews_clean, and review_labels from the same run. Use the exported
strict_unedited/v1 cohort. The app filters labels and features to the manifest versions.
For large source tables, export the demo games before starting the service.

The model manifest uses `spark_pipeline`, model_version, training_label_end (ISO date), feature_version,
label_version, snapshot_id, ordered features, and per-category saved Spark PipelineModel paths.
The API loads all four at startup and rejects requests before the training cutoff. The handoff
check is `application.check_apex_replay`; it compares every API prediction with the offline bundle
and checks that the expected alert reproduces. When the bundle specifies a validation winner for all four categories, `best_validation` uses that
method per category from the replay cutoff onwards. The response includes `resolved_method`.
The API uses observed alerts from the same published baseline table when available and uses the
pipeline thresholds for fixture or feature-only replay.

Run both application suites with:
`uv run --with-requirements application/requirements.txt python -m unittest application.replay_checks application.publication_checks`.
