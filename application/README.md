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
predicts 17.54%. The realized share was 19.30%, so the model underestimates this emerging issue
at the alert's start. This is a useful limitation to explain: the alert detects the observed rise,
while the forecasting model adjusts more slowly. In validation, the model's macro MAE was 1.61
percentage points versus 1.87 for persistence, but overall validation performance does not imply
that the model wins on every replay day or category.

Verify the real handoff after starting the API:

```sh
uv run --with-requirements application/requirements.txt python -m application.check_apex_replay "$STEAMPULSE_DATA_DIR"
```

This checks all four API predictions against the handoff's saved offline predictions and checks
the model cutoff, feature and label versions, and the `match_server` alert. Keep the extracted
bundle under `data/staging/`; do not commit the review text or saved model files.

### Synthetic offline backup

Select fixture game 999 and February 4, 2023. The default trained method reports the missing artifact.
Select persistence. Crash reached 26 of 100 recent negative reviews (26%), compared with
28 of 280 (10%) in the preceding 28 days. The observed increase is 16 percentage points.
Persistence forecasts 26% over February 5–11. Switching to rolling28 forecasts 47/310 (15.16%).
The two estimates are alternative baselines. This synthetic case demonstrates the application,
and supplies no evidence about actual games or model accuracy. Show the crash history and dated
excerpts. Switch to game 888 to demonstrate a gap and a null forecast.

If the UI is unavailable, run the same request from the FastAPI docs. If the API is unavailable,
show tests/fixtures/expected_windows.csv and the hand calculation above as a static backup.

## Integration handoffs

* A: review counts, gap/zero handling, covered dates, cohort and version selection. Replace the
  fixture loader with a reader for one checked, version-pinned publication of the feature,
  daily and evidence tables. Use canonical features in production.
* B: supply saved transforms/models, ordered feature manifest, versions, training outcome cutoff,
  validation method selection, and offline predictions for agreed rows. Load once, reject
  incompatible features and models trained beyond the replay day, and compare all four API
  predictions with offline outputs before describing this as model deployment.
* C: repeat contract checks with real data, measure cold and warm latency on the presentation machine,
  restart both services and capture a real replay for backup. Add evidence to the report and slides.

## Serving a team publication

Set STEAMPULSE_DATA_DIR to a compact export for the selected games before starting the API.
The directory contains A's feature_manifest.json and Parquet tables named daily_category,
forecast_features, reviews_clean, and review_labels from the same run. Use the exported
strict_unedited/v1 cohort. The app filters labels and features to the manifest versions.
For large source tables, export the demo games before starting the service.

The optional model_manifest.json is the proposed B-to-C interface. Supply format
spark_pipeline, model_version, training_label_end (ISO date), feature_version,
label_version, snapshot_id, features (ordered column names), and categories
(an object mapping each category ID to an object with path pointing to its saved
Spark PipelineModel). Include every fitted transform in each saved pipeline. The API
loads all four at startup and rejects requests before the training cutoff. For a
linear model exported on raw features, format linear takes intercept and weights per
category. The synthetic publication test shows this form. Actual B artifacts still
require parity testing. Keep best_validation unavailable until the validation winner
and selection cutoff are published.

Run both application suites with:
`uv run --with-requirements application/requirements.txt python -m unittest application.replay_checks application.publication_checks`.
