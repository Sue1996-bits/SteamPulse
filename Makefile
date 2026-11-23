.DEFAULT_GOAL := help
JAVA_MAJOR := $(shell java -version 2>&1 | awk -F'"' '/version/ {print $$2}' | cut -d. -f1)

FILES ?= 2

.PHONY: help check-java setup lock test download-sample download profile select-games clean-reviews label-sample labels label-eval daily features baselines alerts train mlflow-ui checks publish pipeline manifest clean-cache

help:  ## Show available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

check-java:  ## Fail unless Java 17 is active
	@if [ "$(JAVA_MAJOR)" != "17" ]; then \
		echo "Need Java 17 (found: $(or $(JAVA_MAJOR),none)). See README > Setup."; exit 1; \
	fi
	@echo "Java $(JAVA_MAJOR) OK"

setup: check-java  ## Create .venv from uv.lock
	uv sync --locked

lock:  ## Update uv.lock after editing pyproject.toml
	uv lock

test: check-java  ## Run tests
	uv run pytest -q

download-sample:  ## Download the first FILES part files (default 2) to data/raw/
	uv run python -m pipeline.download --files $(FILES)

download:  ## Download all 663 part files (~23.5 GB) to data/raw/
	uv run python -m pipeline.download

profile: check-java  ## Profile every part file in data/raw/ (NOLOG=1 skips the scale log)
	uv run python -m pipeline.profile_raw $(if $(NOLOG),--no-log)

select-games: check-java  ## Rank candidate games on the training period -> data/profile/candidates.csv
	uv run python -m pipeline.select_games

clean-reviews: check-java  ## Build reviews_clean for the chosen games (APPS="1172470" for a subset)
	uv run python -m pipeline.clean $(if $(APPS),--apps $(APPS))

label-sample: check-java  ## Draw the hand-labeling sample (300 train + 150 later) -> data/samples/
	uv run python -m pipeline.label_sample $(if $(APPS),--apps $(APPS)) $(if $(SEED),--seed $(SEED))

labels: check-java  ## Rule labels for English negatives -> data/staging/review_labels/ (LABEL_VERSION=v, default in config)
	uv run python -m pipeline.labels $(if $(LABEL_VERSION),--label-version $(LABEL_VERSION)) $(if $(APPS),--apps $(APPS))

label-eval: check-java  ## Score the rules on hand labels (GOLD=path, PERIOD=train|later)
	uv run python -m pipeline.label_eval score $(GOLD) $(if $(PERIOD),--period $(PERIOD)) $(if $(LABEL_VERSION),--label-version $(LABEL_VERSION))

daily: check-java  ## Build daily_negative (and daily_category with LABELS=path LABEL_VERSION=v)
	uv run python -m pipeline.aggregate $(if $(LABELS),--labels $(LABELS) --label-version $(LABEL_VERSION))

features: check-java  ## Build forecast_features + feature_manifest.json (LABEL_VERSION=v required)
	uv run python -m pipeline.features --label-version $(LABEL_VERSION)

baselines: check-java  ## Persistence and rolling-28 forecast_results + baseline_scores.csv
	uv run python -m pipeline.baselines

alerts: check-java  ## Alert counts, threshold sensitivity (validation) and what follows an alert (LABEL_VERSION=v; TEST=1 adds test)
	uv run python -m pipeline.alerts $(if $(LABEL_VERSION),--label-version $(LABEL_VERSION)) $(if $(TEST),--include-test)

train: check-java  ## GBT per category + MLflow -> data/staging/model/ (LABEL_VERSION=v; TEST=1 scores the test period, once)
	uv run python -m pipeline.train $(if $(LABEL_VERSION),--label-version $(LABEL_VERSION)) $(if $(TEST),--score-test)

mlflow-ui:  ## Browse training runs in mlruns/ at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri mlruns

checks: check-java  ## Run quality checks on data/staging/ -> check_report.json
	uv run python -m pipeline.checks

publish: check-java  ## Run checks, then copy staging to data/published/ with manifest.json
	uv run python -m pipeline.checks --publish

pipeline: check-java  ## Run every step in order (LABEL_VERSION=v [LABELS=path], PUBLISH=1, LOG=1 for the scale log)
	uv run python -m pipeline.run $(if $(LABELS),--labels $(LABELS)) $(if $(LABEL_VERSION),--label-version $(LABEL_VERSION)) $(if $(PUBLISH),--publish) $(if $(LOG),--log)

manifest:  ## Write data/published/manifest.json for the published outputs
	uv run python -m pipeline.manifest

clean-cache:  ## Remove Python/pytest caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
