.DEFAULT_GOAL := help
JAVA_MAJOR := $(shell java -version 2>&1 | awk -F'"' '/version/ {print $$2}' | cut -d. -f1)

FILES ?= 2

.PHONY: help check-java setup lock test download-sample download profile manifest clean-cache

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

manifest:  ## Write data/published/manifest.json for the published outputs
	uv run python -m pipeline.manifest

clean-cache:  ## Remove Python/pytest caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
