# SteamPulse

SteamPulse finds complaint types that are picking up in Steam reviews and forecasts whether they'll keep growing over the next week.
Steam reviews are full of player feedback, but a rising problem is easy to miss. SteamPulse sorts negative reviews into complaint categories such as crashes, performance, matchmaking/server, and monetization, and tracks each category's daily share of negative reviews for selected games.

A category is flagged as **emerging** when its share over the last 7 days beats its share over the previous 28 days by a set threshold. Minimum-volume checks keep small samples from setting off false alarms. A Spark ML model then uses recent volumes and trends to predict each category's share for the next 7 days. We compare it against simple baselines using chronological train/validation/test splits.

The trained model is served through a FastAPI service. In the Streamlit dashboard, you pick a game and a date and see observed and predicted growth, with review excerpts as evidence.

SteamPulse is meant to help teams decide which issues are worth looking into. It doesn't identify root causes, judge technical severity, or set fix priorities.

## Scope

- **Data:** the public Steam reviews snapshot on Hugging Face ([`adde023/steam-reviews`](https://huggingface.co/datasets/adde023/steam-reviews)), processed in batch with Spark. Modeling uses English negative reviews for one to five games.
- **Complaint categories:** `crash`, `performance`, `match_server`, `monetization` (multi-label). Negative reviews that match none of these are kept as unclassified.
- **Product:** batch data pipeline, rule-based emerging-issue detection, seven-day share forecast (Spark ML + MLflow), FastAPI service, and a Streamlit historical replay.

## Team

| Member | Role | Responsibilities |
|---|---|---|
| [Isabella Shi](https://github.com/isabellaywshi) | Data engineering | Data profiling and cleaning, daily tables, features, baselines, quality checks, pipeline orchestration |
| [Yifan Huang](https://github.com/Sue1996-bits)| Analysis / ML | Complaint taxonomy, labeling rules, alerts, model training and evaluation, MLflow tracking |
| [Yunseo Oh](https://github.com/yunseoyunnieoh) | Application | API design, model serving, Streamlit dashboard, integration, demo |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), `make`, and Java 17.

```bash
# macOS: install Java 17 once, then make it the default
brew install openjdk@17
echo 'export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

make setup   # checks Java, creates .venv from uv.lock
make test
```