"""Train and evaluate the seven-day share forecast: one Spark GBT model per category.

    python -m pipeline.train --label-version rules_v2               # search, refit, export (gbt_v2)
    python -m pipeline.train --label-version rules_v2 --score-test  # also score the test period (once, at the end)
    python -m pipeline.train --target level --out-dir data/staging/compare_gbt_v1   # gbt_v1, for comparison

Two targets:

  residual (gbt_v2, deployed)  prediction = share_28d + shrinkage x GBT(target_share_next7 - share_28d).
                               The rolling 28-day share supplies the level and the trees learn only
                               a correction, so the model follows level shifts that never occurred in
                               training (trees can't extrapolate) and falls back to the baseline when
                               it learns nothing. shrinkage is chosen on validation from SHRINKAGE;
                               0 is left out so that a learned component is always deployed.
  level (gbt_v1)               prediction = GBT(target_share_next7), kept for comparison.

Splits are in pipeline/splits.py (by target week, with a purged week at each
boundary). Only eligible rows with a target are fitted or scored. Per category,
pooled across games:

  1. Fit each GRID setting (and, for residual, each shrinkage) on train and score
     MAE on validation.
  2. Compare the best setting with persistence (share_7d) and rolling-28
     (share_28d) on exactly the same validation rows.
  3. Refit the best setting on every row with t + 7 <= VAL_END. That model is the
     one deployed; the last target day it saw is its training_label_end.

Each saved model is a Spark PipelineModel: Imputer (median, fitted on the model's
own training rows) -> VectorAssembler -> GBTRegressor, plus for residual a
SQLTransformer that adds the correction to share_28d. It reads the raw
FEATURE_COLS, in feature_manifest order, and writes `prediction`, so serving is
the same for both targets. Predictions are
clipped to [0, 1] wherever they are scored or published, for every method.

Writes to data/staging/:
  model/<category>/              saved PipelineModels
  model_manifest.json            what the API loads and checks (application/README.md)
  model_predictions/             forecast_results rows from the deployed model, for every
                                 as_of_date >= training_label_end
  model_parity.csv               expected API outputs for a few agreed rows, with the tolerance
                                 in model_manifest.json
  model_scores.csv               MAE (percentage points) per split, category, game and method on
                                 identical rows; mae_pp_weighted weights rows by target_n_next7;
                                 category "macro" is the mean over categories
  model_row_counts.csv           scored rows and every exclusion reason, per split and category
  model_errors.csv               the largest validation (and test) errors, for error analysis
  model_feature_importance.csv
  model_card.md                  generated from the files above
Every run is logged to MLflow under mlruns/ (`make mlflow-ui`).
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, SQLTransformer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from pipeline import config
from pipeline.baselines import METHODS as BASELINES, observed_status
from pipeline.config import get_spark
from pipeline.features import MANIFEST_PATH as FEATURE_MANIFEST, OUT_DIR as FEATURES_DIR
from pipeline.manifest import _code_revision
from pipeline.splits import SPLITS, row_status, split_col, usable

STAGING = config.STAGING_DIR
MODEL_DIR = STAGING / "model"
MODEL_MANIFEST = STAGING / "model_manifest.json"
PREDICTIONS_DIR = STAGING / "model_predictions"
PARITY_PATH = STAGING / "model_parity.csv"
SCORES_PATH = STAGING / "model_scores.csv"
ROW_COUNTS_PATH = STAGING / "model_row_counts.csv"
ERRORS_PATH = STAGING / "model_errors.csv"
IMPORTANCE_PATH = STAGING / "model_feature_importance.csv"
CARD_PATH = STAGING / "model_card.md"
OUTPUT_FILES = [MODEL_MANIFEST, PARITY_PATH, SCORES_PATH, ROW_COUNTS_PATH, ERRORS_PATH, IMPORTANCE_PATH, CARD_PATH]
MLRUNS = config.ROOT / "mlruns"
EXPERIMENT = "steampulse-forecast"
SEED = 226
PARITY_TOLERANCE = 1e-9  # offline vs API prediction, same row and model
TOP_ERRORS = 10
# Small on purpose: ~850 training rows per category. Tune here, on validation only.
GRID = [{"maxDepth": d, "maxIter": n, "stepSize": 0.1} for d in (2, 3, 4) for n in (20, 50)]
TARGETS = {"residual": config.MODEL_VERSION, "level": "gbt_v1"}  # target -> model_version
TARGET_FORMULAS = {"residual": "share_28d + shrinkage * GBT(target_share_next7 - share_28d)",
                   "level": "GBT(target_share_next7)"}
SHRINKAGE = [0.25, 0.5, 1.0]  # weight on the learned correction; 0 would deploy the baseline itself
RESIDUAL_COL = "residual_target"
RESIDUAL_PRED = "residual_prediction"
BASE_FILLED = "share_28d__filled"


# ---- model --------------------------------------------------------------------

def build_pipeline(params: dict, seed: int = SEED, target: str = "residual") -> Pipeline:
    """The fitted part. For residual, finalize() adds the step that turns the correction into a share."""
    filled = [f"{c}__filled" for c in config.FEATURE_COLS]
    label, pred = (RESIDUAL_COL, RESIDUAL_PRED) if target == "residual" else (config.LABEL_COL, "prediction")
    return Pipeline(stages=[
        Imputer(inputCols=config.FEATURE_COLS, outputCols=filled, strategy="median"),
        VectorAssembler(inputCols=filled, outputCol="features"),
        GBTRegressor(featuresCol="features", labelCol=label, predictionCol=pred, seed=seed, **params),
    ])


def add_back(shrinkage: float) -> SQLTransformer:
    return SQLTransformer(
        statement=f"SELECT *, {BASE_FILLED} + {shrinkage!r} * {RESIDUAL_PRED} AS prediction FROM __THIS__")


def finalize(fitted: PipelineModel, target: str, shrinkage: float | None) -> PipelineModel:
    """The servable model: reads raw features, writes `prediction`."""
    return PipelineModel(stages=[*fitted.stages, add_back(shrinkage)]) if target == "residual" else fitted


def with_residual(df: DataFrame) -> DataFrame:
    return df.withColumn(RESIDUAL_COL, F.col(config.LABEL_COL) - F.col("share_28d"))


def clip(c: Column) -> Column:
    return F.least(F.greatest(c, F.lit(0.0)), F.lit(1.0))


def abs_error(pred: Column) -> Column:
    return F.abs(clip(pred) - F.col(config.LABEL_COL))


def mae_pp(df: DataFrame, pred: Column) -> tuple[float | None, int]:
    """Mean absolute error in percentage points, and the number of rows scored."""
    r = df.select(F.avg(abs_error(pred)).alias("mae"), F.count("*").alias("n")).first()
    return (None if r.mae is None else 100 * r.mae), r.n


def methods(model_name: str) -> list[tuple[str, Column]]:
    return [(model_name, F.col("prediction")), *[(m, F.col(c)) for m, c in BASELINES.items()]]


def score(scored: DataFrame, model_name: str, split: str, category: str) -> list[dict]:
    """MAE of the model and both baselines on the same rows, for all games and per game."""
    aggs = [F.count("*").alias("rows")]
    for i, (_, pred) in enumerate(methods(model_name)):
        aggs += [F.avg(abs_error(pred)).alias(f"mae{i}"),
                 (F.sum(abs_error(pred) * F.col("target_n_next7")) / F.sum("target_n_next7")).alias(f"wmae{i}")]
    table = (scored.groupBy("app_id").agg(*aggs)
             .unionByName(scored.agg(*aggs).withColumn("app_id", F.lit("all")))).toPandas()
    out = []
    for r in table.itertuples():
        for i, (method, _) in enumerate(methods(model_name)):
            mae, wmae = getattr(r, f"mae{i}"), getattr(r, f"wmae{i}")
            out.append(dict(split=split, category=category, app_id=r.app_id, method=method, rows=r.rows,
                            mae_pp=None if pd.isna(mae) else 100 * mae,
                            mae_pp_weighted=None if pd.isna(wmae) else 100 * wmae))
    return out


def largest_errors(scored: DataFrame, model_name: str, split: str, n: int = TOP_ERRORS) -> pd.DataFrame:
    pred = clip(F.col("prediction"))
    return (scored.select(
                F.lit(split).alias("split"), F.lit(model_name).alias("method"), "category", "app_id",
                "as_of_date", F.col(config.LABEL_COL).alias("target_share"), pred.alias("predicted_share"),
                clip(F.col("share_7d")).alias("persistence_share"), "baseline_share", "lift_pp", "n_7d",
                "target_n_next7", (100 * F.abs(pred - F.col(config.LABEL_COL))).alias("abs_error_pp"))
            .orderBy(F.desc("abs_error_pp")).limit(n).toPandas())


def row_counts(feats: DataFrame) -> pd.DataFrame:
    """Rows per split, category and status (scored, no_target or an eligibility reason)."""
    counts = (feats.withColumn("status", row_status()).groupBy("split", "category", "status").count()
              .toPandas().pivot_table(index=["split", "category"], columns="status", values="count",
                                      fill_value=0).reset_index())
    counts.columns.name = None
    statuses = [c for c in counts.columns if c not in ("split", "category")]
    counts[statuses] = counts[statuses].astype(int)
    order = {name: i for i, name in enumerate(SPLITS)}
    return counts.sort_values(["split", "category"], key=lambda s: s.map(order) if s.name == "split" else s,
                              ignore_index=True)


def predictions(feats: DataFrame, models: dict[str, PipelineModel], model_version: str,
                training_label_end: str) -> DataFrame:
    """forecast_results rows from the deployed model, as_of_date >= training_label_end only."""
    eligible = F.col("eligibility") == "eligible"
    cutoff = F.col("as_of_date") >= F.lit(training_label_end).cast("date")
    parts = []
    for category, model in models.items():
        rows = feats.where((F.col("category") == category) & cutoff)
        pred = F.when(eligible, clip(F.col("prediction")))
        parts.append(model.transform(rows).select(
            "app_id", "as_of_date", "category",
            F.lit(model_version).alias("method_version"),
            F.col("share_7d").alias("recent_share"),
            "baseline_share",
            F.col("lift_pp").alias("observed_lift_pp"),
            observed_status().alias("observed_status"),
            pred.alias("predicted_share"),
            (100 * (pred - F.col("share_7d"))).alias("predicted_change_pp"),
            F.when(eligible, "ok").otherwise("insufficient_data").alias("forecast_status"),
            F.when(~eligible, F.col("eligibility")).alias("eligibility_reason"),
            F.lit(model_version).alias("model_version"),
            F.lit(training_label_end).cast("date").alias("training_label_end"),
            "feature_version",
        ))
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    return out.select(*config.FORECAST_RESULTS_COLS)


def parity_rows(preds: DataFrame) -> pd.DataFrame:
    """First, middle and last forecastable date per game, all four categories."""
    ok = (preds.where(F.col("forecast_status") == "ok")
          .select("app_id", "as_of_date", "category", "predicted_share").toPandas())
    keep = []
    for _, g in ok.groupby("app_id"):
        days = sorted(g.as_of_date.unique())
        keep += [(g.app_id.iloc[0], d) for d in {days[0], days[len(days) // 2], days[-1]}]
    picked = ok.merge(pd.DataFrame(keep, columns=["app_id", "as_of_date"]), on=["app_id", "as_of_date"])
    picked["as_of_date"] = picked.as_of_date.astype(str)
    return picked.sort_values(["app_id", "as_of_date", "category"]).reset_index(drop=True)


# ---- model card -----------------------------------------------------------------

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def cell(v):
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else \
            floatfmt.format(v) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def model_card(m: dict, scores: pd.DataFrame, counts: pd.DataFrame, importance: pd.DataFrame) -> str:
    mv = m["model_version"]
    pooled = scores[scores.app_id == "all"]

    def mae_table(split: str) -> str:
        t = pooled[pooled.split == split].pivot_table(index="category", columns="method", values="mae_pp", sort=False)
        t = t[[mv, *BASELINES]].reset_index()
        if split == "validation":
            t["winner"] = t.category.map(m["validation_winner"]).fillna("")
        return _md_table(t)

    params = pd.DataFrame([{"category": c, **v["params"], "shrinkage": v.get("shrinkage"), "fit_rows": v["fit_rows"]}
                           for c, v in m["categories"].items()])
    if params.shrinkage.isna().all():
        params = params.drop(columns="shrinkage")
    residual = m.get("target", "").startswith("share_28d")
    how = ("A Spark GBT regressor per category learns the correction to the rolling 28-day share "
           "(`target_share_next7 - share_28d`); the prediction is `share_28d + shrinkage x correction`, with the "
           "shrinkage chosen on validation. The baseline supplies the level, so the model can follow level shifts "
           "that never occurred in training, and it falls back to the baseline when it learns nothing."
           if residual else "One Spark GBT regressor per category predicts the share directly.")
    first = counts[counts.category == config.CATEGORIES[0]].drop(columns="category")
    top = (importance.sort_values("importance", ascending=False).groupby("category").head(5)
           .groupby("category").apply(lambda g: ", ".join(f"{f} ({v:.2f})" for f, v in zip(g.feature, g.importance)),
                                      include_groups=False).reindex(config.CATEGORIES).reset_index(name="top features"))
    test = mae_table("test") if "test" in set(pooled.split) else \
        "Not scored yet. Run `make train TEST=1` once, after every choice is frozen."
    return f"""# Model card: {mv} ({m['label_version']})

Generated by `pipeline/train.py` at {m['created_at']} (MLflow run `{m['mlflow_run_id']}`, commit `{(m['code'].get('commit') or '')[:12]}`). Every number comes from the files next to this one.

## What it predicts

For one game, one complaint category and an as-of date t: the share of English negative reviews over the next seven days (t+1..t+7) that fall in that category. {how} Models are trained on all games pooled, on the {len(m['features'])} features in `feature_manifest.json`. Output is clipped to [0, 1].

## Intended use

Pointing at complaint categories worth a closer look, in a historical replay of the selected games. It doesn't identify root causes, judge severity or set fix priorities, and it hasn't been tested on a live feed.

## Data and versions

| | |
|---|---|
| Snapshot | `{m['snapshot_id'][:7]}` ({config.HF_DATASET}) |
| Cohort | `{m['cohort_policy']}` / `{m['cohort_version']}` |
| Labels | `{m['label_version']}` (rules in `pipeline/labels.py`) |
| Features | `{m['feature_version']}`, {len(m['features'])} features |
| Splits | train targets by {m['splits']['train_end']}, validation by {m['splits']['val_end']}, test after; one purged week at each boundary |
| Deployed model trained on targets through | {m['training_label_end']} (replays start here) |

## Rows per split

Eligibility is decided per game and day, so every category has the same counts. `scored` rows are eligible and have a target week; the other columns are exclusions.

{_md_table(first)}

## Settings chosen on validation

{_md_table(params)}

## Validation MAE (percentage points, identical rows)

{mae_table('validation')}

## Test MAE (percentage points, identical rows)

{test}

Per-game results and volume-weighted MAE are in `model_scores.csv`; the largest errors are in `model_errors.csv`.

## Most important features (deployed models)

{_md_table(top)}

## Limitations

- Targets come from keyword rules, so label errors carry into the targets. Rule precision and recall are in `label_eval_*.csv`.
- The strict unedited cohort drops 14-19% of English negatives, more of them from older reviews (`docs/temporal_policy.md`).
- Seven-day targets of neighbouring days overlap, so daily errors aren't independent. No significance tests or confidence intervals are claimed.
- A few games over one year: the pooled models may not transfer to other games or periods.
- A validation win over persistence doesn't promise a test win. The test result is reported as it comes out.
"""


# ---- run ----------------------------------------------------------------------

def gbt_stage(model: PipelineModel):
    return next(st for st in model.stages if hasattr(st, "featureImportances"))


def run(feats: DataFrame, label_version: str, out_dir: Path = STAGING, grid: list[dict] = GRID,
        score_test: bool = False, tracking_uri: str | None = None, target: str = "residual",
        shrinkage: list[float] = SHRINKAGE, seed: int = SEED) -> dict:
    """Search, compare, refit and export. Returns the model manifest."""
    model_version = TARGETS[target]
    lambdas = shrinkage if target == "residual" else [None]
    mlflow.set_tracking_uri(tracking_uri or MLRUNS.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    feature_types = {f.name: f.dataType.simpleString() for f in feats.schema.fields if f.name in config.FEATURE_COLS}
    # A few thousand rows: a handful of partitions avoids 200-way shuffles on every fit.
    feats = feats.withColumn("split", split_col()).coalesce(4).cache()
    data = with_residual(feats.where(usable())).cache()
    fit_end = F.date_add("as_of_date", 7) <= F.lit(config.VAL_END).cast("date")

    model_dir = out_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    scores, errors, importance, categories, models, label_ends = [], [], [], {}, {}, []

    with mlflow.start_run(run_name=f"{model_version} {label_version}") as parent:
        mlflow.set_tags({"model_version": model_version, "label_version": label_version})
        mlflow.log_params({
            "label_version": label_version, "feature_version": config.FEATURE_VERSION,
            "cohort": f"{config.COHORT_POLICY}/{config.COHORT_VERSION}", "snapshot_id": config.SNAPSHOT_ID[:7],
            "train_end": config.TRAIN_END, "val_end": config.VAL_END, "seed": seed,
            "games": ",".join(sorted(r.app_id for r in data.select("app_id").distinct().collect())),
            "grid": json.dumps(grid), "code_commit": (_code_revision().get("commit") or "")[:12],
            "target": target, "shrinkage": json.dumps(lambdas),
        })
        for category in config.CATEGORIES:
            rows = data.where(F.col("category") == category)
            train, val = rows.where(F.col("split") == "train"), rows.where(F.col("split") == "validation")
            if train.limit(1).count() == 0 or val.limit(1).count() == 0:
                raise SystemExit(f"{category}: no eligible train or validation rows with a target.")

            search = []  # (validation MAE, GBT params, shrinkage, fitted model)
            for params in grid:
                with mlflow.start_run(run_name=f"{category} {params}", nested=True):
                    fitted = build_pipeline(params, seed, target).fit(train)
                    scored = fitted.transform(val).cache()
                    metrics = {}
                    for lam in lambdas:
                        pred = F.col("prediction") if lam is None else F.col(BASE_FILLED) + lam * F.col(RESIDUAL_PRED)
                        mae, n = mae_pp(scored, pred)
                        metrics["val_mae_pp" if lam is None else f"val_mae_pp_shrinkage_{lam}"] = mae
                        search.append((mae, params, lam, fitted))
                    mlflow.log_params({"category": category, **params})
                    mlflow.log_metrics({**metrics, "val_rows": n})
            _, best, best_lam, fitted = min(search, key=lambda s: s[0])
            train_model = finalize(fitted, target, best_lam)

            scores += score(train_model.transform(train), model_version, "train", category)  # in-sample
            val_scored = train_model.transform(val).cache()
            scores += score(val_scored, model_version, "validation", category)
            errors.append(largest_errors(val_scored, model_version, "validation"))

            fit_rows = rows.where(fit_end)
            final = finalize(build_pipeline(best, seed, target).fit(fit_rows), target, best_lam)
            label_end = fit_rows.select(F.date_add(F.max("as_of_date"), 7)).first()[0].isoformat()
            label_ends.append(label_end)
            if score_test:
                test_scored = final.transform(rows.where(F.col("split") == "test")).cache()
                scores += score(test_scored, model_version, "test", category)
                errors.append(largest_errors(test_scored, model_version, "test"))

            final.write().overwrite().save(str(model_dir / category))
            models[category] = final
            importance += [dict(category=category, feature=f, importance=float(v))
                           for f, v in zip(config.FEATURE_COLS, gbt_stage(final).featureImportances.toArray())]
            val_scores = {s["method"]: s["mae_pp"] for s in scores
                          if s["split"] == "validation" and s["category"] == category and s["app_id"] == "all"}
            categories[category] = {
                "path": f"model/{category}", "params": best, "shrinkage": best_lam, "fit_rows": fit_rows.count(),
                "training_label_end": label_end, "val_mae_pp": val_scores,
                "validation_winner": min(val_scores, key=val_scores.get),
            }
            mlflow.log_metrics({f"val_mae_pp.{category}.{m.split('_v')[0]}": v
                                for m, v in val_scores.items() if v is not None})

        scores_df = pd.DataFrame(scores)
        pooled = scores_df[scores_df.app_id == "all"]
        macro = (pooled.groupby(["split", "method"], as_index=False, sort=False)
                 .agg(rows=("rows", "sum"), mae_pp=("mae_pp", "mean"), mae_pp_weighted=("mae_pp_weighted", "mean"))
                 .assign(category="macro", app_id="all"))
        scores_df = pd.concat([scores_df, macro[scores_df.columns]], ignore_index=True)
        training_label_end = max(label_ends)

        preds = predictions(feats.drop("split"), models, model_version, training_label_end).cache()
        (preds.repartition("app_id").write.mode("overwrite").partitionBy("app_id")
         .parquet(str(out_dir / PREDICTIONS_DIR.name)))

        manifest = {
            "format": "spark_pipeline",
            "model_version": model_version,
            "target": TARGET_FORMULAS[target],
            "training_label_end": training_label_end,
            "replay_start": training_label_end,
            "feature_version": config.FEATURE_VERSION,
            "label_version": label_version,
            "snapshot_id": config.SNAPSHOT_ID,
            "cohort_policy": config.COHORT_POLICY,
            "cohort_version": config.COHORT_VERSION,
            "features": list(config.FEATURE_COLS),
            "feature_types": {c: feature_types[c] for c in config.FEATURE_COLS},
            "prediction_column": "prediction",
            "prediction_clip": [0.0, 1.0],
            "categories": categories,
            "splits": {"train_end": config.TRAIN_END, "val_end": config.VAL_END,
                       "rule": "by target week t+1..t+7, purged week at each boundary (pipeline/splits.py)"},
            "validation_winner": {c: v["validation_winner"] for c, v in categories.items()},
            "parity": {"file": PARITY_PATH.name, "tolerance": PARITY_TOLERANCE},
            "seed": seed,
            "mlflow_run_id": parent.info.run_id,
            "code": _code_revision(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        counts = row_counts(feats)
        importance_df = pd.DataFrame(importance)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / MODEL_MANIFEST.name).write_text(json.dumps(manifest, indent=2) + "\n")
        parity_rows(preds).to_csv(out_dir / PARITY_PATH.name, index=False)
        scores_df.to_csv(out_dir / SCORES_PATH.name, index=False)
        counts.to_csv(out_dir / ROW_COUNTS_PATH.name, index=False)
        pd.concat(errors, ignore_index=True).to_csv(out_dir / ERRORS_PATH.name, index=False)
        importance_df.to_csv(out_dir / IMPORTANCE_PATH.name, index=False)
        (out_dir / CARD_PATH.name).write_text(model_card(manifest, scores_df, counts, importance_df))

        for m in scores_df[scores_df.category == "macro"].itertuples():
            if m.mae_pp is not None and not pd.isna(m.mae_pp):
                mlflow.log_metric(f"{m.split}_mae_pp.macro.{m.method.split('_v')[0]}", m.mae_pp)
        for f in OUTPUT_FILES:
            mlflow.log_artifact(str(out_dir / f.name))
        if (out_dir / FEATURE_MANIFEST.name).exists():
            mlflow.log_artifact(str(out_dir / FEATURE_MANIFEST.name))
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
    return manifest


def main(label_version: str, score_test: bool, target: str = "residual", out_dir: Path = STAGING) -> None:
    spark = get_spark("steampulse-train")
    feats = spark.read.parquet(str(FEATURES_DIR)).where(
        (F.col("label_version") == label_version) & (F.col("feature_version") == config.FEATURE_VERSION))
    if feats.limit(1).count() == 0:
        raise SystemExit(f"No forecast_features with label_version {label_version!r}. Run `make features` first.")
    if not FEATURE_MANIFEST.exists():
        raise SystemExit("data/staging/feature_manifest.json is missing. Run `make features` first.")
    fm = json.loads(FEATURE_MANIFEST.read_text())
    if [f["name"] for f in fm["features"]] != config.FEATURE_COLS or fm["label_version"] != label_version:
        raise SystemExit("feature_manifest.json doesn't match FEATURE_COLS or this label_version. Rerun `make features`.")
    manifest = run(feats, label_version, out_dir=out_dir, score_test=score_test, target=target)

    scores = pd.read_csv(out_dir / SCORES_PATH.name)
    shown = scores[scores.split.isin(["validation", "test"]) & (scores.app_id == "all")]
    table = shown.pivot_table(index=["split", "category"], columns="method", values="mae_pp", sort=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print("MAE (percentage points), identical rows per split and category:")
        print(table.to_string())
    print(f"\nValidation winner: {manifest['validation_winner']}")
    print(f"Deployed {manifest['model_version']} trained on targets through {manifest['training_label_end']}; "
          f"MLflow run {manifest['mlflow_run_id']}")
    print(f"Shrinkage per category: { {c: v['shrinkage'] for c, v in manifest['categories'].items()} }")
    print(f"Wrote {out_dir}/: model/, {PREDICTIONS_DIR.name}/ and " + ", ".join(f.name for f in OUTPUT_FILES))
    if out_dir != STAGING:
        print("This output is outside data/staging, so checks and publish won't pick it up (comparison only).")
    if not score_test:
        print("Test period not scored. Run with --score-test once, after every choice is frozen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-version", default=config.LABEL_VERSION)
    parser.add_argument("--score-test", action="store_true", help="score the held-out test period (do this once)")
    parser.add_argument("--target", choices=list(TARGETS), default="residual",
                        help="residual = gbt_v2 (deployed), level = gbt_v1 (comparison)")
    parser.add_argument("--out-dir", type=Path, default=STAGING,
                        help="where to write; use another folder for comparison runs")
    args = parser.parse_args()
    main(args.label_version, args.score_test, args.target, args.out_dir.resolve())
