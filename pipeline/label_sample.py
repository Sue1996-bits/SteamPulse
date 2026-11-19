"""Draw the hand-labeling sample from reviews_clean.

    python -m pipeline.label_sample                   # Apex, 300 + 150, seed 226
    python -m pipeline.label_sample --apps 1172470 1517290 --seed 7

Samples English negative reviews in two groups:
  train  created WINDOW_START..TRAIN_END   (default 300) for writing the labeling rules
  later  created after TRAIN_END..WINDOW_END (default 150) for checking that the
         rules still hold later on; don't tune rules on this group

Each review's place in the draw comes from a hash of the seed and its review_id,
so the same seed always gives the same sample, regardless of how Spark splits the data.

Writes to data/samples/:
  label_sample_<seed>.csv    one row per review, in random order, with empty
                             columns for the four categories and notes
  label_sample_<seed>.json   where the sample came from and how it was drawn
"""
import argparse
import json
from datetime import datetime, timezone

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from pipeline import config
from pipeline.clean import OUT_DIR as CLEAN_DIR
from pipeline.config import get_spark
from pipeline.manifest import _code_revision

SAMPLES_DIR = config.DATA / "samples"
DEFAULT_APPS = ["1172470"]  # Apex, the first game
SAMPLE_COLS = ["review_id", "app_id", "game", "review_date", "period", "text_raw"]


def draw(clean: DataFrame, n_train: int, n_later: int, seed: int,
         start: str = config.WINDOW_START, train_end: str = config.TRAIN_END,
         end: str = config.WINDOW_END) -> DataFrame:
    """Exactly n_train and n_later English negatives (or all of them, if fewer)."""
    d = F.col("review_date")
    period = (F.when((d >= F.lit(start).cast("date")) & (d <= F.lit(train_end).cast("date")), "train")
              .when((d > F.lit(train_end).cast("date")) & (d <= F.lit(end).cast("date")), "later"))
    pool = (clean.where((F.col("language") == config.LANGUAGE) & ~F.col("voted_up"))
            .withColumn("period", period).where(F.col("period").isNotNull())
            .withColumn("draw", F.xxhash64(F.lit(seed), F.col("review_id"))))
    w = Window.partitionBy("period").orderBy("draw", "review_id")
    n = F.when(F.col("period") == "train", n_train).otherwise(n_later)
    return (pool.withColumn("_rank", F.row_number().over(w))
            .where(F.col("_rank") <= n).drop("_rank"))


def main(apps: list[str], n_train: int, n_later: int, seed: int) -> None:
    spark = get_spark("steampulse-label-sample")
    clean = spark.read.parquet(str(CLEAN_DIR)).where(F.col("app_id").isin(apps))
    pool_sizes = {
        r.period: r["count"]
        for r in draw(clean, 10**12, 10**12, seed).groupBy("period").count().collect()
    }
    keep = [c for c in SAMPLE_COLS if c != "game"] + ["draw"]
    sample = draw(clean, n_train, n_later, seed).select(*keep).toPandas()
    sample["game"] = sample["app_id"].map(config.GAMES)
    sample = sample.sort_values("draw")[SAMPLE_COLS]  # random order, so periods and dates are mixed
    for c in [*config.CATEGORIES, "notes"]:
        sample[c] = ""

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SAMPLES_DIR / f"label_sample_{seed}.csv"
    sample.to_csv(csv_path, index=False)

    got = sample["period"].value_counts().to_dict()
    provenance = {
        "file": csv_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "data/staging/reviews_clean",
        "snapshot_id": config.SNAPSHOT_ID,
        "cohort_policy": config.COHORT_POLICY,
        "apps": {a: config.GAMES.get(a) for a in apps},
        "filter": f"language == '{config.LANGUAGE}' and not voted_up",
        "seed": seed,
        "method": "lowest xxhash64(seed, review_id) within each period",
        "periods": {
            "train": {"dates": [config.WINDOW_START, config.TRAIN_END], "requested": n_train,
                      "sampled": got.get("train", 0), "pool": pool_sizes.get("train", 0)},
            "later": {"dates": [config.TRAIN_END, config.WINDOW_END], "date_rule": "after TRAIN_END",
                      "requested": n_later, "sampled": got.get("later", 0), "pool": pool_sizes.get("later", 0)},
        },
        "code": _code_revision(),
        "license": "CC BY-NC 4.0 (adde023/steam-reviews on Hugging Face, collected by kieranpoc on Kaggle)",
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{got.get('train', 0)} train + {got.get('later', 0)} later reviews "
          f"(pools: {pool_sizes.get('train', 0):,} / {pool_sizes.get('later', 0):,}) -> {csv_path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apps", nargs="+", default=DEFAULT_APPS)
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--later", type=int, default=150)
    parser.add_argument("--seed", type=int, default=226)
    args = parser.parse_args()
    main(args.apps, args.train, args.later, args.seed)
