"""Rank candidate games for modeling using the training period only.

    python -m pipeline.select_games      (after `make profile`)

Reads data/profile/ and the raw snapshot, writes data/profile/candidates.csv.
Two parts:

1. Eligibility from daily counts: strict English negatives per day, the share of
   training days that pass the N7/N28 guards, and days with no reviews at all.
2. A rough complaint mix: keyword matches in strict English negative reviews.
   This is a proxy to check each category has enough support. It is NOT the
   labeling rules, which are built and evaluated separately.

Only training-period data is used, so the choice of games does not leak
information from validation or test.
"""
import time

import pandas as pd
import pyarrow.parquet as pq
from pyspark.sql import functions as F

from pipeline import config
from pipeline.config import get_spark

# Tier cutoffs
CORE = {"min_pct_eligible": 80, "max_zero_days": 7}
NEAR_MISS = {"min_pct_eligible": 40, "max_zero_days": 40}
MIN_TRAIN_STRICT = 1000  # skip tiny games before the expensive steps

# Keyword proxies per category (lowercased text, regex word boundaries).
KEYWORDS = {
    "crash": r"\b(crash\w*|freez\w*|ctd|black screen|won'?t (launch|start|open)|doesn'?t (launch|start)"
             r"|not launching|fails? to launch|blue screen|bsod)\b",
    "performance": r"\b(fps|frame ?rates?|stutter\w*|optimi[sz]\w*|performance|frame drops?|low frames?"
                   r"|loading times?|long load\w*|gpu usage|cpu usage|memory leak)\b",
    "match_server": r"\b(matchmaking|match making|servers?|disconnect\w*|desync\w*|queue\w*|ping|latency"
                    r"|rubber ?band\w*|sbmm|lobby|lobbies|connection)\b",
    "monetization": r"\b(microtransactions?|micro transactions?|mtx|pay ?(to|2) ?win|p2w|battle ?pass"
                    r"|loot ?box\w*|cash ?grab|overpriced|money grab|predatory|gacha|premium currency"
                    r"|dlcs?|price|\$\d+)\b",
}


def train_eligibility(daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Per-game training-period eligibility from profile daily counts.

    `daily` has appid, date, all_reviews, en_neg, en_neg_strict (days with no
    reviews are absent). A forecast origin t is eligible when the strict N7
    (t-6..t) >= MIN_N7 and N28 (t-34..t-7) >= MIN_N28.
    The 34 days before `start` are included so early origins have history.
    """
    lookback = pd.Timestamp(start) - pd.Timedelta(days=config.HISTORY_DAYS - 1)
    days = pd.date_range(lookback, end)
    origins = pd.date_range(start, end)
    daily = daily.assign(date=pd.to_datetime(daily["date"]))
    daily = daily[(daily["date"] >= lookback) & (daily["date"] <= pd.Timestamp(end))]

    rows = []
    for app, g in daily.groupby("appid"):
        g = g.set_index("date")
        in_train = g.index >= pd.Timestamp(start)
        if g.loc[in_train, "en_neg_strict"].sum() < MIN_TRAIN_STRICT:
            continue
        strict = g["en_neg_strict"].reindex(days, fill_value=0)
        n7 = strict.rolling(7).sum()
        n28 = strict.shift(7).rolling(28).sum()
        eligible = ((n7 >= config.MIN_N7) & (n28 >= config.MIN_N28)).reindex(origins)
        all_reviews = g["all_reviews"].reindex(origins, fill_value=0)
        en_neg = g.loc[in_train, "en_neg"].sum()
        train_strict = int(strict.reindex(origins).sum())
        rows.append({
            "appid": app,
            "train_strict": train_strict,
            "per_day": train_strict / len(origins),
            "pct_eligible": 100 * eligible.mean(),
            "zero_days": int((all_reviews == 0).sum()),
            "edit_share": 1 - train_strict / en_neg if en_neg else None,
        })
    return pd.DataFrame(rows)


def assign_tier(df: pd.DataFrame) -> pd.Series:
    def tier(r) -> str:
        if r.pct_eligible >= CORE["min_pct_eligible"] and r.zero_days <= CORE["max_zero_days"]:
            return "core"
        if r.pct_eligible >= NEAR_MISS["min_pct_eligible"] and r.zero_days <= NEAR_MISS["max_zero_days"]:
            return "near_miss"
        return "excluded"
    return df.apply(tier, axis=1)


def keyword_mix(spark, appids: list[float], start: str, end: str) -> pd.DataFrame:
    """Keyword matches per category in strict English negatives, per game."""
    raw = spark.read.parquet(str(config.RAW_DIR / "part.*.parquet"))
    reviews = (
        raw.select("appid", "language", "voted_up", "review", "timestamp_created", "timestamp_updated")
        .where(F.col("appid").isin(appids)
               & (F.col("language") == config.LANGUAGE)
               & (F.col("voted_up") == 0)
               & (F.col("timestamp_created") == F.col("timestamp_updated")))
        .withColumn("date", F.to_date(F.timestamp_seconds("timestamp_created")))
        .where(F.col("date").between(start, end))
        .withColumn("text", F.lower(F.coalesce("review", F.lit(""))))
    )
    any_match = "|".join(KEYWORDS.values())
    aggs = [F.count("*").alias("n_scanned")]
    aggs += [F.sum(F.col("text").rlike(p).cast("int")).alias(f"{c}_hits") for c, p in KEYWORDS.items()]
    aggs.append(F.sum((~F.col("text").rlike(any_match)).cast("int")).alias("no_match_hits"))
    return reviews.groupBy("appid").agg(*aggs).toPandas()


def main() -> None:
    start, end = config.WINDOW_START, config.TRAIN_END
    t0 = time.perf_counter()
    profile = pd.read_csv(config.PROFILE_DIR / "game_profile.csv")
    daily = pq.read_table(config.PROFILE_DIR / "daily_counts").to_pandas()

    elig = train_eligibility(daily, start, end).merge(
        profile[["appid", "game", "first_date", "last_date"]], on="appid")
    # Must exist when training starts and have data through the end of the window.
    elig = elig[(elig["first_date"] <= start) & (elig["last_date"] >= config.WINDOW_END)]
    elig["tier"] = assign_tier(elig)
    elig = elig[elig["tier"] != "excluded"]

    spark = get_spark("steampulse-select-games")
    mix = keyword_mix(spark, [float(a) for a in elig["appid"]], start, end)
    out = elig.merge(mix, on="appid", how="left")
    n_days = len(pd.date_range(start, end))
    for c in [*KEYWORDS, "no_match"]:
        out[f"{c}_per_day"] = out[f"{c}_hits"] / n_days
        out[f"{c}_share"] = out[f"{c}_hits"] / out["n_scanned"]
    out = out.sort_values(["tier", "per_day"], ascending=[True, False])

    path = config.PROFILE_DIR / "candidates.csv"
    out.to_csv(path, index=False)

    show = ["tier", "appid", "game", "per_day", "pct_eligible", "zero_days", "edit_share"]
    show += [f"{c}_per_day" for c in KEYWORDS]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(out[show].round(2).to_string(index=False))
    print(f"\n{(out.tier == 'core').sum()} core, {(out.tier == 'near_miss').sum()} near-miss "
          f"(training {start}..{end}); wrote {path} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
