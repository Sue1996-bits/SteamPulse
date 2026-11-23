"""Rule-based complaint labels for English negative reviews -> review_labels.

    python -m pipeline.labels                          # all chosen games, label_version from config
    python -m pipeline.labels --label-version rules_v2 --apps 1172470

Each category has an ordered list of rules. A rule is a regular expression run on
text_normalized (lowercase, BBCode and URLs removed; see clean.normalize_text).
The first rule that matches gives the row its rule_id, and the matched text is
saved as evidence, so every label can be traced back to the words that caused it.

Negation: rules marked negatable ignore mentions like "no crashes", "never lags"
or "not pay to win". Before those rules run, a negator (no, not, never, without,
didn't, ...) followed by up to three words and the rule's term is blanked out.
"not only", "can't stop" and "won't stop" are not treated as negation. Rules that
are complaints even when negated ("not optimized", "won't launch") are not negatable.

Output: one row per review and assigned category (review_labels schema). No row
means the category wasn't assigned; the review still counts in negative_total as
unclassified. Only English negatives are labeled, since nothing else is counted.

Writes data/staging/review_labels/, partitioned by label_version, so several
versions can sit side by side and a rerun replaces only its own version, and
data/staging/label_coverage_<version>.csv (labeled vs unclassified, per game).
Change any rule -> bump LABEL_VERSION in config.py.
"""
import argparse
from pathlib import Path

import pandas as pd
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from pipeline import config
from pipeline.clean import OUT_DIR as CLEAN_DIR
from pipeline.config import get_spark

OUT_DIR = config.STAGING_DIR / "review_labels"
COVERAGE_PATH = config.STAGING_DIR / "label_coverage_{version}.csv"
EVIDENCE_CHARS = 120

# (rule_id, pattern, negatable). Patterns are Java regex (Spark rlike) on lowercase text.
RULES: dict[str, list[tuple[str, str, bool]]] = {
    "crash": [
        ("crash_word", r"\bcrash(es|ed|ing|y)?\b", True),
        ("crash_desktop", r"\bctd\b|\bbsod\b|\bblue ?screens?\b|\bfatal error\b", True),
        ("crash_freeze", r"\bfreez(e|es|ing)\b|\bfroze\b|\bnot responding\b", True),
        ("crash_launch", r"\b(won'?t|doesn'?t|does not|can'?t|cannot|unable to|fails? to|failed to|refuses? to)"
                         r" (even )?(launch|boot|start up|open)\b|\b(launch|startup) (issues?|problems?|errors?)\b",
         False),
    ],
    "performance": [
        ("perf_fps", r"\bfps\b|\bframe ?rates?\b|\bframes? per second\b|\bframe ?drops?\b|\bdropped frames\b", True),
        ("perf_stutter", r"\b(micro ?)?stutter(s|ing|y)?\b|\binput lag\b", True),
        ("perf_optimization", r"\bun-?optimi[sz]ed\b|\boptimi[sz](ation|ed)\b", False),
        ("perf_bad", r"\b(bad|poor|terrible|horrible|awful|worse|worst|garbage|trash|abysmal) performance\b"
                     r"|\bperformance (issues?|problems?)\b|\bruns (like )?(garbage|trash|shit|crap|terribl[ey]"
                     r"|horribl[ey]|awful|badly|poorly)\b|\bslideshow\b", False),
        ("perf_loading", r"\b(long|slow|endless|infinite|insane) load(ing)?( times?| screens?)?\b"
                         r"|\bstuck (on|at|in) (the |a )?loading\b"
                         r"|\bload(ing)? times? (is|are) (so |really |very )?(long|slow|terrible|horrible|awful)\b",
         False),
        ("perf_resources", r"\bmemory leaks?\b|\b(high|100%) (cpu|gpu|ram|memory|vram)( usage)?\b"
                           r"|\boverheat(s|ing)?\b", False),
    ],
    "match_server": [
        ("server_matchmaking", r"\bmatch ?making\b|\bsbmm\b|\bskill[- ]based match", False),
        ("server_queue", r"\bqueue(s|d)?\b|\bqueue times?\b|\bqueuing\b"
                         r"|\b(can'?t|cannot|unable to) (find|get into) (a )?(match|game|lobby)\b", False),
        ("server_disconnect", r"\bdisconnect(s|ed|ing|ions?)?\b|\bdc(s|'?d|ed)?\b|\blost connection\b"
                              r"|\bconnection (issues?|problems?|errors?)\b"
                              r"|\bkicked (from|out of) (the |a )?(match|game|server|lobby)\b", True),
        # "no servers" or "can't connect to servers" is still a server complaint, so not negatable.
        ("server_quality", r"\bservers?\b|\bnetcode\b|\btick ?rate\b|\bhit ?reg(istration)?\b", False),
        # Plain "lag" is read as network lag in these online games; "input lag" is performance.
        ("server_lag", r"(?<!input )\blag(s|gy|ging|ged)?\b|\bdesync\w*|\brubber ?band\w*"
                       r"|\bhigh ping\b|\bpacket loss\b|\blatency\b", True),
    ],
    "monetization": [
        ("money_mtx", r"\bmicro ?-?transactions?\b|\bmtx\b|\bloot ?box(es)?\b|\bgacha\b", True),
        ("money_p2w", r"\bpay(ing)? ?(2|to) ?(win|progress)\b|\bp2w\b", True),
        ("money_greed", r"\b(cash|money) ?grab\w*|\bcash ?cow\b|\bgreed(y|iness)?\b"
                        r"|\bmilk(s|ing|ed)? (the )?(players|community|customers|money|fans|us)\b", False),
        ("money_price", r"\bover-?priced\b|\btoo expensive\b|\bnot worth (the )?(price|money|\$)"
                        r"|\$ ?\d+ (for (a|one) )?(skins?|bundles?|cosmetics?|battle ?pass)\b"
                        r"|\b(skins?|bundles?|cosmetics?) (cost|costs|for) (\$|€|£)? ?\d+", False),
        ("money_store", r"\bbattle ?pass(es)?\b|\b(fifa|apex|r6) (points|coins|credits)\b|\bpack (odds|weight)\b"
                        r"|\b(paywall|pay ?wall)(ed|s)?\b", True),
    ],
}

NEGATOR = (r"\b(no|not|never|without|zero|hardly|barely|rarely|didn'?t|doesn'?t|don'?t|isn'?t|wasn'?t"
           r"|aren'?t|weren'?t|haven'?t|hasn'?t|hadn'?t|won'?t|wouldn'?t|can'?t|cannot)\b"
           r"(?!\W+(only|stop|stops|stopping)\b)")


def negation_pattern(category: str) -> str | None:
    """Negator, up to three words, then any negatable term of the category."""
    terms = [p for _, p, negatable in RULES[category] if negatable]
    if not terms:
        return None
    return NEGATOR + r"(\W+\w+){0,3}?\W+(" + "|".join(f"(?:{t})" for t in terms) + ")"


def _first(pairs: list[tuple[Column, Column]]) -> Column:
    """Value of the first pair whose condition holds, else null."""
    out = F.lit(None).cast("string")
    for cond, value in reversed(pairs):
        out = F.when(cond, value).otherwise(out)
    return out


def match_category(text: Column, category: str) -> Column:
    """struct(category, rule_id, evidence) for the first matching rule, or null."""
    neg = negation_pattern(category)
    masked = F.regexp_replace(text, neg, " ") if neg else text
    hits = []
    for rule_id, pattern, negatable in RULES[category]:
        src = masked if negatable else text
        hits.append((src.rlike(pattern), F.lit(rule_id),
                     F.substring(F.regexp_extract(src, pattern, 0), 1, EVIDENCE_CHARS)))
    rule_id = _first([(c, r) for c, r, _ in hits])
    evidence = _first([(c, e) for c, _, e in hits])
    return F.when(rule_id.isNotNull(),
                  F.struct(F.lit(category).alias("category"), rule_id.alias("rule_id"),
                           evidence.alias("evidence")))


def label_reviews(reviews: DataFrame, label_version: str) -> DataFrame:
    """reviews with review_id and text_normalized -> review_labels rows (one per assigned category)."""
    matches = F.array(*[match_category(F.col("text_normalized"), c) for c in config.CATEGORIES])
    return (reviews.select("review_id", F.explode(F.filter(matches, lambda m: m.isNotNull())).alias("m"))
            .select("review_id", F.col("m.category").alias("category"), F.lit(True).alias("is_match"),
                    F.col("m.rule_id").alias("rule_id"), F.col("m.evidence").alias("evidence"),
                    F.lit(label_version).alias("label_version"))
            .select(*config.REVIEW_LABELS_COLS))


def coverage(neg: DataFrame, labels: DataFrame) -> DataFrame:
    """Per game and overall: English negatives, labeled, unclassified share and count per category."""
    per_review = (labels.groupBy("review_id").agg(F.collect_set("category").alias("cats")))
    joined = neg.select("review_id", "app_id").join(per_review, "review_id", "left")
    aggs = [F.count("*").alias("negatives"),
            F.sum(F.col("cats").isNotNull().cast("long")).alias("labeled")]
    aggs += [F.sum(F.coalesce(F.array_contains("cats", c), F.lit(False)).cast("long")).alias(c)
             for c in config.CATEGORIES]
    per_game = joined.groupBy("app_id").agg(*aggs)
    total = joined.agg(*aggs).withColumn("app_id", F.lit("all")).select(*per_game.columns)
    return (per_game.unionByName(total)
            .withColumn("unclassified_share", 1 - F.col("labeled") / F.col("negatives"))
            .orderBy(F.col("app_id") == "all", "app_id"))


def english_negatives(clean: DataFrame) -> DataFrame:
    return clean.where((F.col("language") == config.LANGUAGE) & ~F.col("voted_up"))


def main(label_version: str, apps: list[str] | None = None) -> None:
    spark = get_spark("steampulse-labels")
    clean = spark.read.parquet(str(CLEAN_DIR))
    if apps:
        clean = clean.where(F.col("app_id").isin(apps))
    neg = english_negatives(clean).select("review_id", "app_id", "text_normalized").cache()
    labels = label_reviews(neg, label_version).cache()
    # Dynamic partition overwrite (get_spark): only this label_version's folder is replaced.
    labels.coalesce(4).write.mode("overwrite").partitionBy("label_version").parquet(str(OUT_DIR))

    cov = coverage(neg, labels).toPandas()
    out = Path(str(COVERAGE_PATH).format(version=label_version))
    cov.assign(label_version=label_version).to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(cov.to_string(index=False))
    rules = labels.groupBy("category", "rule_id").count().orderBy("category", F.desc("count"))
    rules.show(50, truncate=False)
    print(f"Coverage -> {out.relative_to(config.ROOT)}")
    print(f"Wrote {OUT_DIR.relative_to(config.ROOT)}/label_version={label_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-version", default=config.LABEL_VERSION)
    parser.add_argument("--apps", nargs="+", help="subset of app IDs (default: every game in reviews_clean)")
    args = parser.parse_args()
    main(args.label_version, args.apps)
