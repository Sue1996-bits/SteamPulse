"""Labeling rules on hand-written examples: matches, negation, multi-label and the output schema."""
import pytest
from pyspark.sql import functions as F

from pipeline import config, labels
from pipeline.clean import normalize_text
from pipeline.config import get_spark

VERSION = "test_v1"

# (text, categories expected). Lowercasing and BBCode removal come from normalize_text.
CASES = [
    ("Game CRASHES every match.", {"crash"}),
    ("Crashed to desktop twice, then a blue screen.", {"crash"}),
    ("It won't launch at all after the update", {"crash"}),
    ("Freezes on the main menu", {"crash"}),
    ("I have never had a crash in 200 hours, but the servers are trash", {"match_server"}),
    ("no crashes, no lag, just boring", set()),
    ("it doesn't even crash, it just stutters", {"performance"}),
    ("Not only does it crash, the fps drops to 20", {"crash", "performance"}),
    ("can't stop crashing", {"crash"}),
    ("poorly optimized, not optimized at all", {"performance"}),
    ("Loading times are so long", {"performance"}),
    ("I have an RTX 4090 and an i9", set()),                       # hardware alone isn't performance
    ("input lag is awful", {"performance"}),
    ("constant lag and rubberbanding", {"match_server"}),
    ("matchmaking puts me with level 500 players", {"match_server"}),
    ("Disconnected from the match three times", {"match_server"}),
    ("[b]Pay to win[/b] garbage", {"monetization"}),
    ("it's not pay to win, the gameplay is just bad", set()),
    ("$20 for a skin? greedy EA", {"monetization"}),
    ("I bought it for $30 on sale", set()),                        # a neutral price mention
    ("crashes, 30 fps, queue times of 10 minutes and microtransactions everywhere",
     {"crash", "performance", "match_server", "monetization"}),
    ("not fun", set()),
    ("best fps game ever made, too bad about the cheaters", set()),      # "fps" as the genre
    ("a fun goofy fps, but the servers are trash", {"match_server"}),
    ("i get 40fps on low settings", {"performance"}),
    ("my fps tanks after an hour", {"performance"}),
    ("no fps drops at all, just a boring game", set()),
]


@pytest.fixture(scope="module")
def labeled():
    spark = get_spark("test")
    reviews = (spark.createDataFrame([(str(i), t) for i, (t, _) in enumerate(CASES)], ["review_id", "text_raw"])
               .withColumn("text_normalized", normalize_text(F.col("text_raw"))))
    return labels.label_reviews(reviews, VERSION).toPandas()


@pytest.mark.parametrize("i", range(len(CASES)))
def test_case(labeled, i):
    text, expected = CASES[i]
    got = set(labeled.loc[labeled.review_id == str(i), "category"])
    assert got == expected, text


def test_schema_and_versions(labeled):
    assert list(labeled.columns) == config.REVIEW_LABELS_COLS
    assert labeled.is_match.all()
    assert set(labeled.label_version) == {VERSION}
    assert set(labeled.category) <= set(config.CATEGORIES)
    assert not labeled.duplicated(config.REVIEW_LABELS_KEY).any()


def test_evidence_and_rule_ids(labeled):
    rule_ids = {r for rules in labels.RULES.values() for r, _, _ in rules}
    assert set(labeled.rule_id) <= rule_ids
    assert (labeled.evidence.str.len() > 0).all()
    first = labeled[labeled.review_id == "0"].iloc[0]
    assert (first.rule_id, first.evidence) == ("crash_word", "crashes")


def test_rule_ids_are_unique_and_prefixed():
    prefix = {"crash": "crash_", "performance": "perf_", "match_server": "server_", "monetization": "money_"}
    ids = [r for rules in labels.RULES.values() for r, _, _ in rules]
    assert len(ids) == len(set(ids))
    for cat, rules in labels.RULES.items():
        assert all(r.startswith(prefix[cat]) for r, _, _ in rules)


def test_only_english_negatives_are_labeled():
    spark = get_spark("test")
    rows = [("1", "english", False), ("2", "english", True), ("3", "german", False)]
    clean = spark.createDataFrame(rows, ["review_id", "language", "voted_up"])
    assert [r.review_id for r in labels.english_negatives(clean).collect()] == ["1"]


def test_partitioned_output_feeds_daily_category(tmp_path):
    """B -> A handoff: written labels are read by aggregate and the denominator is unchanged."""
    from pipeline import aggregate
    from tests.test_aggregate import END, read

    spark = get_spark("test")
    clean = (spark.createDataFrame(read("reviews_clean.csv"))
             .withColumn("review_date", F.to_date("review_date")))
    fixture = aggregate.read_labels(spark, labels.config.ROOT / "tests/fixtures/review_labels.csv")
    out = tmp_path / "review_labels"
    for version in ("fixture_v1", "other_v1"):  # dynamic overwrite keeps the earlier version
        (fixture.withColumn("label_version", F.lit(version))
         .write.mode("overwrite").partitionBy("label_version").parquet(str(out)))
    read_back = aggregate.read_labels(spark, out)
    assert {r.label_version for r in read_back.select("label_version").distinct().collect()} == {"fixture_v1", "other_v1"}

    neg = aggregate.daily_negative(clean, END)
    cat = aggregate.daily_category(clean, read_back, neg, "fixture_v1").toPandas()
    expected = read("expected_daily_category.csv")
    assert cat.category_count.sum() == expected.category_count.sum()
    assert cat.groupby(["app_id", "date"]).negative_total.nunique().max() == 1


def test_checks_catch_bad_labels():
    from pipeline import checks

    spark = get_spark("test")
    clean = spark.createDataFrame([("1", "english", False), ("2", "english", True)],
                                  ["review_id", "language", "voted_up"])
    good = spark.createDataFrame([("1", "crash", True, "crash_word", "crash", VERSION)], config.REVIEW_LABELS_COLS)
    assert all(c.passed for c in checks.check_labels(good, clean))
    bad = spark.createDataFrame([("2", "crash", True, "crash_word", "crash", VERSION),
                                 ("1", "bugs", True, "x", "bug", VERSION)], config.REVIEW_LABELS_COLS)
    failed = {c.name for c in checks.check_labels(bad, clean) if not c.passed}
    assert failed == {"category_allowed", "on_english_negatives"}


def test_coverage_counts_unclassified_per_game(labeled):
    spark = get_spark("test")
    neg = spark.createDataFrame([(str(i), "999" if i < 10 else "888") for i in range(len(CASES))],
                                ["review_id", "app_id"])
    cov = labels.coverage(neg, spark.createDataFrame(labeled)).toPandas().set_index("app_id")
    total = cov.loc["all"]
    assert total.negatives == len(CASES)
    assert total.labeled == labeled.review_id.nunique()
    assert total.crash == (labeled.category == "crash").sum()
    assert cov.loc["999"].negatives + cov.loc["888"].negatives == total.negatives
    assert total.unclassified_share == pytest.approx(1 - total.labeled / total.negatives)
