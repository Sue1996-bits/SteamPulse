"""Label evaluation: reading hand labels, metrics, agreement, and rules scored on known examples."""
import pandas as pd
import pytest
from pyspark.sql import functions as F

from pipeline import config, label_eval
from pipeline.clean import normalize_text
from pipeline.config import get_spark
from tests.test_labels import CASES


def hand(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["review_id", "period", *config.CATEGORIES, "notes"])


def test_read_hand_skips_unfilled_rows():
    df = hand([("1", "train", "1", "0", "0", "0", ""),
               ("2", "train", "", "", "", "", ""),        # not labeled yet
               ("3", "train", "0", "0", "0", "0", "?"),     # labeled, no category
               ("4", "later", "1", "", "0", "0", "")])      # incomplete -> skipped
    got = label_eval.read_hand(df)
    assert list(got.review_id) == ["1", "3"]
    assert got.crash.tolist() == [True, False] and got.crash.dtype == bool


def test_read_hand_rejects_other_values():
    with pytest.raises(ValueError, match="1 or 0"):
        label_eval.read_hand(hand([("1", "train", "maybe", "0", "0", "0", "")]))
    with pytest.raises(ValueError, match="more than once"):
        label_eval.read_hand(hand([("1", "train", "1", "0", "0", "0", ""), ("1", "train", "0", "0", "0", "0", "")]))


def test_metrics_by_hand():
    idx = pd.Index(["a", "b", "c", "d"], name="review_id")
    gold = pd.DataFrame(False, index=idx, columns=config.CATEGORIES)
    pred = gold.copy()
    gold.loc[["a", "b", "c"], "crash"] = True      # 3 true crashes
    pred.loc[["a", "b", "d"], "crash"] = True      # 2 found, 1 false alarm, 1 missed
    m = label_eval.metrics(gold, pred).set_index("category")
    crash = m.loc["crash"]
    assert (crash.support, crash.tp, crash.fp, crash.fn) == (3, 2, 1, 1)
    assert crash.precision == pytest.approx(2 / 3) and crash.recall == pytest.approx(2 / 3)
    assert crash.f1 == pytest.approx(2 / 3)
    assert m.loc["performance"].support == 0 and pd.isna(m.loc["performance"].precision)
    assert m.loc["any"].support == 3 and m.loc["any"].predicted == 3


def test_agreement_and_kappa():
    a = label_eval.read_hand(hand([(str(i), "train", c, "0", "0", "0", "") for i, c in enumerate("1100")]))
    b = label_eval.read_hand(hand([(str(i), "train", c, "0", "0", "0", "") for i, c in enumerate("1000")]))
    crash = label_eval.agreement(a, b).set_index("category").loc["crash"]
    # po = 3/4, pa = 1/2, pb = 1/4, pe = 1/8 + 3/8 = 1/2, kappa = (0.75 - 0.5) / 0.5
    assert crash.disagreements == 1 and crash.agreement == pytest.approx(0.75)
    assert crash.kappa == pytest.approx(0.5)


def test_rules_score_perfectly_on_their_own_examples():
    spark = get_spark("test")
    texts = (spark.createDataFrame([(str(i), t) for i, (t, _) in enumerate(CASES)], ["review_id", "text_raw"])
             .withColumn("text_normalized", normalize_text(F.col("text_raw"))))
    gold = pd.DataFrame(False, index=pd.Index([str(i) for i in range(len(CASES))], name="review_id"),
                        columns=config.CATEGORIES)
    for i, (_, cats) in enumerate(CASES):
        for c in cats:
            gold.loc[str(i), c] = True
    pred, evidence = label_eval.predict(texts, "test_v1")
    m = label_eval.metrics(gold, pred)
    assert (m.fp == 0).all() and (m.fn == 0).all()
    assert label_eval.errors(gold, pred, evidence).empty
    gold.loc["21", "crash"] = True  # "not fun" -> a miss the error list must show
    err = label_eval.errors(gold, pred, evidence)
    assert err[["review_id", "category", "error"]].values.tolist() == [["21", "crash", "fn"]]


def test_load_hand_accepts_spreadsheet_exports(tmp_path):
    path = tmp_path / "from_sheets.csv"
    df = hand([("123456789", "train", "1", "0", "0", "1", "has, comma\nand newline")])
    path.write_bytes(df.to_csv(index=False).encode("utf-8-sig"))  # with a byte-order mark
    got = label_eval.load_hand(path)
    assert got.review_id.tolist() == ["123456789"] and got.monetization.tolist() == [True]
