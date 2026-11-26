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


def test_read_hand_counts_only_complete_rows_and_reports_the_rest():
    df = hand([("1", "train", "1", "0", "0", "0", ""),     # complete
               ("2", "train", "1", "", "", "", ""),        # started, not finished: not counted
               ("3", "train", "", "", "", "", ""),         # not started
               ("4", "later", "0", "0", "0", "0", "?")])   # complete, no category
    got = label_eval.read_hand(df)
    assert list(got.review_id) == ["1", "4"]
    assert got.attrs["incomplete"] == ["2"] and got.attrs["blank"] == ["3"]
    assert got.crash.tolist() == [True, False] and got.crash.dtype == bool


def test_read_hand_rejects_other_values():
    with pytest.raises(ValueError, match="1 or 0"):
        label_eval.read_hand(hand([("1", "train", "maybe", "0", "0", "0", "")]))
    with pytest.raises(ValueError, match="1 or 0"):         # checked in unfinished rows too
        label_eval.read_hand(hand([("1", "train", "maybe", "", "", "", "")]))
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
    assert pd.isna(m.loc["performance"].f1)            # absent from gold and predictions
    assert m.loc["any"].support == 3 and m.loc["any"].predicted == 3


def test_f1_is_zero_when_there_are_errors_but_no_hits():
    idx = pd.Index(["a", "b"], name="review_id")
    gold = pd.DataFrame(False, index=idx, columns=config.CATEGORIES)
    pred = gold.copy()
    gold.loc["a", "crash"] = True                      # TP=0, FP=1, FN=1
    pred.loc["b", "crash"] = True
    crash = label_eval.metrics(gold, pred).set_index("category").loc["crash"]
    assert (crash.tp, crash.fp, crash.fn) == (0, 1, 1)
    assert crash.precision == 0 and crash.recall == 0 and crash.f1 == 0


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


def sample_frame(n_train: int = 300, n_later: int = 150) -> pd.DataFrame:
    rows = [(str(1000 + i), "train" if i < n_train else "later", f"text {i}") for i in range(n_train + n_later)]
    df = pd.DataFrame(rows, columns=["review_id", "period", "text_raw"])
    for c in config.CATEGORIES:
        df[c] = ""
    return df


def test_split_and_gold_round_trip():
    sample = sample_frame()
    people = ["a", "b", "c"]
    files = label_eval.split_sample(sample, people)
    for i, who in enumerate(people):
        f = files[who]
        assert len(f) == 180 and "period" not in f
        assert f.role.value_counts().to_dict() == {"primary": 150, "second": 30}
        nxt = files[people[(i + 1) % 3]]
        assert set(f[f.role == "second"].review_id) <= set(nxt[nxt.role == "primary"].review_id)
    primary = pd.concat([f[f.role == "primary"] for f in files.values()])
    assert primary.review_id.is_unique and set(primary.review_id) == set(sample.review_id)
    for f in files.values():          # everyone marks crash on every row they own
        f["crash"] = "1"
        for c in config.CATEGORIES[1:]:
            f[c] = "0"
    gold = label_eval.build_gold(sample, [label_eval.read_hand(f) for f in files.values()])
    assert len(gold) == 450 and gold.crash.all()
    assert gold.set_index("review_id").period.equals(sample.set_index("review_id").period.loc[gold.review_id])


def test_score_refuses_files_without_period(tmp_path):
    path = tmp_path / "label_sample_226_a.csv"
    hand([("1", "train", "1", "0", "0", "0", "")]).drop(columns="period").to_csv(path, index=False)
    with pytest.raises(SystemExit, match="period"):
        label_eval.score(path, "rules_v1", "train")


def test_split_assigns_every_review_once_even_with_a_remainder():
    sample = sample_frame(301, 151)
    files = label_eval.split_sample(sample, ["a", "b", "c"])
    primary = pd.concat([f[f.role == "primary"] for f in files.values()]).review_id
    assert len(primary) == 452 and primary.is_unique and set(primary) == set(sample.review_id)
    sizes = [(f.role == "primary").sum() for f in files.values()]
    assert max(sizes) - min(sizes) <= 2                 # train and later blocks each differ by at most one


@pytest.mark.parametrize("people", [["a", "a", "b"], ["a"]])
def test_split_rejects_duplicate_or_single_names(people):
    with pytest.raises(ValueError, match="different annotator names"):
        label_eval.split_sample(sample_frame(), people)


def test_split_never_overwrites_without_force(tmp_path):
    path = tmp_path / "label_sample_226.csv"
    sample_frame().to_csv(path, index=False)
    first = label_eval.split(path, ["a", "b", "c"])
    first[0].write_text("labels someone typed in")
    with pytest.raises(SystemExit, match="already exist"):
        label_eval.split(path, ["a", "b", "c"])
    assert first[0].read_text() == "labels someone typed in"
    label_eval.split(path, ["a", "b", "c"], out_dir=tmp_path / "round2")      # a new path is fine
    label_eval.split(path, ["a", "b", "c"], force=True)                       # explicit replace
    assert first[0].read_text() != "labels someone typed in"


def complete_files(tmp_path, sample_path, people=("a", "b", "c")):
    files = label_eval.split(sample_path, list(people), out_dir=tmp_path / "files")
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        df[config.CATEGORIES] = "0"
        df.to_csv(f, index=False)
    return files


def test_gold_never_writes_over_its_inputs(tmp_path):
    sample_path = tmp_path / "sample.csv"            # not named label_sample_*
    sample_frame().to_csv(sample_path, index=False)
    files = complete_files(tmp_path, sample_path)
    assert label_eval.gold_path(sample_path) == tmp_path / "sample_gold.csv"
    dest = label_eval.gold_file(sample_path, files)
    assert dest == tmp_path / "sample_gold.csv" and len(pd.read_csv(dest)) == 450
    for bad in (sample_path, files[0]):
        with pytest.raises(SystemExit, match="input file"):
            label_eval.gold_file(sample_path, files, out=bad)


def test_gold_refuses_unfinished_primary_rows_unless_partial(tmp_path):
    sample_path = tmp_path / "label_sample_226.csv"
    sample_frame().to_csv(sample_path, index=False)
    files = complete_files(tmp_path, sample_path)
    df = pd.read_csv(files[0], dtype=str, keep_default_na=False)
    row = df.index[df.role == "primary"][0]
    df.loc[row, config.CATEGORIES[1:]] = ""            # one primary row left unfinished
    df.to_csv(files[0], index=False)
    with pytest.raises(SystemExit, match="1 sample reviews have no complete primary label"):
        label_eval.gold_file(sample_path, files)
    dest = label_eval.gold_file(sample_path, files, allow_partial=True)
    assert len(pd.read_csv(dest)) == 449


def test_invalid_period_values_are_rejected(tmp_path):
    sample = sample_frame()
    sample.loc[0, "period"] = "latre"
    with pytest.raises(ValueError, match="period must be one of"):
        label_eval.build_gold(sample, [])
    path = tmp_path / "label_gold_x.csv"
    hand([("7", "latre", "1", "0", "0", "0", ""), ("8", "train", "0", "0", "0", "0", "")]).to_csv(path, index=False)
    with pytest.raises(SystemExit, match=r"period must be one of.*'7'"):
        label_eval.score(path, "rules_v1", "train")


# ---- gold output protection ---------------------------------------------------------

def gold_env(tmp_path, sample_name="label_sample_226.csv"):
    sample_path = tmp_path / sample_name
    sample_frame().to_csv(sample_path, index=False)
    return sample_path, complete_files(tmp_path, sample_path)


def leave_one_unfinished(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.loc[df.index[df.role == "primary"][0], config.CATEGORIES[1:]] = ""
    df.to_csv(path, index=False)


@pytest.mark.parametrize("sample_name, official, partial", [
    ("label_sample_226.csv", "label_gold_226.csv", "label_gold_226_partial.csv"),
    ("sample.csv", "sample_gold.csv", "sample_gold_partial.csv"),
])
def test_partial_never_touches_the_official_file(tmp_path, sample_name, official, partial):
    sample_path, files = gold_env(tmp_path, sample_name)
    assert label_eval.gold_path(sample_path) == tmp_path / official
    assert label_eval.gold_path(sample_path, partial=True) == tmp_path / partial
    first = label_eval.gold_file(sample_path, files)
    assert first == tmp_path / official and len(pd.read_csv(first)) == 450
    kept = first.read_bytes()
    leave_one_unfinished(files[0])
    with pytest.raises(SystemExit, match="no complete primary label"):     # official mode still refuses
        label_eval.gold_file(sample_path, files, force=True)
    preview = label_eval.gold_file(sample_path, files, allow_partial=True)
    assert preview == tmp_path / partial and len(pd.read_csv(preview)) == 449
    assert first.read_bytes() == kept
    assert not list(tmp_path.glob(".*.tmp"))


def test_partial_out_cannot_target_the_official_file_the_sample_or_inputs(tmp_path):
    sample_path, files = gold_env(tmp_path)
    official = label_eval.gold_file(sample_path, files)
    leave_one_unfinished(files[0])
    targets = [official, sample_path, files[0], files[1]]
    before = {t: t.read_bytes() for t in targets}
    for target in targets:
        for force in (False, True):
            with pytest.raises(SystemExit, match="official gold file|input file"):
                label_eval.gold_file(sample_path, files, out=target, allow_partial=True, force=force)
    assert {t: t.read_bytes() for t in targets} == before
    other = label_eval.gold_file(sample_path, files, out=tmp_path / "preview.csv", allow_partial=True)
    assert len(pd.read_csv(other)) == 449


def test_existing_outputs_are_replaced_only_with_force(tmp_path):
    sample_path, files = gold_env(tmp_path)
    official = label_eval.gold_file(sample_path, files)
    kept = official.read_bytes()
    with pytest.raises(SystemExit, match="already exists"):                # rerun after adjudication
        label_eval.gold_file(sample_path, files)
    assert official.read_bytes() == kept
    assert label_eval.gold_file(sample_path, files, force=True) == official

    leave_one_unfinished(files[0])
    preview = label_eval.gold_file(sample_path, files, allow_partial=True)
    preview.write_text("an older preview")
    with pytest.raises(SystemExit, match="already exists"):
        label_eval.gold_file(sample_path, files, allow_partial=True)
    assert preview.read_text() == "an older preview"
    label_eval.gold_file(sample_path, files, allow_partial=True, force=True)
    assert len(pd.read_csv(preview)) == 449 and official.read_bytes() == kept
