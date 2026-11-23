"""Score the labeling rules against hand labels, and check agreement between annotators.

    python -m pipeline.label_eval score data/samples/label_gold_226.csv                 # train period
    python -m pipeline.label_eval score data/samples/label_gold_226.csv --period later  # once, rules frozen
    python -m pipeline.label_eval agree data/samples/label_sample_226_a.csv data/samples/label_sample_226_b.csv
    python -m pipeline.label_eval export data/samples/label_gold_226.csv

Hand-label files are label_sample CSVs (make label-sample) with the category
columns filled in: 1 when the category applies, 0 when it doesn't. A row counts as
labeled only when all four category cells are filled; rows left empty are skipped,
so each annotator can fill in just their own block. See docs/labeling_guide.md.

score   Runs the current rules (pipeline/labels.py) on the hand-labeled reviews,
        using text_normalized from data/staging/reviews_clean, and reports
        precision, recall, F1 and support per category, plus "any" (labeled with at
        least one category vs unclassified). Writes data/staging/label_eval_<version>_<period>.csv
        and the disagreements, with text, to data/samples/label_errors_<version>_<period>.csv.
agree   Agreement and Cohen's kappa per category on the reviews both files labeled,
        and the disagreements to adjudicate (data/samples/disagreements_<a>_<b>.csv).
export  Writes labels/<name>.csv with IDs, periods and labels only (no review text),
        which is safe to commit.
"""
import argparse
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame

from pipeline import config, labels
from pipeline.clean import OUT_DIR as CLEAN_DIR
from pipeline.config import get_spark

SAMPLES_DIR = config.DATA / "samples"
EXPORT_DIR = config.ROOT / "labels"
TRUE, FALSE = {"1", "1.0", "true", "yes", "y", "x"}, {"0", "0.0", "false", "no", "n"}
EXCERPT_CHARS = 300


def read_hand(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with all four categories filled; categories become booleans."""
    df = df.astype({"review_id": str}).copy()
    cells = df[config.CATEGORIES].fillna("").astype(str).apply(lambda s: s.str.strip().str.lower())
    filled = cells.ne("").all(axis=1)
    bad = cells[filled].apply(lambda s: ~s.isin(TRUE | FALSE)).any(axis=1)
    if bad.any():
        raise ValueError(f"Use 1 or 0 in the category columns; check review_ids {df.loc[bad[bad].index, 'review_id'].tolist()}")
    out = df[filled].copy()
    out[config.CATEGORIES] = cells[filled].isin(TRUE)
    if out.review_id.duplicated().any():
        raise ValueError("A review_id appears more than once")
    return out


def load_hand(path: Path) -> pd.DataFrame:
    # utf-8-sig: Excel and Google Sheets exports may start with a byte-order mark.
    df = read_hand(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"))
    print(f"{path.name}: {len(df)} labeled reviews")
    return df


def predict(texts: DataFrame, label_version: str) -> pd.DataFrame:
    """Rule labels as booleans (one column per category, indexed by review_id), and rule_id/evidence per label."""
    ids = [r.review_id for r in texts.select("review_id").collect()]
    got = labels.label_reviews(texts, label_version).toPandas()
    wide = pd.DataFrame(False, index=pd.Index(ids, name="review_id"), columns=config.CATEGORIES)
    for r in got.itertuples():
        wide.loc[r.review_id, r.category] = True
    evidence = got.set_index(["review_id", "category"])[["rule_id", "evidence"]]
    return wide, evidence


def metrics(gold: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """gold and pred: boolean frames with the same review_id index and category columns."""
    gold, pred = gold[config.CATEGORIES], pred.loc[gold.index, config.CATEGORIES]
    cols = {**{c: (gold[c], pred[c]) for c in config.CATEGORIES}, "any": (gold.any(axis=1), pred.any(axis=1))}
    rows = []
    for name, (g, p) in cols.items():
        tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        rows.append(dict(category=name, reviews=len(g), support=int(g.sum()), predicted=int(p.sum()),
                         tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1))
    return pd.DataFrame(rows)


def errors(gold: pd.DataFrame, pred: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in config.CATEGORIES:
        for rid in gold.index[gold[c] != pred.loc[gold.index, c]]:
            ev = evidence.loc[(rid, c)] if (rid, c) in evidence.index else None
            rows.append(dict(review_id=rid, category=c, error="fp" if pred.loc[rid, c] else "fn",
                             rule_id=None if ev is None else ev.rule_id,
                             evidence=None if ev is None else ev.evidence))
    return pd.DataFrame(rows, columns=["review_id", "category", "error", "rule_id", "evidence"])


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    both = a.set_index("review_id")[config.CATEGORIES].join(
        b.set_index("review_id")[config.CATEGORIES], how="inner", lsuffix="_a", rsuffix="_b")
    rows = []
    for c in config.CATEGORIES:
        x, y = both[f"{c}_a"], both[f"{c}_b"]
        n = len(both)
        po = (x == y).mean() if n else None
        pa, pb = (x.mean(), y.mean()) if n else (None, None)
        pe = pa * pb + (1 - pa) * (1 - pb) if n else None
        kappa = (po - pe) / (1 - pe) if n and pe < 1 else None
        rows.append(dict(category=c, reviews=n, a_positive=int(x.sum()), b_positive=int(y.sum()),
                         disagreements=int((x != y).sum()), agreement=po, kappa=kappa))
    return pd.DataFrame(rows)


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))


def score(path: Path, label_version: str, period: str) -> None:
    gold = load_hand(path)
    if "period" in gold:
        gold = gold[gold.period == period]
    if gold.empty:
        raise SystemExit(f"No labeled reviews in period {period!r}.")
    spark = get_spark("steampulse-label-eval")
    texts = (spark.read.parquet(str(CLEAN_DIR)).select("review_id", "text_normalized", "text_raw")
             .join(spark.createDataFrame(gold[["review_id"]]), "review_id").cache())
    missing = set(gold.review_id) - {r.review_id for r in texts.select("review_id").collect()}
    if missing:
        raise SystemExit(f"{len(missing)} review_ids are not in reviews_clean, e.g. {sorted(missing)[:5]}")

    pred, evidence = predict(texts, label_version)
    gold = gold.set_index("review_id")
    result = metrics(gold, pred)
    out = config.STAGING_DIR / f"label_eval_{label_version}_{period}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.assign(label_version=label_version, period=period, gold_file=path.name).to_csv(out, index=False)

    err = errors(gold, pred, evidence)
    text = texts.select("review_id", "text_raw").toPandas().set_index("review_id").text_raw.str[:EXCERPT_CHARS]
    err["text"] = err.review_id.map(text)
    if "notes" in gold:
        err["notes"] = err.review_id.map(gold.notes)
    err_path = SAMPLES_DIR / f"label_errors_{label_version}_{period}.csv"
    err_path.parent.mkdir(parents=True, exist_ok=True)
    err.to_csv(err_path, index=False)

    print(f"\n{label_version} on {len(gold)} hand-labeled {period} reviews:")
    _print(result)
    print(f"\nWrote {out.relative_to(config.ROOT)} and {len(err)} disagreements to {err_path.relative_to(config.ROOT)}")
    if period == "later":
        print("This is the held-out check: don't change the rules because of these numbers. "
              "If you do, bump LABEL_VERSION and report it.")


def agree(path_a: Path, path_b: Path) -> None:
    a, b = load_hand(path_a), load_hand(path_b)
    result = agreement(a, b)
    _print(result)
    both = a.merge(b, on="review_id", suffixes=("_a", "_b"))
    rows = []
    for c in config.CATEGORIES:
        for r in both[both[f"{c}_a"] != both[f"{c}_b"]].itertuples():
            rows.append(dict(review_id=r.review_id, category=c, a=int(getattr(r, f"{c}_a")),
                             b=int(getattr(r, f"{c}_b")), text=getattr(r, "text_raw_a", None),
                             notes_a=getattr(r, "notes_a", None), notes_b=getattr(r, "notes_b", None), decision=""))
    out = SAMPLES_DIR / f"disagreements_{path_a.stem}_{path_b.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} disagreements to adjudicate -> {out.relative_to(config.ROOT)}")


def export(path: Path) -> None:
    gold = load_hand(path)
    keep = [c for c in ("review_id", "app_id", "period") if c in gold] + config.CATEGORIES
    out = gold[keep].copy()
    out[config.CATEGORIES] = out[config.CATEGORIES].astype(int)
    EXPORT_DIR.mkdir(exist_ok=True)
    dest = EXPORT_DIR / path.name
    out.to_csv(dest, index=False)
    print(f"Wrote {len(out)} labels without text -> {dest.relative_to(config.ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="rules vs hand labels")
    s.add_argument("gold", type=Path)
    s.add_argument("--label-version", default=config.LABEL_VERSION)
    s.add_argument("--period", choices=["train", "later"], default="train")
    a = sub.add_parser("agree", help="agreement between two annotators")
    a.add_argument("file_a", type=Path)
    a.add_argument("file_b", type=Path)
    e = sub.add_parser("export", help="IDs and labels only, for committing")
    e.add_argument("gold", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        score(args.gold, args.label_version, args.period)
    elif args.command == "agree":
        agree(args.file_a, args.file_b)
    else:
        export(args.gold)
