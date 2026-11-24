"""Split the hand-labeling sample, check agreement, build the gold file and score the rules.

    python -m pipeline.label_eval split data/samples/label_sample_226.csv --people yifan isabella yunseo
    python -m pipeline.label_eval agree data/samples/label_sample_226_yifan.csv data/samples/label_sample_226_isabella.csv
    python -m pipeline.label_eval gold data/samples/label_sample_226.csv data/samples/label_sample_226_*.csv
    python -m pipeline.label_eval score data/samples/label_gold_226.csv                 # train period
    python -m pipeline.label_eval score data/samples/label_gold_226.csv --period later  # once, rules frozen
    python -m pipeline.label_eval export data/samples/label_gold_226.csv

Hand-label files are label_sample CSVs (make label-sample) with the four category
columns filled in: 1 when the category applies, 0 when it doesn't. A row counts as
labeled only when all four cells hold 1 or 0, so a finished row can't be confused
with one that was only started. Rows with some cells filled are reported as
incomplete and rows with none as blank, with their review_ids; neither is ever
counted as "no category", and gold refuses to build while a primary row is missing.
See docs/labeling_guide.md.

split   One file per annotator, never overwriting an existing file: a block of the
        train and later reviews (role=primary; blocks differ by at most one review,
        and every sample review is primary exactly once) plus the first OVERLAP train
        reviews of the next person's block (role=second), for agreement. The period
        column is left out so annotators don't see it.
agree   Agreement and Cohen's kappa per category on the reviews both files labeled,
        and the disagreements to adjudicate (data/samples/disagreements_<a>_<b>.csv).
gold    Takes every annotator's primary rows, restores period from the original
        sample and writes data/samples/label_gold_<seed>.csv. It stops while a primary
        row is unfinished; --partial builds a preview from what's done instead, always
        into its own file (label_gold_<seed>_partial.csv by default), never the official
        gold file. An existing output is only replaced with --force, and the sample and
        input files never are. Adjudicated decisions go into the primary annotator's
        file first.
score   Needs a valid period (train or later) on every row; gold provides it. Runs the
        current rules (pipeline/labels.py) on the hand-labeled reviews, using
        text_normalized from data/staging/reviews_clean, and reports precision,
        recall, F1 and support per category, plus "any" (labeled with at least one
        category vs unclassified). Writes data/staging/label_eval_<version>_<period>.csv
        and the disagreements, with text, to data/samples/label_errors_<version>_<period>.csv.
export  Writes labels/<name>.csv with IDs, periods and labels only (no review text),
        which is safe to commit.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame

from pipeline import config, labels
from pipeline.clean import OUT_DIR as CLEAN_DIR
from pipeline.config import get_spark

SAMPLES_DIR = config.DATA / "samples"
EXPORT_DIR = config.ROOT / "labels"
TRUE, FALSE = {"1", "1.0", "true", "yes", "y", "x"}, {"0", "0.0", "false", "no", "n"}
PERIODS = ("train", "later")
EXCERPT_CHARS = 300
OVERLAP = 30  # train reviews per annotator that a second person also labels


# ---- reading files ----------------------------------------------------------------

def read_hand(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with all four categories filled; categories become booleans.

    The review_ids of rows with only some cells filled are in out.attrs["incomplete"],
    and of rows with none filled in out.attrs["blank"].
    """
    df = df.astype({"review_id": str}).copy()
    cells = df[config.CATEGORIES].fillna("").astype(str).apply(lambda s: s.str.strip().str.lower())
    filled = cells.ne("")
    bad = (filled & ~cells.isin(TRUE | FALSE)).any(axis=1)
    if bad.any():
        raise ValueError(f"Use 1 or 0 in the category columns; check review_ids {df.loc[bad, 'review_id'].tolist()}")
    complete, started = filled.all(axis=1), filled.any(axis=1)
    out = df[complete].copy()
    out[config.CATEGORIES] = cells[complete].isin(TRUE)
    if out.review_id.duplicated().any():
        raise ValueError("A review_id appears more than once")
    out.attrs["incomplete"] = df.loc[started & ~complete, "review_id"].tolist()
    out.attrs["blank"] = df.loc[~started, "review_id"].tolist()
    return out


def load_hand(path: Path) -> pd.DataFrame:
    # utf-8-sig: Excel and Google Sheets exports may start with a byte-order mark.
    df = read_hand(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"))
    notes = [f"{len(ids)} {what}, e.g. {ids[:5]}" for what, ids in
             (("incomplete (some cells blank, not counted)", df.attrs["incomplete"]),
              ("blank (not labeled yet)", df.attrs["blank"])) if ids]
    print(f"{path.name}: {len(df)} labeled reviews" + "".join(f"; {n}" for n in notes))
    return df


def check_periods(df: pd.DataFrame, where: str) -> None:
    bad = ~df.period.isin(PERIODS)
    if bad.any():
        raise ValueError(f"{where}: period must be one of {PERIODS}; check review_ids "
                         f"{df.loc[bad, 'review_id'].astype(str).tolist()[:10]}")


def read_sample(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if sample.review_id.duplicated().any():
        raise ValueError(f"{path.name}: a review_id appears more than once")
    check_periods(sample, path.name)
    return sample


# ---- split and gold -----------------------------------------------------------------

def write_csv(df: pd.DataFrame, dest: Path) -> None:
    """Write next to dest, then swap in, so a failed write never leaves a half-written file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)


def _blocks(n: int, k: int) -> list[tuple[int, int]]:
    """k contiguous slices covering range(n); sizes differ by at most one (larger ones first)."""
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    starts = [sum(sizes[:i]) for i in range(k)]
    return [(s, s + size) for s, size in zip(starts, sizes)]


def split_sample(sample: pd.DataFrame, people: list[str], overlap: int = OVERLAP) -> dict[str, pd.DataFrame]:
    """One labeling file per person; see the module docstring."""
    if len(people) < 2 or len(set(people)) != len(people):
        raise ValueError(f"Need at least two different annotator names, got {people}")
    train = sample[sample.period == "train"].reset_index(drop=True)
    later = sample[sample.period == "later"].reset_index(drop=True)
    tb, lb = _blocks(len(train), len(people)), _blocks(len(later), len(people))
    smallest = min(e - s for s, e in tb)
    if overlap > smallest:
        raise ValueError(f"overlap {overlap} is larger than the smallest block of {smallest} train reviews")
    out = {}
    for i, who in enumerate(people):
        (ts, te), (ls, le) = tb[i], lb[i]
        mine = pd.concat([train.iloc[ts:te], later.iloc[ls:le]]).assign(role="primary")
        ns = tb[(i + 1) % len(people)][0]
        second = train.iloc[ns:ns + overlap].assign(role="second")
        out[who] = pd.concat([mine, second]).drop(columns=["period"]).sample(frac=1, random_state=i)
    primary = pd.concat([f[f.role == "primary"] for f in out.values()]).review_id
    if len(primary) != len(sample) or set(primary) != set(sample.review_id):
        raise AssertionError("split must assign every sample review as primary exactly once")
    return out


def build_gold(sample: pd.DataFrame, files: list[pd.DataFrame], allow_partial: bool = False) -> pd.DataFrame:
    """Primary rows from every annotator's file (read with read_hand), with period restored from the sample."""
    check_periods(sample, "sample")
    parts = [f[f.role == "primary"] if "role" in f else f for f in files]
    gold = pd.concat(parts, ignore_index=True)
    if gold.review_id.duplicated().any():
        dup = gold.review_id[gold.review_id.duplicated()].tolist()
        raise ValueError(f"These reviews are primary in more than one file: {dup[:5]}")
    period = sample.astype({"review_id": str}).set_index("review_id").period
    gold["period"] = gold.review_id.map(period)
    if gold.period.isna().any():
        raise ValueError(f"Reviews not in the sample: {gold.review_id[gold.period.isna()].tolist()[:5]}")
    missing = sorted(set(period.index) - set(gold.review_id))
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} sample reviews have no complete primary label yet (blank or incomplete "
                         f"rows), e.g. {missing[:5]}. Finish them, or pass --partial to build from what's done.")
    gold.attrs["missing"] = missing
    return gold


# ---- scoring ------------------------------------------------------------------------

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
    """gold and pred: boolean frames with the same review_id index and category columns.

    precision and recall are null when their denominator is 0. F1 = 2TP / (2TP + FP + FN),
    so it is 0 when there are errors but no hits, and null only when the category is absent
    from both gold and predictions.
    """
    gold, pred = gold[config.CATEGORIES], pred.loc[gold.index, config.CATEGORIES]
    cols = {**{c: (gold[c], pred[c]) for c in config.CATEGORIES}, "any": (gold.any(axis=1), pred.any(axis=1))}
    rows = []
    for name, (g, p) in cols.items():
        tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
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


# ---- commands -----------------------------------------------------------------------

def score(path: Path, label_version: str, period: str) -> None:
    gold = load_hand(path)
    if "period" not in gold:
        raise SystemExit(f"{path.name} has no period column, so train and later can't be kept apart. "
                         "Build the file with `python -m pipeline.label_eval gold ...`, which restores it.")
    try:
        check_periods(gold, path.name)
    except ValueError as e:
        raise SystemExit(str(e))
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
    print(f"{len(rows)} disagreements to adjudicate -> {out}")


def split(sample_path: Path, people: list[str], out_dir: Path | None = None, force: bool = False) -> list[Path]:
    sample = read_sample(sample_path)
    files = split_sample(sample, people)
    out_dir = out_dir or sample_path.parent
    dests = {who: out_dir / f"{sample_path.stem}_{who}.csv" for who in files}
    clash = [d for d in dests.values() if d.resolve() == sample_path.resolve()]
    existing = [d for d in dests.values() if d.exists()]
    if clash:
        raise SystemExit(f"{clash[0]} would replace the sample itself.")
    if existing and not force:
        raise SystemExit("These files already exist and may hold labels: "
                         + ", ".join(str(d) for d in existing)
                         + ". Use --out-dir for a new split, or --force if you really mean to replace them.")
    for who, df in files.items():
        write_csv(df, dests[who])
        print(f"{dests[who].name}: {len(df)} reviews ({(df.role == 'primary').sum()} primary, "
              f"{(df.role == 'second').sum()} second)")
    return list(dests.values())


def gold_path(sample_path: Path, partial: bool = False) -> Path:
    """label_sample_226.csv -> label_gold_226.csv (or label_gold_226_partial.csv); sample.csv -> sample_gold.csv."""
    stem = sample_path.stem
    base = stem.replace("label_sample_", "label_gold_", 1) if stem.startswith("label_sample_") else f"{stem}_gold"
    return sample_path.with_name(f"{base}{'_partial' if partial else ''}.csv")


def gold_file(sample_path: Path, paths: list[Path], out: Path | None = None, allow_partial: bool = False,
              force: bool = False) -> Path:
    """Every check runs before anything is written, so a failed run leaves all files as they were."""
    inputs = [p for p in paths if p.resolve() != sample_path.resolve()]
    dest = out or gold_path(sample_path, partial=allow_partial)
    if dest.resolve() in {p.resolve() for p in [sample_path, *inputs]}:
        raise SystemExit(f"{dest} is an input file; choose another output with --out.")
    if allow_partial and dest.resolve() == gold_path(sample_path).resolve():
        raise SystemExit(f"{dest} is the official gold file; --partial only writes a separate preview.")
    if dest.exists() and not force:
        raise SystemExit(f"{dest} already exists. Use --force to replace it, or --out for a new file.")
    sample = read_sample(sample_path)
    try:
        gold = build_gold(sample, [load_hand(p) for p in inputs], allow_partial)
    except ValueError as e:
        raise SystemExit(str(e))
    written = gold.copy()
    written[config.CATEGORIES] = written[config.CATEGORIES].astype(int)
    write_csv(written, dest)
    missing = gold.attrs["missing"]
    print(f"Wrote {len(gold)} labeled reviews to {dest} ({gold.period.value_counts().to_dict()})"
          + (f"; {len(missing)} sample reviews aren't labeled yet (--partial)" if missing else ""))
    return dest


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
    sp = sub.add_parser("split", help="one labeling file per annotator (never overwrites)")
    sp.add_argument("sample", type=Path)
    sp.add_argument("--people", nargs="+", required=True)
    sp.add_argument("--out-dir", type=Path, help="folder for the files (default: next to the sample)")
    sp.add_argument("--force", action="store_true", help="replace existing files, which may hold labels")
    g = sub.add_parser("gold", help="primary rows from every file, with period restored")
    g.add_argument("sample", type=Path)
    g.add_argument("files", type=Path, nargs="+")
    g.add_argument("--out", type=Path, help="output file (default: label_gold_<seed>.csv, or "
                   "label_gold_<seed>_partial.csv with --partial, next to the sample)")
    g.add_argument("--partial", action="store_true",
                   help="preview from the finished rows, into its own file; never the official gold file")
    g.add_argument("--force", action="store_true",
                   help="replace an existing output (never the sample, an input, or, with --partial, the official file)")
    s = sub.add_parser("score", help="rules vs hand labels")
    s.add_argument("gold", type=Path)
    s.add_argument("--label-version", default=config.LABEL_VERSION)
    s.add_argument("--period", choices=PERIODS, default="train")
    a = sub.add_parser("agree", help="agreement between two annotators")
    a.add_argument("file_a", type=Path)
    a.add_argument("file_b", type=Path)
    e = sub.add_parser("export", help="IDs and labels only, for committing")
    e.add_argument("gold", type=Path)
    args = parser.parse_args()
    if args.command == "split":
        split(args.sample, args.people, args.out_dir, args.force)
    elif args.command == "gold":
        gold_file(args.sample, args.files, args.out, args.partial, args.force)
    elif args.command == "score":
        score(args.gold, args.label_version, args.period)
    elif args.command == "agree":
        agree(args.file_a, args.file_b)
    else:
        export(args.gold)
