"""Chronological splits shared by training, evaluation and the alert analysis.

Rows are split by forecast origin t = as_of_date. The target is the share over
t+1..t+7, so a row belongs to the split that contains its whole target week:

  train       t + 7 <= TRAIN_END
  validation  t > TRAIN_END and t + 7 <= VAL_END
  test        t > VAL_END
  purged      TRAIN_END-6..TRAIN_END and VAL_END-6..VAL_END. Their target weeks
              cross into the next split, so they are never scored.

With the config dates: train origins run to 2023-05-24, validation 2023-06-01 to
2023-08-08, and test from 2023-08-16. Features only look back (<= t), so a
validation row may use training-period days as inputs; it never sees a target
day of a later split.
"""
from pyspark.sql import Column
from pyspark.sql import functions as F

from pipeline import config

SPLITS = ["train", "validation", "test", "purged"]


def split_col(train_end: str = config.TRAIN_END, val_end: str = config.VAL_END) -> Column:
    t, target_end = F.col("as_of_date"), F.date_add("as_of_date", 7)
    tr, va = F.lit(train_end).cast("date"), F.lit(val_end).cast("date")
    return (F.when(target_end <= tr, "train")
            .when((t > tr) & (target_end <= va), "validation")
            .when(t > va, "test")
            .otherwise("purged"))


def usable() -> Column:
    """Rows that can be fitted or scored: eligible at t and with an observed target week."""
    return (F.col("eligibility") == "eligible") & F.col(config.LABEL_COL).isNotNull()


def row_status() -> Column:
    """Why a row is or isn't scored, so exclusions can be reported separately."""
    return (F.when(F.col("eligibility") != "eligible", F.col("eligibility"))
            .when(F.col(config.LABEL_COL).isNull(), "no_target")
            .otherwise("scored"))
