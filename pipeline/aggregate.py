"""Build daily_negative and daily_category on the covered-date grid."""
from pipeline.config import get_spark


def main() -> None:
    spark = get_spark()
    raise NotImplementedError("daily aggregation not implemented yet")


if __name__ == "__main__":
    main()
