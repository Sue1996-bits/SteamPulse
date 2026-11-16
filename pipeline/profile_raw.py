"""Per-game profile of the raw snapshot (rows, unique IDs, English negatives, edit rate, date range, coverage)."""
from pipeline.config import get_spark


def main() -> None:
    spark = get_spark()
    raise NotImplementedError("profiling not implemented yet")


if __name__ == "__main__":
    main()
