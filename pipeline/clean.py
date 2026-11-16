"""Build reviews_clean (normalize IDs, map negatives, quarantine bad rows, dedupe, normalize text)."""
from pipeline.config import get_spark


def main() -> None:
    spark = get_spark()
    raise NotImplementedError("cleaning not implemented yet")


if __name__ == "__main__":
    main()
