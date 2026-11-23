"""Data quality checks run on staging output before publish."""
from pipeline.config import get_spark


def main() -> None:
    spark = get_spark()
    raise NotImplementedError("quality checks not implemented yet")


if __name__ == "__main__":
    main()
