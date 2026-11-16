"""Build forecast_features with range-based windows and a separate target column."""
from pipeline.config import get_spark


def main() -> None:
    spark = get_spark()
    raise NotImplementedError("feature building not implemented yet")


if __name__ == "__main__":
    main()
