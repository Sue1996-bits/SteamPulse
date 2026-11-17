"""Persistence and rolling-28 baselines, scored on the same test rows as the trained model."""
from pipeline.config import get_spark


def main() -> None:
    spark = get_spark()
    raise NotImplementedError("baselines not implemented yet")


if __name__ == "__main__":
    main()
