"""Download the raw snapshot from Hugging Face into data/raw/, pinned to SNAPSHOT_ID.

    python -m pipeline.download --files 2   # first 2 part files (benchmark)
    python -m pipeline.download             # all 663 part files (~23.5 GB)

Files already present are skipped, so an interrupted download can be rerun.
"""
import argparse
import time

from huggingface_hub import HfApi, snapshot_download

from pipeline import config


def part_files(limit: int | None = None) -> list[str]:
    """Part files in numeric order (part.0, part.1, ...), not string order."""
    files = HfApi().list_repo_files(config.HF_DATASET, repo_type="dataset",
                                    revision=config.SNAPSHOT_ID)
    parts = sorted((f for f in files if f.startswith("part.") and f.endswith(".parquet")),
                   key=lambda f: int(f.split(".")[1]))
    return parts[:limit] if limit else parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", type=int, default=None,
                        help="download only the first N part files")
    args = parser.parse_args()

    files = part_files(args.files)
    start = time.perf_counter()
    snapshot_download(
        repo_id=config.HF_DATASET,
        repo_type="dataset",
        revision=config.SNAPSHOT_ID,
        local_dir=config.RAW_DIR,
        allow_patterns=files + ["README.md"],
    )
    elapsed = time.perf_counter() - start
    size = sum((config.RAW_DIR / f).stat().st_size for f in files)
    print(f"{len(files)} files, {size / 1e9:.2f} GB in {config.RAW_DIR} ({elapsed:.0f}s)")
    print(f"snapshot_id: {config.SNAPSHOT_ID}")


if __name__ == "__main__":
    main()
