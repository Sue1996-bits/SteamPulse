"""Write data/published/manifest.json describing the current published outputs.

The manifest links every output to its source snapshot, the cohort/label/feature
versions, the code revision, the dependency lock file and a run ID. It is generated,
never hand-edited: run `make manifest` after a pipeline run, or call write_manifest()
at the end of the publish step.
"""
import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from pipeline import config

MANIFEST_NAME = "manifest.json"


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=config.ROOT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return (out.stdout or out.stderr).strip()


def _code_revision() -> dict:
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict:
    import pyspark

    java = _run(["java", "-version"])
    return {
        "python": platform.python_version(),
        "pyspark": pyspark.__version__,
        "java": java.splitlines()[0] if java else None,
        "uv_lock_sha256": _sha256(config.ROOT / "uv.lock"),
    }


def describe_output(path: Path) -> dict:
    """Row count, file count and bytes for a Parquet file or partitioned directory.

    Reads Parquet footers only, so it is cheap even for large outputs.
    """
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    return {
        "path": str(path.relative_to(config.ROOT)) if path.is_relative_to(config.ROOT) else str(path),
        "files": len(files),
        "rows": sum(pq.ParquetFile(f).metadata.num_rows for f in files),
        "bytes": sum(f.stat().st_size for f in files),
    }


def build_manifest(outputs: dict[str, Path], run_id: str | None = None) -> dict:
    return {
        "run_id": run_id or uuid.uuid4().hex[:12],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "dataset": config.HF_DATASET,
            "snapshot_id": config.SNAPSHOT_ID,
        },
        "versions": {
            "cohort_policy": config.COHORT_POLICY,
            "cohort_version": config.COHORT_VERSION,
            "label_version": config.LABEL_VERSION,
            "feature_version": config.FEATURE_VERSION,
        },
        "code": _code_revision(),
        "runtime": _runtime(),
        "outputs": {name: describe_output(p) for name, p in sorted(outputs.items())},
    }


def write_manifest(outputs: dict[str, Path], out_dir: Path = config.PUBLISH_DIR,
                   run_id: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(build_manifest(outputs, run_id), indent=2) + "\n")
    return path


def discover_outputs(publish_dir: Path = config.PUBLISH_DIR) -> dict[str, Path]:
    """Each top-level folder or .parquet file in publish_dir is one output table."""
    if not publish_dir.exists():
        return {}
    return {
        p.stem: p for p in publish_dir.iterdir()
        if p.is_dir() or p.suffix == ".parquet"
    }


def main() -> None:
    path = write_manifest(discover_outputs())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
