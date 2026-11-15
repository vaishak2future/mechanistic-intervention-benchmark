"""Split full safetensors weights into separately transferable checkpoint parts."""

import argparse
import hashlib
import json
from pathlib import Path

PART_BYTES = 400 * 1024**2
MAX_PARTS = 10


def submit(weights: Path, destination: Path, part_bytes: int = PART_BYTES):
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("Provide a regular full-model safetensors file.")
    size = weights.stat().st_size
    if not 0 < size <= part_bytes * MAX_PARTS:
        raise ValueError("Weights must be nonempty and fit in ten 400 MiB parts.")
    destination.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    count = 0
    with weights.open("rb") as source:
        while chunk := source.read(part_bytes):
            count += 1
            (destination / f"weights.{count:02d}").write_bytes(chunk)
            digest.update(chunk)
    manifest = {"parts": count, "bytes": size, "sha256": digest.hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/workspace/submission"))
    options = parser.parse_args()
    print(json.dumps(submit(options.weights, options.output), indent=2))
