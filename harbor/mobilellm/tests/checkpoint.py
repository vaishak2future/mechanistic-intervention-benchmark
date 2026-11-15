"""Reassemble checked weight parts while retaining trusted model metadata."""

import hashlib
import json
import shutil
from pathlib import Path

PART_BYTES = 400 * 1024**2
MAX_PARTS = 10


def regular_file(path: Path, maximum: int):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or nonregular submission file: {path.name}")
    if not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"Invalid submission file size: {path.name}")


def stage_checkpoint(base: Path, submission: Path, destination: Path) -> None:
    if submission.is_symlink() or not submission.is_dir():
        raise ValueError("Submission must be a regular directory.")
    manifest_path = submission / "manifest.json"
    regular_file(manifest_path, 4096)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid submission manifest.") from error
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object.")
    count, size, expected_hash = (manifest.get(key) for key in ("parts", "bytes", "sha256"))
    if type(count) is not int or not 1 <= count <= MAX_PARTS:
        raise ValueError("Expected one to ten weight parts.")
    if type(size) is not int or not 0 < size <= PART_BYTES * count:
        raise ValueError("Invalid total weight size.")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Invalid SHA-256 digest.")
    parts = [submission / f"weights.{number:02d}" for number in range(1, count + 1)]
    for part in parts:
        regular_file(part, PART_BYTES)
    if sum(part.stat().st_size for part in parts) != size:
        raise ValueError("Weight parts do not match the declared size.")
    destination.mkdir()
    digest = hashlib.sha256()
    with (destination / "model.safetensors").open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(8 * 1024**2):
                    output.write(chunk)
                    digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Weight checksum mismatch.")
    for source in base.iterdir():
        if source.is_file() and source.suffix in {".json", ".jinja", ".model", ".txt"}:
            if not source.name.endswith(".index.json"):
                shutil.copyfile(source, destination / source.name)
