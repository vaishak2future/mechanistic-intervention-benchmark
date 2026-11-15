"""Accept submitted weights while retaining trusted model metadata."""

import shutil
from pathlib import Path


def stage_checkpoint(base: Path, weights: Path, destination: Path) -> None:
    """Use trusted metadata and only the submitted safetensors weight file."""
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("Submission must be a regular model.safetensors file.")
    if not 0 < weights.stat().st_size <= 10 * 1024**3:
        raise ValueError("Submission must be nonempty and at most 10 GiB.")
    destination.mkdir()
    for source in base.iterdir():
        if source.is_file() and source.suffix in {".json", ".jinja", ".model", ".txt"}:
            if source.name.endswith(".index.json"):
                continue
            shutil.copyfile(source, destination / source.name)
    shutil.copyfile(weights, destination / "model.safetensors")
