"""Assemble one self-contained Harbor task without putting private data in Git."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "cdd21efc69cc89fa9608cfbad863f9eebb9dc9fb"
BASELINE_EVALUATOR_SHA256 = "ee36e5074c65cd36e5be657cd2753a169b2fd3fc486bc592a32ca07b0641e2d6"
PUBLIC_FILES = ("eval.py", "tasks.json")


def matching_baseline(holdout, cache):
    if hashlib.sha256((ROOT / "eval.py").read_bytes()).hexdigest() != BASELINE_EVALUATOR_SHA256:
        raise ValueError("The evaluator changed. Measure a new baseline before packaging.")
    digest = hashlib.sha256(holdout).hexdigest()
    # This image has the pinned source, vLLM and evaluator used by this task version.
    image = "im-4jG8f7AjuepDH3Nu7wqezd"
    for path in sorted(cache.glob("*.json")):
        entry = json.loads(path.read_text())
        identity = entry.get("identity", {})
        if identity.get("image_id") == image and identity.get("holdout_sha256") == digest and identity.get("gpu") == "H100":
            return entry
    raise ValueError("No measured baseline matches this holdout and pinned H100 image.")


def package(destination, holdout, baseline_cache):
    if destination.exists():
        raise ValueError(f"Choose a fresh output directory: {destination}")
    hidden = holdout.read_bytes()
    baseline = matching_baseline(hidden, baseline_cache)
    task = destination / "improve-mobilellm"
    shutil.copytree(ROOT / "harbor/mobilellm", task, ignore=shutil.ignore_patterns("__pycache__"))
    for name in PUBLIC_FILES:
        shutil.copyfile(ROOT / name, task / "environment" / name)
    shutil.copyfile(ROOT / "eval.py", task / "tests/eval.py")
    (task / "tests/holdout.json").write_bytes(hidden)
    (task / "tests/baseline.json").write_text(json.dumps(baseline["report"], indent=2) + "\n")
    (task / "tests/baseline-provenance.json").write_text(json.dumps({
        "source_revision": MODEL_REVISION, "cache_identity": baseline["identity"],
        "sampling": {"n": 4, "temperature": 0.6, "top_p": 0.95, "max_tokens": 3072, "seed": 0},
    }, indent=2) + "\n")
    (task / "tests/test.sh").chmod(0o755)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/tasks")
    parser.add_argument("--holdout", type=Path, default=ROOT / "private/tasks_holdout.json")
    parser.add_argument("--baseline-cache", type=Path, default=ROOT / "private/baselines")
    options = parser.parse_args()
    print(package(options.output, options.holdout, options.baseline_cache))


if __name__ == "__main__":
    main()
