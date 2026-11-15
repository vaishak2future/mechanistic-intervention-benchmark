"""Trusted entrypoint for one fresh Modal grading sandbox."""

import argparse
import subprocess
import sys
from pathlib import Path

from environment import stage_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("baseline", "submission"))
    options = parser.parse_args()
    model = Path("/base")
    if options.kind == "submission":
        model = Path("/workspace/candidate")
        stage_checkpoint(Path("/base"), Path("/submission/submission/model.safetensors"), model)
    subprocess.run([
        sys.executable, "/task/eval.py", "--model", str(model),
        "--tasks", "/workspace/holdout.json", "--report", "/workspace/report.json",
    ], check=True)


if __name__ == "__main__":
    main()
