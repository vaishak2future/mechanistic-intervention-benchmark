"""Operator-only launcher: start a GPU workspace, then freeze and grade it."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def docker(*arguments: str) -> str:
    return subprocess.check_output(["docker", *arguments], text=True).strip()


def mount(source: Path, destination: str, readonly: bool = True) -> list[str]:
    specification = f"type=bind,src={source.resolve()},dst={destination}"
    return ["--mount", specification + (",readonly" if readonly else "")]


def isolation() -> list[str]:
    return ["--gpus", "device=0", "--network", "none", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "512",
            "--memory", "32g", "--memory-swap", "32g", "--cpus", "8",
            "--shm-size", "2g", "--read-only", "--tmpfs", "/tmp:rw,size=8g"]


def start(options: argparse.Namespace) -> None:
    if options.seconds < 1:
        raise ValueError("Time budget must be positive.")
    base = options.base.resolve(strict=True)
    if not (base / "config.json").is_file():
        raise ValueError("Base must be a complete local Hugging Face model directory.")
    # Record immutable image identity, not a tag that could change before grading.
    image = docker("image", "inspect", "--format", "{{.Id}}", options.image)
    docker("run", "-d", "--name", options.name, *isolation(),
           "--label", f"benchmark.base={base}",
           "--mount", f"type=volume,dst=/workspace",
           *mount(base, "/base"), image, "sleep", str(options.seconds))
    print(f"Task: docker exec {options.name} cat /task/TASK.txt")
    print(f"Shell: docker exec -it {options.name} bash")


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


def grade(options: argparse.Namespace) -> None:
    details = json.loads(docker("inspect", options.name))[0]
    base = Path(details["Config"]["Labels"]["benchmark.base"])
    holdout = options.holdout.resolve(strict=True)
    report = options.report.resolve()
    if report.exists():
        raise ValueError("Choose a new report directory; existing results are preserved.")
    report.mkdir(parents=True)
    # End all training processes before taking the checkpoint snapshot.
    docker("stop", "--time", "10", options.name)
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        weights = staging / "model.safetensors"
        docker("cp", f"{options.name}:/workspace/submission/model.safetensors", str(weights))
        candidate = staging / "candidate"
        stage_checkpoint(base, weights, candidate)
        for label, model in [("baseline", base), ("submission", candidate)]:
            docker("run", "--rm", *isolation(),
                   "--user", f"{os.getuid()}:{os.getgid()}",
                   *mount(model, "/model"), *mount(holdout, "/hidden/tasks.json"),
                   *mount(report, "/reports", readonly=False),
                   details["Image"], "python", "/task/eval.py",
                   "--model", "/model", "--tasks", "/hidden/tasks.json",
                   "--report", f"/reports/{label}.json")
    baseline = json.loads((report / "baseline.json").read_text())["score"]
    submitted = json.loads((report / "submission.json").read_text())["score"]
    summary = {"baseline": baseline, "submission": submitted,
               "improvement": submitted - baseline}
    (report / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("start")
    launch.add_argument("--seconds", type=int, default=3600)
    launch.add_argument("--base", type=Path, required=True)
    launch.add_argument("--image", default="mobilellm-lab")
    launch.add_argument("--name", default="mobilellm-agent")
    launch.set_defaults(run=start)
    finish = commands.add_parser("grade")
    finish.add_argument("--name", default="mobilellm-agent")
    finish.add_argument("--holdout", type=Path, default=ROOT / "private/tasks_holdout.json")
    finish.add_argument("--report", type=Path, default=ROOT / "runs/final")
    finish.set_defaults(run=grade)
    options = parser.parse_args()
    options.run(options)


if __name__ == "__main__":
    main()
