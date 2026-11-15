"""Operator-only Modal sandbox lifecycle; credentials never enter the sandbox."""

import argparse
import hashlib
import json
import os
import shlex
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = ("eval.py", "TASK.txt", "tasks.json", "environment.py", "grade_checkpoint.py",
                "prepare.py", "warm_eval.py", "download_model.py")


def save_state(run: Path, state: dict) -> None:
    temporary = run / "state.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(run / "state.json")


def build_image(modal, model: str, revision: str, secret: str | None):
    """All downloads, GPU checks, and compilation happen in Modal's image builder."""
    image = (
        modal.Image.from_registry(
            "vllm/vllm-openai:v0.11.0",
            setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/local/bin/python"],
        )
        .entrypoint([])
        .pip_install("peft==0.17.1", "accelerate==1.10.1")
        .run_commands("mkdir -p /workspace /task /cache /prepared")
        .env({"VLLM_NO_USAGE_STATS": "1", "VLLM_CACHE_ROOT": "/cache/vllm",
              "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor", "TRITON_CACHE_DIR": "/cache/triton"})
        .workdir("/workspace")
    )
    image = image.add_local_file(ROOT / "download_model.py", "/task/download_model.py", copy=True)
    image = image.run_commands(
        shlex.join(["python", "/task/download_model.py", model, "--revision", revision]),
        secrets=[modal.Secret.from_name(secret)] if secret else [],
    ).env({"HF_HUB_OFFLINE": "1"})
    for filename in PUBLIC_FILES:
        image = image.add_local_file(ROOT / filename, f"/task/{filename}", copy=True)
    return image.run_commands("python /task/prepare.py", gpu="H100")


def create_sandbox(modal, app, image, gpu: str, seconds: int, volumes=None):
    return modal.Sandbox.create(
        "sleep", "infinity", app=app, image=image, gpu=gpu,
        cpu=(8, 8), memory=(32768, 32768), timeout=seconds,
        block_network=True, workdir="/workspace", volumes=volumes or {},
    )


def build(modal, options):
    os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2026.10")
    options.run.mkdir(parents=True, exist_ok=True)
    if (options.run / "state.json").exists():
        raise ValueError("Choose a new run directory; this one already has a built image.")
    app = modal.App.lookup("mobilellm-lab", create_if_missing=True)
    with modal.enable_output():
        image = build_image(modal, options.model, options.revision, options.hf_secret).build(app=app)
    save_state(options.run, {"image_id": image.object_id, "gpu": "H100", "status": "built", "task_setup": "weights-dev-eval-v2"})
    print(f"Prepared H100 image: {image.object_id}. No agent timer has started.")


def start(modal, options):
    state = json.loads((options.run / "state.json").read_text())
    if not options.agent_model.strip() or not options.agent_reasoning.strip():
        raise ValueError("Record an explicit agent model and reasoning effort before starting.")
    if state["status"] != "built":
        raise ValueError("Build a fresh run before starting its agent.")
    if not 1 <= options.seconds <= 72000:
        raise ValueError("Agent duration must be between 1 and 72000 seconds.")
    from watchdog import app as timer_app, enforce_deadline
    timer_app.deploy()
    app = modal.App.lookup("mobilellm-lab", create_if_missing=True)
    image = modal.Image.from_id(state["image_id"])
    volume_name = f"mobilellm-{uuid.uuid4().hex}"
    volume = modal.Volume.from_name(volume_name, create_if_missing=True, version=2)
    preparing = time.monotonic()
    sandbox = create_sandbox(modal, app, image, "H100", options.seconds + 3600,
                             {"/workspace": volume})
    try:
        if state.get("task_setup") == "weights-dev-eval-v2":
            run_command(sandbox, ["python", "-c",
                "from pathlib import Path; assert not Path('/task/train.py').exists(); assert not Path('/task/train.jsonl').exists()"], 30)
        # Warm in the same live process that the agent will use, before its clock.
        run_command(sandbox, ["bash", "-lc",
            "nohup python /task/warm_eval.py --serve > /tmp/warm-eval.log 2>&1 < /dev/null & echo $! > /tmp/warm-eval.pid"], 30)
        run_command(sandbox, ["python", "-c", WAIT_FOR_ENGINE], 3500)
        for filename in ("baseline.json", "readiness.json"):
            report = sandbox.filesystem.read_text(f"/prepared/{filename}")
            (options.run / filename).write_text(report)
        readiness = json.loads((options.run / "readiness.json").read_text())
        if not readiness.get("ready"):
            raise RuntimeError("Image readiness checks did not pass.")
        started = time.time()
        deadline = started + options.seconds
        timer = enforce_deadline.spawn(sandbox.object_id, deadline)
        state.update(sandbox_id=sandbox.object_id, volume_name=volume_name,
                     status="running", started_at=started, deadline=deadline,
                     agent={"model": options.agent_model, "reasoning_effort": options.agent_reasoning,
                            "runner": "codex-subagent", "resolved_revision": None},
                     timer_id=timer.object_id, preparation_seconds=time.monotonic() - preparing)
        save_state(options.run, state)
    except BaseException:
        sandbox.terminate(wait=True)
        raise
    print(f"Ready: {sandbox.object_id}. Agent budget: {options.seconds}s. Task: /task/TASK.txt")


WAIT_FOR_ENGINE = """
import time
from pathlib import Path
from urllib.request import urlopen
for attempt in range(1700):
    try:
        with urlopen('http://127.0.0.1:8765', timeout=2) as response:
            assert response.status == 200
        break
    except OSError:
        pid = Path('/tmp/warm-eval.pid').read_text().strip()
        process = Path('/proc') / pid / 'stat'
        if not process.exists() or process.read_text().split()[2] == 'Z':
            raise RuntimeError(Path('/tmp/warm-eval.log').read_text()[-6000:])
        time.sleep(2)
else:
    raise RuntimeError('Warm engine failed; inspect /tmp/warm-eval.log')
"""


def run_command(sandbox, command: list[str], seconds: int = 3600) -> None:
    process = sandbox.exec(*command, timeout=seconds)
    print(process.stdout.read(), end="")
    print(process.stderr.read(), end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Sandbox command failed with exit code {process.returncode}.")


def submit(modal, run: Path, state: dict) -> None:
    """Flush cloud storage and stop training; never download weights locally."""
    if state["status"] != "running":
        raise ValueError("Only a running session can submit.")
    sandbox = modal.Sandbox.from_id(state["sandbox_id"])
    try:
        if time.time() >= state["deadline"]:
            raise ValueError("Agent deadline has passed; late submissions are rejected.")
        run_command(sandbox, ["sync", "/workspace"], 60)
    except BaseException:
        state["status"] = "failed"
        raise
    else:
        state["status"] = "submitted"
        state["submitted_at"] = time.time()
    finally:
        sandbox.terminate(wait=True)
        save_state(run, state)
    print("Training ended. Checkpoint stays in Modal storage for isolated grading.")


def baseline_identity(state: dict, holdout: bytes) -> dict:
    # The immutable image fixes weights, tokenizer, evaluator, and sampling defaults.
    return {"version": 1, "image_id": state["image_id"], "gpu": state["gpu"],
            "holdout_sha256": hashlib.sha256(holdout).hexdigest()}


def grade_checkpoint(modal, state: dict, holdout: bytes, kind: str, seconds: int) -> dict:
    app = modal.App.lookup("mobilellm-lab", create_if_missing=True)
    image = modal.Image.from_id(state["image_id"])
    volumes = {}
    if kind == "submission":
        volume = modal.Volume.from_name(state["volume_name"])
        volumes["/submission"] = volume.with_mount_options(read_only=True)
    sandbox = create_sandbox(modal, app, image, state["gpu"], seconds, volumes)
    try:
        # Upload the exact bytes used in the cache key, even if the file later changes.
        sandbox.filesystem.write_bytes(holdout, "/workspace/holdout.json")
        run_command(sandbox, ["python", "/task/grade_checkpoint.py", kind], seconds)
        return json.loads(sandbox.filesystem.read_text("/workspace/report.json"))
    finally:
        sandbox.terminate(wait=True)


def cached_baseline(modal, options, state: dict, holdout: bytes) -> dict:
    identity = baseline_identity(state, holdout)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = options.baseline_cache
    path = cache / f"{key}.json"
    if path.exists():
        entry = json.loads(path.read_text())
        if entry["identity"] != identity:
            raise ValueError("Baseline cache identity does not match; remove the invalid entry.")
        print("Using cached hidden baseline.")
        return entry["report"]
    report = grade_checkpoint(modal, state, holdout, "baseline", options.seconds)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / f"{key}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps({"identity": identity, "report": report}, indent=2) + "\n")
    temporary.replace(path)
    print("Hidden baseline cached outside the agent environment.")
    return report


def grade(modal, options, state: dict) -> None:
    if state["status"] != "submitted":
        raise ValueError("Submit and end training before hidden grading.")
    holdout = options.holdout.read_bytes()
    report_directory = options.run / "reports"
    report_directory.mkdir(exist_ok=False)
    baseline = cached_baseline(modal, options, state, holdout)
    (report_directory / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    submission = grade_checkpoint(modal, state, holdout, "submission", options.seconds)
    (report_directory / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    scores = {"baseline": baseline["score"], "submission": submission["score"],
              "improvement": submission["score"] - baseline["score"]}
    (report_directory / "summary.json").write_text(json.dumps(scores, indent=2) + "\n")
    state["status"] = "graded"
    save_state(options.run, state)
    from report import render_report
    report_path = render_report(options.run, options.holdout)
    print(json.dumps(scores, indent=2))
    print(f"Report: {report_path}")


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/modal")
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("build")
    prepare.add_argument("--model", default="robbiemu/MobileLLM-R1-950M-mlx-fp16")
    prepare.add_argument("--revision", default="main")
    prepare.add_argument("--hf-secret", help="Modal secret containing HF_TOKEN; only used during download")
    launch = commands.add_parser("start")
    launch.add_argument("--seconds", type=int, default=3600)
    launch.add_argument("--agent-model", required=True, help="Exact model identifier passed to the agent runner")
    launch.add_argument("--agent-reasoning", required=True, help="Reasoning effort passed to the agent runner")
    execute = commands.add_parser("exec")
    execute.add_argument("command", nargs=argparse.REMAINDER)
    commands.add_parser("submit")
    commands.add_parser("stop")
    for name in ("baseline", "grade"):
        command = commands.add_parser(name)
        command.add_argument("--holdout", type=Path, default=ROOT / "private/tasks_holdout.json")
        command.add_argument("--baseline-cache", type=Path, default=ROOT / "private/baselines")
        command.add_argument("--seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> None:
    options = parse_options()
    # Local tests and --help don't need cloud dependencies or credentials.
    import modal

    if options.action in {"build", "start"}:
        (build if options.action == "build" else start)(modal, options)
        return
    state = json.loads((options.run / "state.json").read_text())
    if options.action == "baseline":
        cached_baseline(modal, options, state, options.holdout.read_bytes())
    elif options.action == "submit":
        submit(modal, options.run, state)
    elif options.action == "grade":
        grade(modal, options, state)
    elif options.action == "stop":
        modal.Sandbox.from_id(state["sandbox_id"]).terminate()
        if state["status"] == "running":
            state["status"] = "stopped"
        save_state(options.run, state)
    else:
        if state["status"] != "running":
            raise ValueError("The training session has ended.")
        command = options.command
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            raise ValueError("Provide a command after exec.")
        remaining = int(state["deadline"] - time.time())
        if remaining < 1:
            raise ValueError("Agent deadline has passed.")
        run_command(modal.Sandbox.from_id(state["sandbox_id"]), command, remaining)


if __name__ == "__main__":
    main()
