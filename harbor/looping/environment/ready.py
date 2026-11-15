"""Prepare the GPU before the agent clock, keeping the warmed engine alive."""

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def engine_is_ready():
    try:
        with urlopen("http://127.0.0.1:8766", timeout=2) as response:
            return json.load(response).get("ready") is True
    except OSError:
        return False


def main():
    submission = Path("/workspace/patch.json")
    if not submission.exists():
        submission.write_text('{"edits": []}\n')
    with Path("/tmp/mobilellm-ready.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if engine_is_ready():
            return
        with Path("/tmp/warm-eval.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "/pilot/evaluate.py", "--serve"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if engine_is_ready():
                print("Ready: development baseline saved; vLLM warmed; reset verified.")
                return
            if process.poll() is not None:
                raise RuntimeError(Path("/tmp/warm-eval.log").read_text()[-6000:])
            time.sleep(1)
        process.terminate()
        raise TimeoutError("The resident evaluator did not become ready.")


if __name__ == "__main__":
    main()
