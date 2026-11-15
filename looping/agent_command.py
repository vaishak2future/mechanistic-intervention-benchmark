"""Recorded command interface for a smoke-test agent; all commands run remotely."""

import argparse
import fcntl
import json
import sys
import time
import uuid
from pathlib import Path


def record(run: Path, event: dict) -> None:
    with (run / "trace.jsonl").open("a") as trace:
        # The runner and remote-command bridge share one ordered, append-only log.
        fcntl.flock(trace, fcntl.LOCK_EX)
        event = {"time": time.time(), "event_id": uuid.uuid4().hex, **event}
        trace.write(json.dumps(event) + "\n")
        trace.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--note", help="Brief action summary for the visible trace")
    parser.add_argument("--message", help="Public progress update to include in the trace")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    state = json.loads((options.run / "state.json").read_text())
    remaining = int(state["deadline"] - time.time())
    if state["status"] != "running" or remaining <= 0:
        parser.error("Agent session has ended.")
    command = options.command
    if command[:1] == ["--"]:
        command = command[1:]
    if options.message:
        record(options.run, {"event": "agent_message", "message": options.message})
        if not command:
            print(f"Time remaining: {remaining}s")
            return
    action_id = uuid.uuid4().hex
    record(options.run, {"event": "action", "action_id": action_id, "note": options.note,
                         "command": command, "remaining_seconds": remaining})
    if not command:
        print(f"Time remaining: {remaining}s")
        return
    import modal
    started = time.monotonic()
    try:
        sandbox = modal.Sandbox.from_id(state["sandbox_id"])
        process = sandbox.exec(*command, timeout=remaining)
        stdout, stderr = process.stdout.read(), process.stderr.read()
        process.wait()
        record(options.run, {"event": "result", "action_id": action_id, "stdout": stdout, "stderr": stderr,
                             "exit_code": process.returncode,
                             "duration_seconds": time.monotonic() - started})
        print(stdout, end="")
        print(stderr, end="", file=sys.stderr)
        print(f"\nTime remaining: {int(state['deadline'] - time.time())}s")
        sys.exit(process.returncode)
    except Exception as error:
        record(options.run, {"event": "error", "action_id": action_id, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
