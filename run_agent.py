"""Run Codex with live public-message capture and the remote H100 command tool."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from agent_command import record
from modal_environment import save_state


class MessageRecorder:
    """Capture completed public messages once; never persist reasoning events."""

    def __init__(self, run):
        self.run = run
        self.seen = set()

    def receive(self, event):
        item = event.get("item", {})
        if event.get("type") != "item.completed" or item.get("type") != "agent_message":
            return
        identifier = item.get("id")
        if identifier and identifier in self.seen:
            return
        if identifier:
            self.seen.add(identifier)
        record(self.run, {"event": "agent_message", "message": item["text"],
                          "source": "codex.exec", "source_item_id": identifier})


def agent_command(run, state, directory, executable="codex"):
    """Use the recorded model and expose the remote bridge, not the host shell."""
    agent = state["agent"]
    settings = {
        "model_reasoning_effort": agent["reasoning_effort"],
        "approval_policy": "never",
        "web_search": "disabled",
        "project_doc_max_bytes": 0,
        "features.shell_tool": False,
        "features.view_image": False,
        "features.multi_agent": False,
        "features.memories": False,
        "mcp_servers.h100.command": sys.executable,
        "mcp_servers.h100.args": [str(Path(__file__).with_name("modal_agent_tool.py")), "--run", str(run)],
        "mcp_servers.h100.required": True,
        "mcp_servers.h100.tools.command.approval_mode": "approve",
        "mcp_servers.h100.tool_timeout_sec": max(1, int(state["deadline"] - time.time())),
    }
    command = [executable, "exec", "--json", "--ephemeral", "--ignore-user-config",
               "--ignore-rules", "--skip-git-repo-check", "--sandbox", "read-only",
               "--cd", str(directory), "--model", agent["model"]]
    for key, value in settings.items():
        command.extend(["-c", f"{key}={json.dumps(value)}"])
    command.append(
        "Improve the supplied model weights. Read /task/TASK.txt through the H100 command tool "
        "for the task and submission instructions. Choose your own approach. All work must "
        "happen through that tool in the remote workspace. The operator handles submission "
        "and hidden grading after you finish. Save the required checkpoint and finish before "
        f"the remaining {max(0, int(state['deadline'] - time.time()) - 15)} seconds expire."
    )
    return command


def capture(run, process):
    recorder = MessageRecorder(run)
    for line in process.stdout:
        event = json.loads(line)
        recorder.receive(event)
        if event.get("type") in {"error", "turn.failed"}:
            record(run, {"event": "runner_error", "message": json.dumps(event)})
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "mcp_tool_call" and item.get("status") == "failed":
            record(run, {"event": "tool_rejected", "message": json.dumps(item)})
    return process.wait()


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_agent(run, executable="codex"):
    state = json.loads((run / "state.json").read_text())
    remaining = state["deadline"] - time.time()
    if state["status"] != "running" or remaining <= 15:
        raise ValueError("Start a ready session with enough time for the agent and submission.")
    if state["agent"].get("capture"):
        raise ValueError("This run already has an agent capture; use a fresh session.")
    state["agent"].update(runner="codex-exec", capture="live-public-events-v1")
    save_state(run, state)
    with tempfile.TemporaryDirectory(prefix="mobilellm-agent-") as directory:
        command = agent_command(run, state, directory, executable)
        record(run, {"event": "runner_started", "message": "Live public-message capture started."})
        with (run / "runner.stderr.log").open("w") as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                                       stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
            timer = threading.Timer(max(0, remaining - 15), stop_process, args=(process,))
            timer.start()
            try:
                result = capture(run, process)
            except Exception as error:
                record(run, {"event": "runner_error", "message": str(error)})
                raise
            finally:
                timer.cancel()
                stop_process(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                process.stdout.close()
        record(run, {"event": "runner_finished", "message": f"Agent runner exited with code {result}."})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--codex", default="codex", help="Codex CLI executable")
    options = parser.parse_args()
    sys.exit(run_agent(options.run.resolve(), options.codex))


if __name__ == "__main__":
    main()
