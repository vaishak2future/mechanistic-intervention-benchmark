"""Expose only the timed remote command interface to the streamed agent runner."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def execute(run: Path, command: list[str], note: str = "") -> dict:
    if not command or any(not isinstance(part, str) for part in command):
        raise ValueError("Supply a nonempty command argument list.")
    state = json.loads((run / "state.json").read_text())
    remaining = state["deadline"] - time.time()
    if state["status"] != "running" or remaining <= 0:
        raise ValueError("Agent session has ended.")
    wrapper = Path(__file__).with_name("agent_command.py")
    result = subprocess.run(
        [sys.executable, str(wrapper), "--run", str(run), "--note", note, "--", *command],
        capture_output=True, text=True, timeout=remaining + 5,
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    options = parser.parse_args()
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("MobileLLM workspace")

    @server.tool()
    def command(command: list[str], note: str = "") -> dict:
        """Run a command on the H100; use bash -lc for scripts. All paths are remote."""
        return execute(options.run, command, note)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
