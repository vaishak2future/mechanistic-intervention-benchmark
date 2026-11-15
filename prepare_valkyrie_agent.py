"""Adapt the official Codex package to the model used in this experiment."""

import argparse
import shutil
from pathlib import Path


def prepare(source, destination):
    if destination.exists():
        raise ValueError(f"Choose a fresh agent directory: {destination}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("agent_output_example", "__pycache__"))
    contract = destination / "contract.yaml"
    text = contract.read_text().replace("codex-v1.0.0", "codex-mobilellm-v1")
    text = text.replace("default: gpt-5.3-codex", "default: gpt-6.1-sol")
    text = text.replace("    choices:\n", "    choices:\n      - gpt-6.1-sol\n")
    contract.write_text(text)
    setup = destination / "setup.sh"
    setup.write_text(setup.read_text().replace("/bundle/codex-v1.0.0", "/bundle/codex-mobilellm-v1"))
    runner = destination / "collect_outputs.py"
    text = runner.read_text()
    marker = '        "--json",\n'
    if marker not in text:
        raise ValueError("The upstream agent changed; review its command before adapting it.")
    text = text.replace(marker, marker + '        "-c",\n        \'model_reasoning_effort="high"\',\n', 1)
    runner.write_text(text)
    (destination / "MODEL.txt").write_text("Model: gpt-6.1-sol\nReasoning effort: high\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    options = parser.parse_args()
    print(prepare(options.source, options.destination))
