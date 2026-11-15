"""Use the tested Claude Code harness with Tinker's Anthropic-compatible API."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'tinker-looping-v1'
BASE_URL = 'https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api'


def main():
    destination = ROOT/'build/agents'/NAME
    shutil.copytree(ROOT/'build/agents/claude-looping-v3', destination,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for path in destination.iterdir():
        if path.suffix in {'.py', '.sh', '.yaml', '.txt', '.md'}:
            path.write_text(path.read_text().replace('claude-looping-v3', NAME))
    runner = destination/'collect_outputs.py'
    source = runner.read_text()
    source = source.replace('    env["ANTHROPIC_MODEL"] = model_name',
        '    env["ANTHROPIC_MODEL"] = model_name\n'
        f'    env["ANTHROPIC_BASE_URL"] = {BASE_URL!r}\n'
        '    env["ANTHROPIC_API_KEY"] = env.pop("TINKER_API_KEY")\n'
        '    for tier in ["HAIKU", "SONNET", "OPUS"]:\n'
        '        env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] = model_name')
    source = source.replace('"reasoning_effort": "high",',
        f'"reasoning_effort": "high",\n            "provider": "tinker",\n            "base_url": {BASE_URL!r},')
    runner.write_text(source)
    contract = destination/'contract.yaml'
    contract.write_text(contract.read_text().replace('anthropic/claude-sonnet-4-20250514',
                                                     'thinkingmachines/Inkling-2-Small'))
    (destination/'MODEL.txt').write_text('Model: thinkingmachines/Inkling-2-Small\n'
        'Provider: Tinker, Anthropic-compatible endpoint\nReasoning: high\nClaude Code: 2.1.325\n')
    print(destination)


if __name__ == '__main__':
    main()
