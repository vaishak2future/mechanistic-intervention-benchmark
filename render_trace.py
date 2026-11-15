"""Render the recorded agent actions as a readable, chronological transcript."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    options = parser.parse_args()
    state = json.loads((options.run / 'state.json').read_text())
    events = [json.loads(line) for line in (options.run / 'trace.jsonl').read_text().splitlines()]
    lines = ['MobileLLM: one-agent, 10-minute smoke run',
             'Trace contains public action summaries and executed commands, not private reasoning.',
             f"GPU: {state['gpu']} | Image: {state['image_id']}",
             f"Preparation outside agent budget: {state.get('preparation_seconds', 0):.1f}s", '']
    for event in sorted(events, key=lambda event: event['time']):
        elapsed = event['time'] - state['started_at']
        lines.append(f"[{elapsed:7.1f}s] {event['event'].upper()}")
        for key in ('note', 'command', 'remaining_seconds', 'duration_seconds', 'exit_code', 'message'):
            if event.get(key) is not None:
                value = json.dumps(event[key]) if key == 'command' else event[key]
                lines.append(f'{key}: {value}')
        for key in ('stdout', 'stderr'):
            if event.get(key):
                lines.extend([f'{key}:', event[key].rstrip()])
        lines.append('')
    summary = options.run / 'reports/summary.json'
    if summary.exists():
        lines.extend(['FINAL HIDDEN GRADING (after agent submission)', summary.read_text()])
    (options.run / 'trace.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
