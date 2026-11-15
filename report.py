"""Render grading results and an optional agent trace as one readable HTML file."""

import argparse
import html
import json
import re
import shlex
from collections import Counter
from pathlib import Path


STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 system-ui, sans-serif; max-width: 960px; margin: 40px auto;
       padding: 0 20px; color: light-dark(#202124, #e4e5e7);
       background: light-dark(#fff, #17181a); }
h1 { font-size: 26px; margin-bottom: 4px; }
h2 { font-size: 19px; margin-top: 32px; }
h3 { font-size: 15px; margin: 16px 0 8px; }
p { margin: 6px 0 16px; }
a { color: light-dark(#245c9c, #9ac4ff); }
.muted { color: light-dark(#62666c, #acb0b7); }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { text-align: left; padding: 9px 8px; border-bottom: 1px solid light-dark(#e2e4e8, #3b3d42); }
.number { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pass { color: light-dark(#19673b, #85d2a5); }
.fail { color: light-dark(#a12c2c, #ffaaa4); }
details { padding: 10px 0; border-bottom: 1px solid light-dark(#e2e4e8, #3b3d42); }
details details { margin-left: 12px; border: 0; }
summary { cursor: pointer; overflow-wrap: anywhere; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.5 ui-monospace, monospace;
      padding: 12px; background: light-dark(#f5f6f7, #232529); border-radius: 4px; }
.compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.compare > div { min-width: 0; }
@media (max-width: 600px) { .compare { grid-template-columns: 1fr; gap: 0; }
body { margin-top: 20px; padding: 0 12px; } th,td { padding: 8px 4px; } }
"""


def escape(value):
    return html.escape(str(value))


def code(value):
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value)).replace("\r", "\n")
    return f"<pre>{escape(clean)}</pre>"


def details(title, content):
    return f"<details><summary>{title}</summary>{content}</details>"


def passed(task):
    return sum(sample['passed'] for sample in task['samples'])


def percent(score):
    return f"{score * 100:g}%"


def sample_view(label, sample):
    status = 'PASS' if sample['passed'] else 'FAIL'
    output = sample.get('output', '')
    final = output.rsplit('</think>', 1)[-1].strip()
    content = (f"<h3>{label} · <span class='{status.lower()}'>{status}</span></h3>"
               f"<p>{escape(sample['reason'])}</p>" + code(final or '(empty final answer)'))
    if output.strip() != final:
        content += details('Full model output', code(output))
    return f'<div>{content}</div>'


def result_section(baseline, submission, tasks):
    originals = {task['task_id']: task for task in baseline['tasks']}
    rows, drilldowns = [], []
    for task in submission['tasks']:
        original = originals[task['task_id']]
        task_id = escape(task['task_id'])
        count = len(task['samples'])
        before, after = passed(original), passed(task)
        rows.append(f"<tr><td><a href='#task-{task_id}'>{task_id}</a></td>"
                    f"<td>{escape(task['category'].replace('_', ' '))}</td>"
                    f"<td class='number'>{before}/{len(original['samples'])}</td>"
                    f"<td class='number'>{after}/{count}</td></tr>")
        source = tasks.get(task['task_id'])
        content = ''
        if source:
            conversation = '\n\n'.join(f"{m['role']}: {m['content']}" for m in source['messages'] if m['role'] != 'system')
            content += code(conversation)
            content += details('Expected result', code(json.dumps(source['expect'], indent=2)))
        for number, (old, new) in enumerate(zip(original['samples'], task['samples'], strict=True), 1):
            content += details(f"Sample {number} · {'PASS' if new['passed'] else 'FAIL'}",
                               '<div class="compare">' + sample_view('Baseline', old) + sample_view('Submission', new) + '</div>')
        drilldowns.append(f"<div id='task-{task_id}'>" + details(f"{task_id} · {after}/{count} passed", content) + '</div>')
    return ('<h2 id="tasks">Task results</h2><table><thead><tr><th>Task</th><th>Category</th>'
            '<th class="number">Baseline</th><th class="number">Submission</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table><h2>Sample details</h2>' + ''.join(drilldowns))


def trace_section(path, started_at):
    if not path.exists():
        return '<h2 id="trace">Agent trace</h2><p class="muted">No trace recorded for this run.</p>'
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    events.sort(key=lambda event: event['time'])
    actions, finished, entries = [], set(), []
    for event in events:
        elapsed = max(0, event['time'] - started_at)
        timing = ('Archived after run; original timestamp unavailable' if event.get('archived_after_run')
                  else f'{int(elapsed // 60)}:{int(elapsed % 60):02d}')
        kind = event['event']
        if kind == 'agent_message':
            message = escape(event['message']).replace('\n', '<br>')
            entries.append(f'<article><h3>{timing} · Agent</h3><p>{message}</p></article>')
        elif kind == 'action':
            actions.append(event)
            number = len(actions)
            title = f"{timing} · Tool call {number:02d} · {escape(event.get('note') or 'Command')}"
            entries.append(f'<div id="call-{number}">' + details(title, code(shlex.join(event.get('command', [])))) + '</div>')
        elif kind in {'result', 'error'} and (event.get('action_id') or kind == 'result'):
            candidates = [(i, action) for i, action in enumerate(actions, 1) if i not in finished]
            if event.get('action_id'):
                candidates = [(i, action) for i, action in candidates if action.get('action_id') == event['action_id']]
            if candidates:
                began = event['time'] - event.get('duration_seconds', 0)
                number, _ = min(candidates, key=lambda pair: abs(pair[1]['time'] - began))
                finished.add(number)
                label = f'Tool result {number:02d}'
                content = f'<p><a href="#call-{number}">View call {number:02d}</a></p>'
            else:
                label, content = 'Unpaired tool result', ''
            content += f"<p class='muted'>Exit {escape(event.get('exit_code', 'error'))} · {event.get('duration_seconds', 0):.1f}s</p>"
            for field in ('stdout', 'stderr', 'message'):
                if event.get(field):
                    content += f'<h3>{field}</h3>' + code(event[field])
            entries.append(details(f'{timing} · {label}', content))
        else:
            entries.append(details(f'{timing} · {escape(kind.replace("_", " "))}',
                                   code(event.get('message', json.dumps(event, indent=2)))))
    return (f'<h2 id="trace">Agent trace · {len(actions)} actions</h2>'
            '<p class="muted">One chronological transcript of public agent messages, tool calls, and results. '
            'Tool details are expandable. Private internal reasoning is not recorded.</p>'
            + ''.join(entries))


def render_live_trace(run):
    """Refresh a standalone transcript after each event, before grading exists."""
    state = json.loads((run / 'state.json').read_text())
    content = trace_section(run / 'trace.jsonl', state.get('started_at', 0))
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Agent trace</title><style>' + STYLE + '</style><body>'
                + content + '</body></html>')
    temporary = run / 'trace.html.tmp'
    temporary.write_text(document)
    temporary.replace(run / 'trace.html')


def render_report(run, task_path=None):
    reports = run / 'reports'
    baseline = json.loads((reports / 'baseline.json').read_text())
    submission = json.loads((reports / 'submission.json').read_text())
    state = json.loads((run / 'state.json').read_text())
    tasks = {task['id']: task for task in json.loads(task_path.read_text())} if task_path else {}
    total = sum(len(task['samples']) for task in submission['tasks'])
    failures = Counter(sample['reason'] for task in submission['tasks'] for sample in task['samples'] if not sample['passed'])
    delta = 100 * (submission['score'] - baseline['score'])
    content = ('<h1>MobileLLM evaluation</h1>'
               f'<p class="muted">Hidden holdout · {len(submission["tasks"])} tasks · {total} samples · {escape(state["gpu"])}</p>'
               '<nav><a href="#tasks">Task results</a> · <a href="#trace">Agent trace</a> · <a href="#protocol">Protocol</a></nav>'
               '<table><thead><tr><th>Baseline</th><th>Submission</th><th>Change</th></tr></thead>'
               f'<tbody><tr><td>{percent(baseline["score"])}</td><td><strong>{percent(submission["score"])}</strong></td><td>{delta:+g} pp</td></tr></tbody></table>')
    if failures:
        content += '<h2>Remaining failures</h2><ul>' + ''.join(f'<li>{count} samples: {escape(reason)}</li>' for reason, count in failures.items()) + '</ul>'
    content += result_section(baseline, submission, tasks)
    content += trace_section(run / 'trace.jsonl', state.get('started_at', 0))
    content += ('<h2 id="protocol">Protocol</h2>'
                '<p>Scores are mean per-task sample success. Baseline and submission use the same evaluator, task data, and sampling settings; the baseline may be cached. Sample numbers are within-run indices, not matched random trials.</p>'
                '<p class="muted">One small smoke run, not a leaderboard ranking. Multiple samples of the same task are not independent benchmark tasks. Agent-created transfer checks are development evidence, not additional hidden evaluation.</p>'
                f'<p>Image: <code>{escape(state["image_id"])}</code></p>')
    agent = state.get('agent', {})
    content += f"<p>Agent model: <code>{escape(agent.get('model', 'Not recorded'))}</code> · Reasoning: {escape(agent.get('reasoning_effort', 'Not recorded'))}</p>"
    content += f"<p>Task setup: {escape(state.get('task_setup', 'Original scaffolded setup'))}</p>"
    if 'preparation_seconds' in state:
        content += f'<p>Preparation before the agent clock: {state["preparation_seconds"]:.1f}s.</p>'
    content += '<p>Presentation references: <a href="https://inspect.aisi.org.uk/log-viewer.html">Inspect sample viewer</a> · <a href="https://www.swebench.com/">SWE-bench results</a>.</p>'
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>MobileLLM evaluation</title><style>' + STYLE + '</style></head><body>' + content + '</body></html>')
    destination = reports / 'report.html'
    destination.write_text(document)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--tasks', type=Path)
    args = parser.parse_args()
    print(render_report(args.run, args.tasks))
