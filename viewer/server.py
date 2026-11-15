"""A read-only, local view of the saved experiment. Run: python3 viewer/server.py."""
import argparse
import json
import math
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent
SOURCES = {
    'comparison': ['mechanistic-submission-45m', 'inkling-comparison-45m'],
    'smoke': ['mechanistic-sol-10m-check', 'inkling-claude-smoke-10m'],
}
LABELS = {'sol': 'GPT-6.1 Sol', 'astra': 'GPT-6 Astra', 'opus': 'Claude Opus 5.5',
          'sonnet': 'Claude Sonnet 5.5', 'inkling': 'Inkling 2 Small'}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def redact(value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    for filename in ['openai.token', 'anthropic.token', 'docent.token', 'thinky.token', 'tinker.token']:
        path = Path.home() / 'Projects/creds' / filename
        if path.exists():
            for secret in re.findall(r'(?:sk-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{40,})', path.read_text()):
                text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'\b(?:sk-|as-)[A-Za-z0-9_-]{20,}', '[REDACTED]', text)
    return re.sub(r'(?i)((?:authorization|api[_-]?key|access[_-]?token)\s*[:=]\s*)["\']?[^\s"\']{12,}', r'\1[REDACTED]', text)


def score_for(status, scores):
    reward = scores.get('reward') if scores else None
    if status != 'FINISHED' or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        return None
    return reward * 100


def saved_runs():
    """Use only explicitly listed experiment directories; never expose arbitrary files."""
    rows = []
    for cohort, sources in SOURCES.items():
        for source in sources:
            folder = ROOT / 'runs' / source
            status = read_json(folder / 'status.json', {})
            entries = {item['id']: item for item in status.get('entries', [])}
            for receipt in read_json(folder / 'runs.json', []):
                directory = folder / f"{receipt['name']}-{receipt['repeat']}"
                collected = read_json(directory / 'collected.json', {})
                task = read_json(directory / 'task.json', {})
                state = collected.get('status') or task.get('status') or entries.get(receipt['run_id'], {}).get('status') or 'AWAITING_UPDATE'
                scores = collected.get('scores') or (task.get('evaluation_result') or {}).get('rewards') or {}
                trajectories = list((directory / 'outputs').rglob('trajectory.jsonl'))
                collection = read_json(folder / 'docent.json', {})
                upload = read_json(directory / 'docent.json', {})
                docent = collection.get('url')
                if docent and upload.get('agent_run_id'):
                    docent += '/agent_run/' + upload['agent_run_id']
                row = {**receipt, 'cohort': cohort, 'label': LABELS.get(receipt['name'], receipt['name']),
                       'status': state, 'scores': scores, 'score': score_for(state, scores),
                       'timings': collected.get('timings') or task.get('task_breakdown') or {},
                       'error': collected.get('error') or task.get('error_message'),
                       'has_trace': len(trajectories) == 1 or (directory / 'trace.jsonl').exists(), 'docent_url': docent,
                       '_directory': directory}
                rows.append(row)
    return rows


def leaderboard(rows):
    groups = {}
    for row in rows:
        key = (row['cohort'], row['model'], row['agent'], row['budget_seconds'])
        groups.setdefault(key, []).append(row)
    result = []
    for (cohort, model, agent, budget), trials in groups.items():
        completed = [r for r in trials if r['score'] is not None]
        scores = [r['score'] for r in completed]
        result.append({'cohort': cohort, 'model': model, 'agent': agent, 'budget_seconds': budget,
                       'label': trials[0]['label'], 'name': trials[0]['name'], 'completed': len(completed),
                       'total': len(trials), 'mean': mean(scores) if scores else None,
                       'duration': mean([r['timings']['agent_run_duration'] for r in completed if r.get('timings', {}).get('agent_run_duration') is not None])
                                   if any(r.get('timings', {}).get('agent_run_duration') is not None for r in completed) else None,
                       'min': min(scores) if scores else None, 'max': max(scores) if scores else None,
                       'diagnostics': {k: mean([r['scores'][k] for r in completed if k in r['scores']])
                                       if any(k in r['scores'] for r in completed) else None
                                       for k in ['repair', 'specificity', 'transfer']},
                       'trials': [r['run_id'] for r in trials]})
    return sorted(result, key=lambda r: (r['mean'] is None, -(r['mean'] or 0), r['label']))


def normalize_trace(events):
    """Keep public messages, tool calls, and results in their original event order."""
    messages, started, tools, seen = [], set(), {}, set()
    def add(kind, label, text, index, timestamp=None, call_id=None):
        messages.append({'kind': kind, 'label': label, 'text': redact(text),
                         'event_index': index, 'timestamp': timestamp, 'call_id': call_id})
    for index, event in enumerate(events):
        item, kind = event.get('item', {}), event.get('type')
        item_kind = item.get('type')
        stamp = event.get('timestamp')
        if kind == 'item.completed' and item_kind == 'agent_message':
            add('message', 'Agent', item.get('text', ''), index, stamp)
        elif item_kind == 'command_execution':
            identity = item.get('id', str(index))
            if kind == 'item.started' or (kind == 'item.completed' and identity not in started):
                started.add(identity)
                add('tool', 'Bash', item.get('command', ''), index, stamp, identity)
            if kind == 'item.completed':
                add('result', f"Result · exit {item.get('exit_code', '?')}", item.get('aggregated_output', ''), index, stamp, identity)
        elif kind == 'item.completed' and item_kind == 'file_change':
            add('tool', 'File change', item.get('changes', []), index, stamp, item.get('id'))
        elif kind in {'error', 'turn.failed'}:
            add('error', 'Agent error', event, index, stamp)
        message = event.get('message', {})
        if not isinstance(message, dict):
            continue
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            identity = (message.get('id'), json.dumps(block, sort_keys=True))
            if message.get('id') and identity in seen:
                continue
            seen.add(identity)
            if kind == 'assistant' and block.get('type') == 'text':
                add('message', 'Agent', block.get('text', ''), index, stamp)
            elif kind == 'assistant' and block.get('type') == 'tool_use':
                tools[block['id']] = block['name']
                add('tool', block['name'], block.get('input', {}), index, stamp, block['id'])
            elif kind == 'user' and block.get('type') == 'tool_result':
                identity = block.get('tool_use_id')
                add('error' if block.get('is_error') else 'result', 'Result · ' + tools.get(identity, 'tool'),
                    block.get('content', ''), index, stamp, identity)
        if kind == 'result' and event.get('is_error'):
            add('error', 'Agent error', event.get('result', ''), index, stamp)
    return messages


def public_row(row):
    return {key: value for key, value in row.items() if not key.startswith('_')}


def snapshot_messages(path):
    """Group the readable stdout snapshot; canonical events replace it after collection."""
    messages = []
    current = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = event.get('message', '')
        heading = re.match(r'^(Agent|Tool · .+|Result · .+|Error · .+)$', content.strip())
        if heading:
            label = heading.group(1)
            kind = 'message' if label == 'Agent' else 'tool' if label.startswith('Tool') else 'error' if label.startswith('Error') else 'result'
            current = {'kind': kind, 'label': label, 'text': '', 'timestamp': event.get('timestamp'), 'event_index': len(messages)}
            messages.append(current)
        elif current:
            if re.search(r'^(Evaluating|Running verifier|Finished running agent|Agent run finished)', content):
                current = None
            else:
                current['text'] += content + '\n'
    for message in messages:
        message['text'] = redact(message['text'].strip())
    return messages


def trace_for(row):
    directory = row['_directory']
    paths = list((directory / 'outputs').rglob('trajectory.jsonl'))
    if len(paths) != 1:
        snapshot = directory / 'trace.jsonl'
        return {'run': public_row(row), 'messages': snapshot_messages(snapshot) if snapshot.exists() else [],
                'note': 'Saved stdout snapshot; refresh reads local files only. Canonical events appear after collection.'}
    events = []
    for line in paths[0].read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # An in-progress collector can leave an incomplete final line.
    summaries = list((directory / 'outputs').rglob('summary.json'))
    summary = read_json(summaries[0], {}) if summaries else {}
    messages = normalize_trace(events)
    if summary.get('task_prompt'):
        messages.insert(0, {'kind': 'prompt', 'label': 'Task instructions', 'text': redact(summary['task_prompt']), 'event_index': -1})
    patches = list((directory / 'outputs').glob('*/artifacts/workspace/patch.json/patch.json'))
    patch = read_json(patches[0]) if len(patches) == 1 else None
    return {'run': public_row(row), 'metadata': summary.get('metadata', {}), 'messages': messages, 'patch': patch,
            'note': 'Original public messages and tool events. Hidden reasoning is not reconstructed.'}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host', '').split(':')[0] not in {'127.0.0.1', 'localhost'}:
            self.send_error(403)
            return
        path = urlparse(self.path).path
        if path == '/api/experiment':
            rows = saved_runs()
            self.send_json({'runs': [public_row(r) for r in rows], 'leaderboard': leaderboard(rows),
                            'loaded_at': datetime.now(timezone.utc).isoformat()})
        elif path.startswith('/api/trace/'):
            identifier = path.removeprefix('/api/trace/')
            row = next((r for r in saved_runs() if r['run_id'] == identifier), None)
            if row:
                self.send_json(trace_for(row))
            else:
                self.send_error(404)
        elif path in {'/', '/app.js', '/style.css'}:
            filename = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[path]
            content_type = {'/': 'text/html', '/app.js': 'text/javascript', '/style.css': 'text/css'}[path]
            self.respond((STATIC / filename).read_bytes(), content_type)
        else:
            self.send_error(404)

    def send_json(self, data):
        self.respond(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), 'application/json')

    def respond(self, body, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Experiment viewer: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
