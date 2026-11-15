"""Render public agent events for the TUI; the harness saves raw events separately."""
import json


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(block.get('text', '') for block in content
                         if isinstance(block, dict) and block.get('type') == 'text')
    return json.dumps(content, ensure_ascii=False)


def tool_call(name, arguments):
    if name.lower() == 'bash' and 'command' in arguments:
        return f"Tool · {name}\n$ {arguments['command']}"
    return f"Tool · {name}\n{json.dumps(arguments, ensure_ascii=False, indent=2)}"


class LiveMessages:
    """Keep messages, calls and results in stream order without exposing thinking blocks."""

    def __init__(self):
        self.started = set()
        self.seen = set()

    def render(self, line):
        try:
            event = json.loads(line)
        except ValueError:
            return line.rstrip()
        if not isinstance(event, dict):
            return ''
        kind = event.get('type')
        item = event.get('item', {})
        if kind == 'item.completed' and item.get('type') == 'agent_message':
            return 'Agent\n' + item.get('text', '')
        if item.get('type') == 'command_execution':
            identifier = item.get('id')
            if kind == 'item.started':
                self.started.add(identifier)
                return tool_call('bash', {'command': item.get('command', '')})
            if kind == 'item.completed':
                call = '' if identifier in self.started else tool_call('bash', {'command': item.get('command', '')}) + '\n'
                return call + f"Result · {identifier} · exit {item.get('exit_code')}\n" + item.get('aggregated_output', '')
        if kind == 'item.completed' and item.get('type') == 'file_change':
            return 'File changes\n' + json.dumps(item.get('changes', []), ensure_ascii=False, indent=2)
        if kind in {'error', 'turn.failed'}:
            return 'Error\n' + json.dumps(event, ensure_ascii=False)
        if kind not in {'assistant', 'user'}:
            return ''
        message = event.get('message', {})
        rendered = []
        for block in message.get('content', []):
            if not isinstance(block, dict):
                continue
            identity = (message.get('id'), json.dumps(block, sort_keys=True))
            if message.get('id') and identity in self.seen:
                continue
            self.seen.add(identity)
            if kind == 'assistant' and block.get('type') == 'text':
                rendered.append('Agent\n' + block['text'])
            elif kind == 'assistant' and block.get('type') == 'tool_use':
                rendered.append(tool_call(block['name'], block.get('input', {})))
            elif kind == 'user' and block.get('type') == 'tool_result':
                label = 'Error' if block.get('is_error') else 'Result'
                rendered.append(f"{label} · {block.get('tool_use_id')}\n" + text_content(block.get('content', '')))
        return '\n\n'.join(rendered)

    def emit(self, line):
        rendered = self.render(line)
        if rendered:
            print(rendered + '\n', flush=True)
