import json
import unittest

from looping.live_messages import LiveMessages


class LiveMessageTests(unittest.TestCase):
    def test_codex_messages_calls_and_results(self):
        display = LiveMessages()
        events = [
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Testing a candidate.'}},
            {'type': 'item.started', 'item': {'type': 'command_execution', 'id': '1', 'command': 'python evaluate.py'}},
            {'type': 'item.completed', 'item': {'type': 'command_execution', 'id': '1', 'command': 'python evaluate.py', 'exit_code': 0, 'aggregated_output': '3/6 correct'}},
        ]
        text = '\n'.join(display.render(json.dumps(event)) for event in events)
        self.assertEqual(text.count('$ python evaluate.py'), 1)
        self.assertLess(text.index('Testing a candidate.'), text.index('$ python'))
        self.assertIn('3/6 correct', text)

    def test_claude_public_text_deduplicated_and_thinking_omitted(self):
        display = LiveMessages()
        event = {'type': 'assistant', 'message': {'id': 'm1', 'content': [
            {'type': 'thinking', 'thinking': 'private'},
            {'type': 'text', 'text': 'Checking retention.'},
            {'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'python check.py'}},
        ]}}
        text = display.render(json.dumps(event))
        self.assertIn('Checking retention.', text)
        self.assertIn('$ python check.py', text)
        self.assertNotIn('private', text)
        self.assertEqual(display.render(json.dumps(event)), '')

    def test_plain_errors_and_structured_tool_results(self):
        display = LiveMessages()
        self.assertEqual(display.render('connection failed\n'), 'connection failed')
        event = {'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 't1', 'content': [{'type': 'text', 'text': 'done'}]}
        ]}}
        self.assertEqual(display.render(json.dumps(event)), 'Result · t1\ndone')
