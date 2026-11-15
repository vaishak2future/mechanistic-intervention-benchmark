"""Check live capture and its remote-only command boundary without paid compute."""

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_command import record
from modal_agent_tool import execute
from run_agent import agent_command, capture, run_agent


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.state = {"status": "running", "started_at": time.time(), "deadline": time.time() + 60,
                      "agent": {"model": "explicit-test-model", "reasoning_effort": "high"}}
        (self.run / 'state.json').write_text(json.dumps(self.state))

    def events(self):
        return [json.loads(line) for line in (self.run / 'trace.jsonl').read_text().splitlines()]

    def test_messages_are_saved_before_next_event_and_interleave_with_tools(self):
        message = {"type": "item.completed", "item": {"id": "one", "type": "agent_message", "text": "Checking the model."}}

        def stream():
            yield json.dumps(message) + '\n'
            self.assertEqual(self.events()[0]['message'], 'Checking the model.')
            self.assertIn('Checking the model.', (self.run / 'trace.html').read_text())
            record(self.run, {"event": "action", "action_id": "tool", "command": ["ls", "/base"]})
            yield json.dumps(message) + '\n'  # Replayed completion is not another message.
            yield json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "DO NOT SAVE"}}) + '\n'
            record(self.run, {"event": "result", "action_id": "tool", "stdout": "weights", "exit_code": 0})
            yield json.dumps({"type": "item.completed", "item": {"id": "two", "type": "agent_message", "text": "Ready."}}) + '\n'

        process = type('Process', (), {"stdout": stream(), "wait": lambda self: 0})()
        self.assertEqual(capture(self.run, process), 0)
        self.assertEqual([e['event'] for e in self.events()], ['agent_message', 'action', 'result', 'agent_message'])
        self.assertNotIn('DO NOT SAVE', (self.run / 'trace.jsonl').read_text())

    def test_runner_launches_explicit_model_with_remote_bridge(self):
        command = agent_command(self.run, self.state, self.run)
        self.assertEqual(command[command.index('--model') + 1], 'explicit-test-model')
        self.assertIn('--json', command)
        self.assertIn('features.shell_tool=false', command)
        self.assertIn('mcp_servers.h100.tools.command.approval_mode="approve"', command)
        self.assertIn('web_search="disabled"', command)
        self.assertTrue(any('modal_agent_tool.py' in part for part in command))

    def test_command_arguments_go_to_wrapper_without_host_shell(self):
        with patch('modal_agent_tool.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'ok', '')) as execute_process:
            result = execute(self.run, ['bash', '-lc', 'echo "$HOME"'])
        self.assertEqual(result['stdout'], 'ok')
        arguments = execute_process.call_args.args[0]
        self.assertEqual(arguments[-3:], ['bash', '-lc', 'echo "$HOME"'])
        self.assertNotIn('shell', execute_process.call_args.kwargs)

    def test_expired_run_cannot_execute_commands(self):
        self.state['deadline'] = 0
        (self.run / 'state.json').write_text(json.dumps(self.state))
        with patch('modal_agent_tool.subprocess.run') as execute_process:
            with self.assertRaises(ValueError):
                execute(self.run, ['ls'])
            execute_process.assert_not_called()

    def test_tool_rejection_is_visible_even_before_remote_execution(self):
        rejected = {"type": "item.completed", "item": {"type": "mcp_tool_call", "status": "failed",
                    "error": {"message": "Approval required"}}}
        process = type('Process', (), {"stdout": [json.dumps(rejected)], "wait": lambda self: 0})()
        capture(self.run, process)
        self.assertEqual(self.events()[0]['event'], 'tool_rejected')
        self.assertIn('Approval required', self.events()[0]['message'])

    def test_runner_process_is_captured_without_cloud_resources(self):
        executable = self.run / 'fake-codex'
        executable.write_text('#!/usr/bin/env python3\nimport json\nprint(json.dumps({"type":"item.completed","item":{"id":"one","type":"agent_message","text":"Fixture final message"}}),flush=True)\n')
        executable.chmod(0o755)
        self.assertEqual(run_agent(self.run, str(executable)), 0)
        self.assertEqual([e['event'] for e in self.events()], ['runner_started', 'agent_message', 'runner_finished'])
        state = json.loads((self.run / 'state.json').read_text())
        self.assertEqual(state['agent']['runner'], 'codex-exec')


if __name__ == '__main__':
    unittest.main()
