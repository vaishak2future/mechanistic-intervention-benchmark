"""Cloud lifecycle tests without weights or GPU allocation."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from modal_environment import baseline_identity, cached_baseline, PUBLIC_FILES, build_image, create_sandbox, grade, start, submit


class ModalEnvironmentTests(unittest.TestCase):
    def test_image_downloads_and_prepares_remotely(self):
        modal = MagicMock()
        image = MagicMock()
        modal.Image.from_registry.return_value = image
        for method in ('entrypoint', 'pip_install', 'run_commands', 'env', 'workdir', 'add_local_file'):
            getattr(image, method).return_value = image
        build_image(modal, 'owner/model', 'revision', None)
        image.add_local_dir.assert_not_called()
        calls = image.run_commands.call_args_list
        self.assertIn('download_model.py', calls[-2].args[0])
        self.assertEqual(calls[-1].kwargs['gpu'], 'H100')
        self.assertNotIn('tasks_holdout.json', PUBLIC_FILES)
        self.assertNotIn('watchdog.py', PUBLIC_FILES)
        self.assertNotIn('train.py', PUBLIC_FILES)
        self.assertNotIn('train.jsonl', PUBLIC_FILES)

    def test_sandbox_is_offline_and_bounded(self):
        modal = MagicMock()
        create_sandbox(modal, 'app', 'image', 'H100', 60)
        options = modal.Sandbox.create.call_args.kwargs
        self.assertTrue(options['block_network'])
        self.assertEqual(options['timeout'], 60)
        self.assertEqual(options['memory'], (32768, 32768))
        self.assertEqual(options['gpu'], 'H100')
        self.assertNotIn('secrets', options)

    def test_clock_starts_only_after_live_warmup_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'state.json').write_text(json.dumps({'status':'built','image_id':'original'}))
            sandbox = MagicMock()
            sandbox.object_id = 'sandbox'
            sandbox.filesystem.read_text.side_effect = ['{"score":0.5}', '{"ready":true}']
            timer_module = MagicMock()
            timer_module.enforce_deadline.spawn.return_value.object_id = 'timer'
            events = []
            def now():
                self.assertEqual(sandbox.filesystem.read_text.call_count, 2)
                self.assertEqual(events, ['launch', 'ready'])
                return 1000
            with patch.dict('sys.modules', {'watchdog':timer_module}), patch('modal_environment.create_sandbox', return_value=sandbox), patch('modal_environment.run_command', side_effect=lambda *args: events.append('launch' if not events else 'ready')), patch('modal_environment.time.time', side_effect=now):
                start(MagicMock(), Namespace(run=run, seconds=60, agent_model="gpt-6.1-sol", agent_reasoning="high"))
            state = json.loads((run / 'state.json').read_text())
            self.assertEqual(state['deadline'], 1060)
            self.assertEqual(state['agent']['model'], 'gpt-6.1-sol')
            self.assertEqual(state['agent']['reasoning_effort'], 'high')
            timer_module.enforce_deadline.spawn.assert_called_once_with('sandbox', 1060)

    def test_failed_readiness_does_not_start_timer(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'state.json').write_text('{"status":"built","image_id":"original"}')
            sandbox = MagicMock()
            timer = MagicMock()
            with patch.dict('sys.modules', {'watchdog':timer}), patch('modal_environment.create_sandbox', return_value=sandbox), patch('modal_environment.run_command', side_effect=RuntimeError('warmup failed')):
                with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
                    start(MagicMock(), Namespace(run=run, seconds=60, agent_model="gpt-6.1-sol", agent_reasoning="high"))
            sandbox.terminate.assert_called_once_with(wait=True)
            timer.enforce_deadline.spawn.assert_not_called()

    def test_submit_keeps_weights_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            modal = MagicMock()
            sandbox = modal.Sandbox.from_id.return_value
            state = {'status':'running','sandbox_id':'training','deadline':2000}
            with patch('modal_environment.time.time', return_value=1000), patch('modal_environment.run_command') as command:
                submit(modal, Path(directory), state)
            command.assert_called_once_with(sandbox, ['sync','/workspace'], 60)
            sandbox.filesystem.copy_to_local.assert_not_called()
            sandbox.terminate.assert_called_once_with(wait=True)
            self.assertEqual(state['status'], 'submitted')

    def test_late_submit_fails_and_terminates(self):
        with tempfile.TemporaryDirectory() as directory:
            modal = MagicMock()
            state = {'status':'running','sandbox_id':'training','deadline':0}
            with self.assertRaises(ValueError):
                submit(modal, Path(directory), state)
            modal.Sandbox.from_id.return_value.terminate.assert_called_once_with(wait=True)
            self.assertEqual(state['status'], 'failed')

    def test_grading_uses_original_image_and_readonly_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            holdout = run / 'hidden.json'
            holdout.write_text('[]')
            modal = MagicMock()
            sandboxes = [MagicMock(), MagicMock()]
            for sandbox, score in zip(sandboxes, [0.25, 0.75]):
                sandbox.filesystem.read_text.return_value = json.dumps({'score':score, 'tasks':[]})
            state = {'status':'submitted','image_id':'original','gpu':'H100','volume_name':'workspace'}
            with patch('modal_environment.create_sandbox', side_effect=sandboxes), patch('modal_environment.run_command'):
                grade(modal, Namespace(run=run, holdout=holdout, seconds=60, baseline_cache=run / 'cache'), state)
            self.assertEqual(modal.Image.from_id.call_count, 2)
            modal.Image.from_id.assert_called_with('original')
            for sandbox in sandboxes:
                sandbox.filesystem.write_bytes.assert_called_once_with(b'[]', '/workspace/holdout.json')
                sandbox.terminate.assert_called_once_with(wait=True)
            modal.Volume.from_name.return_value.with_mount_options.assert_called_with(read_only=True)
            summary = json.loads((run / 'reports/summary.json').read_text())
            self.assertEqual(summary['improvement'], 0.5)

    def test_baseline_cache_reuses_only_matching_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            options = Namespace(baseline_cache=Path(directory), seconds=60)
            state = {"image_id": "original", "gpu": "H100"}
            with patch('modal_environment.grade_checkpoint', return_value={"score": 0.25}) as evaluate:
                cached_baseline(None, options, state, b'[]')
                cached_baseline(None, options, state, b'[]')
                self.assertEqual(evaluate.call_count, 1)
                cached_baseline(None, options, state, b'[{}]')
                cached_baseline(None, options, {**state, "image_id": "new"}, b'[]')
                cached_baseline(None, options, {**state, "gpu": "A100"}, b'[]')
                self.assertEqual(evaluate.call_count, 4)

    def test_failed_baseline_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            options = Namespace(baseline_cache=Path(directory), seconds=60)
            with patch('modal_environment.grade_checkpoint', side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):
                    cached_baseline(None, options, {"image_id":"image", "gpu":"H100"}, b'[]')
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cached_grade_runs_only_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            holdout = run / 'holdout.json'
            holdout.write_text('[]')
            options = Namespace(run=run, holdout=holdout, seconds=60, baseline_cache=run / 'cache')
            state = {"status":"submitted", "image_id":"image", "gpu":"H100"}
            with patch('modal_environment.grade_checkpoint', return_value={"score":0.25,"tasks":[]}):
                cached_baseline(None, options, state, b'[]')
            with patch('modal_environment.grade_checkpoint', return_value={"score":0.75,"tasks":[]}) as evaluate:
                grade(None, options, state)
                evaluate.assert_called_once_with(None, state, b'[]', 'submission', 60)
            self.assertEqual(json.loads((run / 'reports/summary.json').read_text())['improvement'], 0.5)


if __name__ == '__main__':
    unittest.main()
