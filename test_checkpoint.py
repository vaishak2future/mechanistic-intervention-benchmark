"""Check sharded transport and the weight-only grading boundary without a GPU."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint = load_module('checkpoint', 'harbor/mobilellm/tests/checkpoint.py')
submission = load_module('submission', 'harbor/mobilellm/environment/submit.py')


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base = self.root / 'base'
        self.base.mkdir()
        for name in ['config.json', 'tokenizer.json', 'model.safetensors.index.json', 'custom.py']:
            (self.base / name).write_text('trusted')
        self.weights = self.root / 'model.safetensors'
        self.weights.write_bytes(b'full model weights')
        self.parts = self.root / 'submission'
        submission.submit(self.weights, self.parts, part_bytes=5)

    def stage(self):
        checkpoint.stage_checkpoint(self.base, self.parts, self.root / 'candidate')

    def test_round_trip_uses_only_weights_and_trusted_metadata(self):
        (self.parts / 'config.json').write_text('untrusted')
        self.stage()
        candidate = self.root / 'candidate'
        self.assertEqual({p.name for p in candidate.iterdir()},
                         {'config.json', 'tokenizer.json', 'model.safetensors'})
        self.assertEqual((candidate / 'model.safetensors').read_bytes(), self.weights.read_bytes())
        self.assertEqual((candidate / 'config.json').read_text(), 'trusted')

    def test_corrupted_part_is_rejected(self):
        (self.parts / 'weights.01').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.stage()

    def test_missing_part_is_rejected(self):
        (self.parts / 'weights.02').unlink()
        with self.assertRaisesRegex(ValueError, 'nonregular'):
            self.stage()

    def test_symlink_part_is_rejected(self):
        part = self.parts / 'weights.01'
        part.unlink()
        part.symlink_to(self.weights)
        with self.assertRaisesRegex(ValueError, 'nonregular'):
            self.stage()

    def test_invalid_manifests_are_rejected(self):
        for manifest in [[], {'parts': 11}, {'parts': True},
                         {'parts': 1, 'bytes': -1},
                         {'parts': 1, 'bytes': 1, 'sha256': '../escape'}]:
            with self.subTest(manifest=manifest):
                (self.parts / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    self.stage()

    def test_empty_weights_and_overwriting_submission_are_rejected(self):
        with self.assertRaises(FileExistsError):
            submission.submit(self.weights, self.parts)
        self.weights.write_bytes(b'')
        with self.assertRaises(ValueError):
            submission.submit(self.weights, self.root / 'empty')


if __name__ == '__main__':
    unittest.main()
