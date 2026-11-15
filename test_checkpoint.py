"""Check the weight-only submission boundary without a GPU."""

import tempfile
import unittest
from pathlib import Path
import importlib.util

module_path = Path(__file__).parent / "harbor/mobilellm/tests/checkpoint.py"
spec = importlib.util.spec_from_file_location("checkpoint", module_path)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
stage_checkpoint = checkpoint.stage_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_only_weights_and_trusted_metadata_reach_grading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            for name in ["config.json", "tokenizer.json", "model.safetensors.index.json", "custom.py"]:
                (base / name).write_text("trusted")
            weights = root / "model.safetensors"
            weights.write_bytes(b"test weights")
            destination = root / "candidate"
            stage_checkpoint(base, weights, destination)
            self.assertEqual({p.name for p in destination.iterdir()},
                             {"config.json", "tokenizer.json", "model.safetensors"})
            self.assertEqual((destination / "model.safetensors").read_bytes(), b"test weights")

    def test_empty_and_symlink_submissions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "empty"
            weights.touch()
            link = root / "link"
            link.symlink_to(weights)
            for invalid in [weights, link, root / "missing"]:
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    stage_checkpoint(root, invalid, root / "candidate")


if __name__ == "__main__":
    unittest.main()
