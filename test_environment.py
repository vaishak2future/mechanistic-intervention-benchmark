"""Check the submission boundary without requiring a GPU or Docker daemon."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace

from environment import ROOT, isolation, stage_checkpoint, start


class EnvironmentTests(unittest.TestCase):
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

    def test_training_mounts_only_base_and_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "config.json").write_text("{}")
            with patch("environment.docker", return_value="sha256:image") as docker:
                start(Namespace(base=base, seconds=60, image="lab", name="agent"))
            arguments = docker.call_args.args
            mounts = [arguments[i + 1] for i, arg in enumerate(arguments) if arg == "--mount"]
            self.assertEqual(len(mounts), 2)
            self.assertIn(f"type=bind,src={base.resolve()},dst=/base,readonly", mounts)
            self.assertEqual(arguments[-3:], ("sha256:image", "sleep", "60"))
            self.assertNotIn(str(ROOT), mounts)
            self.assertIn("--read-only", isolation())
            self.assertEqual(arguments[arguments.index("--network") + 1], "none")

    def test_build_context_excludes_private_files(self):
        allowed = [line for line in (ROOT / ".dockerignore").read_text().splitlines() if line.startswith("!")]
        self.assertEqual(set(allowed), {"!Dockerfile", "!eval.py", "!TASK.txt", "!tasks.json"})


if __name__ == "__main__":
    unittest.main()
