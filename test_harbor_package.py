"""Check that the portable task preserves the data and grading boundary."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from package_harbor import package


class HarborPackageTests(unittest.TestCase):
    def test_only_verifier_receives_hidden_data_and_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout = root / "hidden.json"
            holdout.write_text('[{"id":"private-only-sentinel"}]')
            cache = root / "cache"
            cache.mkdir()
            entry = {"identity": {"image_id": "im-4jG8f7AjuepDH3Nu7wqezd", "gpu": "H100",
                      "holdout_sha256": hashlib.sha256(holdout.read_bytes()).hexdigest()},
                     "report": {"score": 0.25, "tasks": []}}
            (cache / "baseline.json").write_text(json.dumps(entry))
            task = package(root / "output", holdout, cache)
            self.assertEqual((task / "tests/holdout.json").read_bytes(), holdout.read_bytes())
            self.assertTrue((task / "tests/baseline.json").is_file())
            for path in (task / "environment").iterdir():
                self.assertNotIn("private-only-sentinel", path.read_text())
            self.assertFalse((task / "environment/train.py").exists())
            self.assertNotIn("baseline.json", (task / "environment/Dockerfile").read_text())
            with self.assertRaises(ValueError):
                package(root / "output", holdout, cache)

    def test_changed_holdout_requires_new_measured_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout = root / "hidden.json"
            holdout.write_text("[]")
            cache = root / "cache"
            cache.mkdir()
            (cache / "old.json").write_text(json.dumps({"identity": {
                "image_id": "im-4jG8f7AjuepDH3Nu7wqezd", "gpu": "H100", "holdout_sha256": "old"}}))
            with self.assertRaisesRegex(ValueError, "No measured baseline"):
                package(root / "output", holdout, cache)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
