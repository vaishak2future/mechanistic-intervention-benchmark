"""Run only in a Modal image build; pin and download the starting model there."""

import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--revision", default="main")
    options = parser.parse_args()
    revision = HfApi().model_info(options.model, revision=options.revision).sha
    snapshot_download(
        options.model, revision=revision, local_dir="/base",
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"],
    )
    Path("/base/source.json").write_text(json.dumps({"model": options.model, "revision": revision}))


if __name__ == "__main__":
    main()
