"""GPU-only readiness checks and baseline generation, before the agent's clock."""

import argparse
import json
import time
from pathlib import Path

from eval import evaluate, load_tasks, result_report


def warm_engine():
    from vllm import LLM, SamplingParams

    engine = LLM("/base", dtype="bfloat16", max_model_len=8192, seed=0,
                 enable_sleep_mode=True)
    output = engine.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0))
    if not output[0].outputs[0].token_ids:
        raise RuntimeError("vLLM failed to generate.")
    return engine


def prepare():
    started = time.monotonic()
    output = Path("/prepared")
    output.mkdir(exist_ok=True)
    engine = warm_engine()
    options = argparse.Namespace(model="/base", n=4, max_tokens=3072)
    results = evaluate(load_tasks(["/task/tasks.json"]), options, model=engine)
    (output / "baseline.json").write_text(json.dumps(result_report(results), indent=2) + "\n")
    # Verify the resident-engine handoff can release and reacquire GPU allocations.
    engine.sleep(level=1)
    engine.wake_up()
    from vllm import SamplingParams
    engine.generate(["Ready"], SamplingParams(max_tokens=4, temperature=0))
    import torch
    import vllm
    (output / "readiness.json").write_text(json.dumps({
        "ready": True, "gpu": torch.cuda.get_device_name(), "vllm": vllm.__version__,
        "seconds": time.monotonic() - started,
        "checks": ["vllm_generate", "baseline", "sleep_wake"],
        "source": json.loads(Path("/base/source.json").read_text()),
        "compile_cache_files": sum(1 for p in Path("/cache").rglob("*") if p.is_file()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
