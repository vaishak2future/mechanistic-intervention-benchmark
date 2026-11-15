"""Grade only submitted weights in a clean verifier and write Harbor's reward."""

import argparse
import json
import shutil
from pathlib import Path

from eval import evaluate, load_tasks, result_report
from checkpoint import stage_checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    reports = Path("/logs/verifier")
    reports.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path("/tests/baseline.json").read_text())
    # Infrastructure errors must fail the verifier, not masquerade as a zero score.
    candidate = Path("/candidate")
    try:
        stage_checkpoint(Path("/base"), Path("/workspace/submission"), candidate)
    except ValueError as error:
        write_json(reports / "summary.json", {"valid_submission": False, "error": str(error)})
        (reports / "reward.txt").write_text(f"{-baseline['score']}\n")
        return
    options = argparse.Namespace(model=str(candidate), n=4, max_tokens=3072)
    submitted = result_report(evaluate(load_tasks(["/tests/holdout.json"]), options))
    write_json(reports / "submission.json", submitted)
    shutil.copyfile("/tests/baseline.json", reports / "baseline.json")
    summary = {"baseline": baseline["score"], "submission": submitted["score"],
               "improvement": submitted["score"] - baseline["score"]}
    write_json(reports / "summary.json", summary)
    (reports / "reward.txt").write_text(f"{summary['improvement']}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
