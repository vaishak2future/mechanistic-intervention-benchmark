"""Reconstruct only the approved patch, then measure all three outcomes."""
import json
from pathlib import Path
import sys
sys.path.insert(0, '/pilot')
from evaluate import Evaluator
from metrics import scores
from patch import validate_patch
from scoring import rewards


def main():
    output = Path('/logs/verifier')
    output.mkdir(parents=True, exist_ok=True)
    path = Path('/workspace/patch.json')
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16384:
            raise ValueError('Submit one regular JSON file no larger than 16KB.')
        patch = validate_patch(json.loads(path.read_text()))
    except (ValueError, OSError) as error:
        (output/'summary.json').write_text(json.dumps({'valid_submission':False,'error':str(error)}))
        (output/'reward.json').write_text(json.dumps({'reward':0,'valid_submission':0}))
        return
    tasks = json.loads(Path('/tests/holdout.json').read_text())
    evaluator = Evaluator()
    baseline = evaluator.evaluate(tasks, {'edits':[]})
    expected = json.loads(Path('/tests/baseline.json').read_text())
    if baseline['tasks'] != expected['tasks']:
        (output/'baseline-mismatch.json').write_text(json.dumps(baseline,indent=2))
        raise RuntimeError('Baseline differs from calibration; do not report a comparable score.')
    submission = evaluator.evaluate(tasks, patch)
    report = {'baseline':baseline, 'submission':submission}
    measured = scores(report)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'summary.json').write_text(json.dumps({'valid_submission':True,'patch':patch,'scores':measured},indent=2))
    reward = rewards({name:measured[name]['submission'] for name in ['repair','specificity','transfer']})
    (output/'reward.json').write_text(json.dumps(reward,indent=2))
    print(json.dumps({'patch':patch,'scores':measured},indent=2))


if __name__ == '__main__':
    main()
