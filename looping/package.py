"""Package the frozen pilot as a one-hour Harbor task; private data stays in tests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--minutes',type=int,default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh package directory.')
    manifest = json.loads((args.baseline_run/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen input changed: {name}')
    task = args.output/'stop-reasoning-loops'
    shutil.copytree(ROOT/'harbor/looping',task,ignore=shutil.ignore_patterns('__pycache__'))
    config=task/'task.toml'
    config.write_text(config.read_text().replace('timeout_sec = 3600',f'timeout_sec = {args.minutes * 60}'))
    for name in ['patch.py','prompts.py','evaluate.py','metrics.py','scoring.py','dev.json']:
        shutil.copyfile(ROOT/'looping'/name,task/'environment'/name)
    shutil.copyfile(ROOT/'eval.py',task/'environment/eval.py')
    instruction=(ROOT/'looping/TASK.txt').read_text().replace('You have ten minutes',f'You have {args.minutes} minutes')
    instruction += '\nUse the available hour to investigate and validate your intervention. Save a valid patch early,\nthen refine it. Do not access external models, datasets or other agents; the supplied model and dev set\nare your research materials. The primary score is net improvement: newly solved failures minus\nnewly broken controls, divided by the number of baseline failures, times 100, with a floor of zero.\nRepair, specificity and transfer are diagnostics. Doing nothing earns zero.\n'
    instruction=instruction.replace('Use the available hour',f'Use the available {args.minutes} minutes')
    (task/'instruction.md').write_text(instruction)
    (task/'environment/TASK.txt').write_text(instruction)
    shutil.copyfile(ROOT/'private/looping/holdout.json',task/'tests/holdout.json')
    shutil.copyfile(args.baseline_run/'baseline.json',task/'tests/baseline.json')
    shutil.copyfile(args.baseline_run/'manifest.json',task/'tests/provenance.json')
    (task/'tests/test.sh').chmod(0o755)
    print(task)


if __name__=='__main__':main()
