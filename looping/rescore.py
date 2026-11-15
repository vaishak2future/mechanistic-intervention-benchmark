"""Recompute the primary score from saved results, without running inference."""
import hashlib
import json
from pathlib import Path

from scoring import net_improvement, rewards

OUTPUT = Path(__file__).resolve().parents[1] / 'runs/looping-hour-comparison'


def main():
    for directory in sorted(OUTPUT.iterdir()):
        receipt = directory / 'collected.json'
        if not receipt.is_file():
            continue
        collected = json.loads(receipt.read_text())
        if collected['status'] != 'FINISHED' or not collected['scores']:
            continue
        original_path = directory / 'results.json'
        original = json.loads(original_path.read_text())
        evaluation = original['evaluation_results']['repair-looping']
        diagnostics = evaluation['rewards']
        result = net_improvement(diagnostics)
        result['source_sha256'] = hashlib.sha256(original_path.read_bytes()).hexdigest()
        result['source'] = 'Original Valkyrie verifier rewards; no agent or inference rerun.'
        result['valkyrie_updated'] = False
        result['valkyrie_blocker'] = 'Original dataset has no metric script; rescore cannot select a new version.'
        (directory / 'net-improvement.json').write_text(json.dumps(result, indent=2) + '\n')
        collected['scores'] = rewards(diagnostics)
        collected['primary_score'] = result['score']
        collected['scoring_version'] = result['scoring_version']
        receipt.write_text(json.dumps(collected, indent=2) + '\n')
        print(f"{directory.name}: {result['score']:.2f} ({result['newly_solved']} fixed, {result['regressions']} broken)")


if __name__ == '__main__':
    main()
