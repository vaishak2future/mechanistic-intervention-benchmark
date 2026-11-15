"""Launch two independent one-hour runs per model after validation passes."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'runs/looping-hour-comparison'
VALIDATION = OUTPUT
MODELS = [
    ('sol','gpt-6.1-sol','codex-looping-v3','OPENAI_API_KEY','openai-api-key'),
    ('astra','gpt-6-astra','codex-looping-v3','OPENAI_API_KEY','openai-api-key'),
    ('opus','anthropic/claude-opus-5-5','claude-looping-v3','ANTHROPIC_API_KEY','anthropic-api-key'),
    ('sonnet','anthropic/claude-sonnet-5-5','claude-looping-v3','ANTHROPIC_API_KEY','anthropic-api-key'),
]
DATASET = 'mechanistic-intervention-benchmark/mechanistic-intervention-benchmark@submission-45m'


def require_validation():
    for name in ['codex','claude']:
        check = json.loads((VALIDATION/f'{name}-check.json').read_text())
        assert check['status']=='FINISHED' and check['evaluation_result']['rewards']['reward']==1
    check = json.loads((ROOT/'runs/looping-valkyrie-validation/task.json').read_text())
    assert check['status']=='FINISHED', 'Hosted verifier has not finished.'
    assert check['evaluation_result']['rewards']=={'repair':0.0,'specificity':1.0,'transfer':0.0}
    pilot_dir = ROOT/'runs/looping-sol-clean-serial-10m'
    pilot = json.loads((pilot_dir/'state.json').read_text())
    cleanup = json.loads((pilot_dir/'cleanup.json').read_text())
    assert pilot['status']=='submitted' and cleanup['all_created_sandboxes_terminated']
    from patch import validate_patch
    validate_patch(json.loads((pilot_dir/'patch.json').read_text()))
    assert all(row['ok'] for row in json.loads((VALIDATION/'api-preflight.json').read_text()))


def launch(item):
    spec, repeat = item
    name, model, agent, variable, secret = spec
    directory = OUTPUT/f'{name}-{repeat}'
    directory.mkdir(parents=True,exist_ok=True)
    receipt = directory/'run.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['dataset'] != DATASET or saved['agent'] != agent:
            raise ValueError('Existing receipt belongs to another experiment; choose a fresh output directory.')
        return saved
    command = ['valkyrie','run','start','--agent',agent,'--dataset',DATASET,
               '--model',model,'--provider','modal','-s',variable,secret,
               '--concurrency','1','--label',f'{OUTPUT.name}-{name}-{repeat}']
    result = subprocess.run(command,capture_output=True,text=True,timeout=120)
    (directory/'start.txt').write_text(result.stdout+result.stderr)
    match = re.search(r'Run ID:\s*([a-f0-9-]{36})',result.stdout)
    if result.returncode or not match:
        raise RuntimeError(f'Launch failed for {name}-{repeat}; inspect its start.txt before retrying.')
    data = {'run_id':match.group(1),'name':name,'model':model,'repeat':repeat,
            'agent':agent,'reasoning_effort':'high','budget_seconds':2700,'dataset':DATASET,
            'task_id':'stop-reasoning-loops','scoring_version':'bounded-net-improvement-v2'}
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data),flush=True)
    return data


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    OUTPUT = parser.parse_args().output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    require_validation()
    checks = json.loads((ROOT/'runs/submission-validation/checks.json').read_text())
    assert all(check['passed'] for check in checks) and len(checks)==3, 'Hosted validation must pass first.'
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs=list(pool.map(launch,[(model,repeat) for model in MODELS for repeat in [1,2]]))
    (OUTPUT/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')


if __name__=='__main__':main()
