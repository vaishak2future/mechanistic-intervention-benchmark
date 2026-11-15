"""Prepare on Modal, run the timed agent, then grade in a fresh H100 sandbox."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import modal
from patch import validate_patch
from run_agent import run_agent

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/looping-sol-clean-10m'
CODEX = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex'


def save(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2)+'\n')


def read(sandbox, path):
    return sandbox.filesystem.read_text(path)


def write(sandbox, path, text):
    sandbox.filesystem.write_text(text, path)


def execute(sandbox, command, log):
    process = sandbox.exec('bash', '-lc', command, timeout=1200)
    output = process.stdout.read() + process.stderr.read()
    (RUN / log).write_text(output)
    process.wait()
    if process.returncode:
        raise RuntimeError(f'Remote command failed; inspect {log}')


from metrics import scores


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--calibrate-only', action='store_true')
    options = parser.parse_args()
    RUN = options.run.resolve()
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'state.json').exists():
        raise RuntimeError('Use a fresh run directory.')
    sources = ['looping/patch.py', 'looping/prompts.py', 'looping/evaluate.py',
               'looping/dev.json', 'looping/TASK.txt', 'eval.py', 'private/looping/holdout.json']
    save('manifest.json', {
        'agent_model': 'gpt-6.1-sol', 'reasoning_effort': 'high', 'budget_seconds': 600,
        'base_revision': 'cdd21efc69cc89fa9608cfbad863f9eebb9dc9fb',
        'sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}})
    (RUN/'task.txt').write_text((ROOT/'looping/TASK.txt').read_text())
    image = modal.Image.from_id('im-4jG8f7AjuepDH3Nu7wqezd')
    for name in ['patch.py','prompts.py','evaluate.py','dev.json','TASK.txt']:
        image = image.add_local_file(ROOT/'looping'/name, '/pilot/'+name, copy=True)
    image = image.add_local_file(ROOT/'eval.py','/pilot/eval.py',copy=True).env({'PYTHONPATH':'/pilot', 'VLLM_ALLOW_INSECURE_SERIALIZATION':'1'})
    app=modal.App.lookup('mobilellm-looping-pilot',create_if_missing=True)
    active=[]
    def create(label):
        sandbox=modal.Sandbox.create('sleep','infinity',app=app,image=image,gpu='H100',
            cpu=8,memory=32768,timeout=2400,block_network=True,workdir='/workspace')
        active.append(sandbox)
        save(label+'-sandbox.json',{'id':sandbox.object_id})
        print(label,sandbox.object_id,flush=True)
        return sandbox
    def stop(sandbox):
        sandbox.terminate(wait=True)
        active.remove(sandbox)
    started=time.time()
    try:
        # Hidden cases only enter operator-owned sandboxes, never the agent's.
        baseline=create('baseline')
        write(baseline,'/workspace/hidden.json',(ROOT/'private/looping/holdout.json').read_text())
        execute(baseline,'python /pilot/evaluate.py --grade /workspace/hidden.json --report /workspace/baseline.json','baseline.log')
        report=json.loads(read(baseline,'/workspace/baseline.json'))
        save('baseline.json',report['baseline'])
        stop(baseline)
        controls=[r for r in report['baseline']['tasks'] if r['group']=='specificity' and r['correct']]
        loops=[r for r in report['baseline']['tasks'] if r['group']=='repair' and r['loop']['detected']]
        print('Baseline',scores(report),flush=True)
        if options.calibrate_only:
            return
        repair_count = sum(row['group']=='repair' for row in report['baseline']['tasks'])
        control_count = sum(row['group']=='specificity' for row in report['baseline']['tasks'])
        transfer = [row for row in report['baseline']['tasks'] if row['group']=='transfer']
        if (len(controls)<3 or len(loops)<3 or len(loops)!=repair_count
                or len(controls)!=control_count or not all(row['loop']['detected'] for row in transfer)):
            raise RuntimeError('Pilot needs calibration before running an agent: insufficient controls or confirmed repetitive nontermination cases.')
        sandbox=create('agent')
        write(sandbox,'/workspace/patch.json','{"edits":[]}\n')
        execute(sandbox,'nohup python /pilot/evaluate.py --serve > /workspace/server.log 2>&1 < /dev/null &','start-server.log')
        ready=False
        for _ in range(180):
            try:
                ready=json.loads(read(sandbox,'/prepared/looping-ready.json'))['ready']
                if ready:break
            except Exception:pass
            time.sleep(5)
        if not ready:
            (RUN/'server.log').write_text(read(sandbox,'/workspace/server.log'))
            raise RuntimeError('Dev evaluator did not become ready.')
        save('dev-baseline.json',json.loads(read(sandbox,'/prepared/looping-dev-baseline.json')))
        clock=time.time()
        state={'status':'running','sandbox_id':sandbox.object_id,'started':clock,'deadline':clock+600,
               'agent':{'model':'gpt-6.1-sol','reasoning_effort':'high'}}
        save('state.json',state)
        print('READY; agent clock started',flush=True)
        exit_code=run_agent(RUN,CODEX)
        elapsed=time.time()-clock
        raw=read(sandbox,'/workspace/patch.json')
        if len(raw.encode())>16384:raise ValueError('Submission exceeds 16KB.')
        patch=validate_patch(json.loads(raw))
        save('patch.json',patch)
        stop(sandbox)
        state = json.loads((RUN/'state.json').read_text())
        state.update(status='submitted',elapsed_seconds=elapsed,runner_exit_code=exit_code)
        save('state.json',state)
        grader=create('grader')
        write(grader,'/workspace/hidden.json',(ROOT/'private/looping/holdout.json').read_text())
        write(grader,'/workspace/patch.json',json.dumps(patch))
        execute(grader,'python /pilot/evaluate.py --grade /workspace/hidden.json --patch /workspace/patch.json --report /workspace/report.json','grading.log')
        report=json.loads(read(grader,'/workspace/report.json'))
        save('report.json',report)
        stop(grader)
        summary={'agent':state['agent'],'agent_seconds':elapsed,'scores':scores(report),'patch':patch,
                 'end_to_end_seconds':time.time()-started,'compute_stopped':True,'workspace_deleted':True}
        save('summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        for sandbox in active:
            sandbox.terminate(wait=True)
        app.stop()
        save('cleanup.json',{'all_created_sandboxes_terminated':True,'app_stopped':True,'persistent_workspace_created':False})


if __name__=='__main__':main()
