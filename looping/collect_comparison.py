"""Retain hosted scores and traces, and clean up only this comparison's compute."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
import modal
from scoring import rewards

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'runs/looping-hour-comparison'
TERMINAL={'FINISHED','ERROR','STOPPED'}


def command(*args):
    result=subprocess.run(['valkyrie',*args],capture_output=True,text=True,timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def collect(run):
    directory=OUTPUT/f"{run['name']}-{run['repeat']}"
    task=json.loads(command('run','task',run['run_id'],run.get('task_id','repair-looping'),'--format','json'))
    save(directory/'task.json',task)
    if task['status'] not in TERMINAL:
        return False
    command('run','results',run['run_id'],'--path',str(directory/'results.json'),'--yes')
    (directory/'trace.jsonl').write_text(command('run','logs',run['run_id'],'--format','jsonl'))
    command('run','outputs',run['run_id'],'--output-dir',str(directory/'outputs'))
    result=task.get('evaluation_result') or {}
    saved={**run,'status':task['status'],'scores':result.get('rewards'),
           'error':task.get('error_message'),'timings':task.get('task_breakdown')}
    if saved['scores'] and 'reward' not in saved['scores'] and task['status']=='FINISHED':
        saved['scores'] = rewards(saved['scores'])
    save(directory/'collected.json',saved)
    print('SAVED',run['name'],run['repeat'],saved['scores'] or saved['error'],flush=True)
    return True


def cleanup(app_id,finished_ids):
    removed=[]
    for sandbox in modal.Sandbox.list(app_id=app_id):
        owner=sandbox.get_tags().get('run-id')
        if owner in finished_ids:
            sandbox.terminate(wait=True)
            removed.append(sandbox.object_id)
    return removed


def summary(runs):
    rows=[json.loads((OUTPUT/f"{r['name']}-{r['repeat']}"/'collected.json').read_text()) for r in runs]
    spread={}
    for model in ['sol','astra','opus','sonnet']:
        group=[row for row in rows if row['name']==model and row['scores']]
        spread[model]={metric:{'values':[r['scores'][metric] for r in group],
                              'mean':statistics.mean(r['scores'][metric] for r in group),
                              'min':min(r['scores'][metric] for r in group),
                              'max':max(r['scores'][metric] for r in group)}
                       for metric in ['reward','repair','specificity','transfer']
                       if all(metric in r['scores'] for r in group)} if group else {}
    save(OUTPUT/'summary.json',{'runs':rows,'spread':spread,'note':'Two trials per model; observed spread, not a precise uncertainty estimate.'})


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--docent-python',type=Path)
    args=parser.parse_args()
    OUTPUT=args.output.resolve()
    runs=json.loads((OUTPUT/'runs.json').read_text())
    app=modal.App.lookup('benchmark-service')
    save(OUTPUT/'provider.json',{'app_id':app.app_id})
    previous={}
    removed=[]
    while True:
        try:
            status=json.loads(command('run','status','--ids',','.join(r['run_id'] for r in runs),'--format','json'))
        except Exception as error:
            print('RETRY status',str(error)[:300],flush=True)
            time.sleep(30)
            continue
        save(OUTPUT/'status.json',status)
        entries={entry['id']:entry for entry in status['entries']}
        ready=[]
        for run in runs:
            entry=entries[run['run_id']]
            state=(entry['status'],str(entry['task_state_counts']))
            if previous.get(run['run_id'])!=state:
                print(run['name'],run['repeat'],*state,flush=True)
                previous[run['run_id']]=state
            receipt=OUTPUT/f"{run['name']}-{run['repeat']}"/'collected.json'
            if entry['status'] in TERMINAL and not receipt.exists():
                ready.append(run)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for run,future in [(run,pool.submit(collect,run)) for run in ready]:
                try:future.result()
                except Exception as error:print('RETRY collection',run['name'],run['repeat'],str(error)[:300],flush=True)
        finished={run['run_id'] for run in runs if (OUTPUT/f"{run['name']}-{run['repeat']}"/'collected.json').exists()}
        removed.extend(cleanup(app.app_id,finished))
        if args.docent_python and finished:
            export=subprocess.run([str(args.docent_python),str(ROOT/'looping/export_comparison.py'),
                                   '--output',str(OUTPUT)],capture_output=True,text=True,timeout=180)
            if export.returncode:
                print('RETRY Docent',export.stderr[-500:],flush=True)
        if len(finished)==len(runs):
            summary(runs)
            remaining=list(modal.Sandbox.list(app_id=app.app_id))
            owned=[sandbox.object_id for sandbox in remaining if sandbox.get_tags().get('run-id') in finished]
            assert not owned,'Comparison sandbox remains active.'
            if not remaining:
                app.stop()
            save(OUTPUT/'cleanup.json',{'app_id':app.app_id,'remaining_comparison_sandboxes':owned,
                                      'terminated_leftovers':removed,'app_stopped':not remaining})
            print('All eight traces and scores saved; comparison compute stopped.',flush=True)
            return
        time.sleep(30)


if __name__=='__main__':main()
